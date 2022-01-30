import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99            # discount factor
TAU = 0.05              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


experience = namedtuple("Experience",field_names=["obs", "obs_full", "actions", "reward",
                                                  "next_obs", "next_obs_full", "done"])

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self,state_size, action_size, random_seed, agent_number,logger):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(2*state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(2*state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_gain = None

        # Replay memory
        self.agent_number=agent_number
        self.logger = logger
        self.iter = 0


    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = state.to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_gain*self.noise.sample()
        return np.clip(action, -1, 1)

    def act_target(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = state.to(device)
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_gain*self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, opponent, gamma=GAMMA):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # states, actions, rewards, next_states, dones = experiences
        agent_states, states, actions, rewards, agent_next_states, next_states, dones =  experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(agent_next_states[self.agent_number])
        actions_next_opponent= opponent.act_target(agent_next_states[int(not self.agent_number)])
        actions_next_opponent = torch.from_numpy(actions_next_opponent).float().to(device)
        if self.agent_number == 0:
            actions_next = torch.cat((actions_next, actions_next_opponent), dim=1)
        else:
            actions_next = torch.cat((actions_next_opponent, actions_next), dim=1)

        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)

        Q_targets = rewards[self.agent_number].view(-1,1) + (gamma * Q_targets_next * (1 - dones[self.agent_number].view(-1,1)))
        # Compute critic loss

        Q_expected = self.critic_local(states, torch.cat((actions), dim=1))
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(agent_states[self.agent_number])
        actions_pred_opponent = opponent.act(agent_states[int(not self.agent_number)])
        actions_pred_opponent = torch.from_numpy(actions_pred_opponent).float().to(device)

        if self.agent_number == 0:
            actions_pred = torch.cat((actions_pred, actions_pred_opponent.detach()), dim=1)
        else:
            actions_pred = torch.cat((actions_pred_opponent.detach(), actions_pred), dim=1)

        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        self.logger.add_scalars('agent%i/losses' % self.agent_number,
                           {'critic loss': cl,
                            'actor_loss': al},
                           self.iter)
        self.iter += 1

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.12, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size

        self.seed = random.seed(seed)

    def push(self, transition):
        """Add a new experience to memory."""
        self.memory.append(transition)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        obs = np.stack([e.obs for e in experiences if e is not None])
        obs = np.swapaxes(np.array(obs), 0, 1)
        obs = [torch.from_numpy(agent_obs).float().to(device) for agent_obs in obs]
        obs_full = torch.from_numpy(np.vstack([e.obs_full for e in experiences if e is not None])).float().to(device)

        actions = np.stack([e.actions for e in experiences if e is not None])
        actions = np.swapaxes(np.array(actions), 0, 1)
        actions = [torch.from_numpy(agent_actions).float().to(device) for agent_actions in actions]

        reward = np.stack([e.reward for e in experiences if e is not None])
        reward = np.swapaxes(reward, 0, 1)
        reward = [torch.from_numpy(agent_reward).float().to(device) for agent_reward in reward]

        next_obs = np.stack([e.obs for e in experiences if e is not None])
        next_obs = np.swapaxes(np.array(next_obs), 0, 1)
        next_obs = [torch.from_numpy(agent_obs).float().to(device) for agent_obs in next_obs]
        next_obs_full = torch.from_numpy(np.vstack([e.next_obs_full for e in experiences if e is not None])).float().to(device)

        done = np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)
        done = np.swapaxes(done, 0, 1)
        done = [torch.from_numpy(agent_done).float().to(device) for agent_done in done]

        return (obs, obs_full, actions, reward, next_obs, next_obs_full, done)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

