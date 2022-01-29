from tensorboardX import SummaryWriter
from unityagents import UnityEnvironment
import matplotlib.pyplot as plt


import torch
import numpy as np
import shutil
import os

# from tensorboardX import SummaryWriter
from maddpg_agent import Agent
from maddpg_loop import maddpg


def print_demo(env, action_size):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
    num_agents = len(env_info.agents)

    for i in range(1, 6):                                      # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

def setup():
    env = UnityEnvironment(file_name="./Tennis_Linux/Tennis.x86_64")


    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    return env, state_size, action_size

def plot_results(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores[0])+1), scores[0], label="agent1")
    plt.plot(np.arange(1, len(scores[1])+1), scores[1], label="agent2")
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    run_demo=False
    eval_only=False

    log_path = os.getcwd()+"/log"

    if  os.path.exists(log_path):
        shutil.rmtree(log_path)

    logger = SummaryWriter(log_dir=log_path)

    env, state_size, action_size = setup()
    if run_demo:
        print_demo(env, action_size)

    if eval_only:
        checkpoint_path="checkpoints/checkpoint.pt"

        agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=0, agent_number=0, logger=logger)
        agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=0, agent_number=1, logger=logger)

        agent1.actor_local.load_state_dict(torch.load(checkpoint_path)[0]['actor_params'])
        agent1.actor_optimizer.load_state_dict(torch.load(checkpoint_path)[0]['actor_optim_params'])
        agent1.critic_local.load_state_dict(torch.load(checkpoint_path)[0]['critic_params'])
        agent1.critic_optimizer.load_state_dict(torch.load(checkpoint_path)[0]['critic_optim_params'])

        agent2.actor_local.load_state_dict(torch.load(checkpoint_path)[1]['actor_params'])
        agent2.actor_optimizer.load_state_dict(torch.load(checkpoint_path)[1]['actor_optim_params'])
        agent2.critic_local.load_state_dict(torch.load(checkpoint_path)[1]['critic_params'])
        agent2.critic_optimizer.load_state_dict(torch.load(checkpoint_path)[1]['critic_optim_params'])

        scores = maddpg(env, agent1, agent2, n_episodes=10, train_mode=True,update_network=False, score_list_len=100)
        plot_results(scores)

    else:
        agent1 = Agent(state_size=state_size, action_size=action_size, random_seed=0, agent_number=0, logger=logger)
        agent2 = Agent(state_size=state_size, action_size=action_size, random_seed=0, agent_number=1, logger=logger)
        scores = maddpg(env, agent1, agent2, n_episodes=10, checkpoints_dir="checkpoints")
        plot_results(scores)


    env.close()

