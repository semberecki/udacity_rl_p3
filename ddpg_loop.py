from collections import deque

import numpy as np
import torch

from ddpg_agent import experience, ReplayBuffer
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

def ddpg(env, agent1, agent2, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.999,
         train_mode=True, update_network=True, score_list_len=100, checkpoints_dir="checkpoints",
         score_required=30, display_frequency=1):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=score_list_len)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    scores_agent1= []
    scores_agent2 = []
    memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=0)

    brain_name = env.brain_names[0]
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
        state = env_info.vector_observations            # get the current state
        episode_average_score = 0
        agents_number=len(env_info.agents)
        current_scores = np.zeros(agents_number)
        while True:
            if not update_network:
                eps = 0.0
            full_state = state.flatten()
            action1 = agent1.act(torch.from_numpy(state[0]).float(), eps)
            action2 = agent2.act(torch.from_numpy(state[1]).float(), eps)
            actions = np.concatenate((action1, action2))[np.newaxis, ...]
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations   # get the next state
            reward = env_info.rewards                   # get the reward
            reward_array =  np.array(reward)
            done = env_info.local_done                  # see if episode has finished
            if update_network:
                transition= experience(state, full_state, [action1, action2], reward_array,
                                       next_state, next_state.flatten(), done )

                memory.push(transition)
                if len(memory) > BATCH_SIZE:
                    experiences1 = memory.sample()
                    experiences2 = memory.sample()
                    agent1.learn(experiences1, agent1)
                    agent2.learn(experiences2, agent2)

            state = next_state
            episode_average_score += np.mean(reward)
            current_scores += reward_array

            if np.any(done):
                break

        scores_window.append(current_scores.max())
        scores_agent1.append(current_scores[0])
        scores_agent2.append(current_scores[1])
        if update_network:
            eps = max(eps_end, eps_decay*eps) # decrease epsilon

        agent1.logger.add_scalar('agent0/mean_episode_rewards', float(current_scores[0]), i_episode)
        agent2.logger.add_scalar('agent1/mean_episode_rewards', float(current_scores[1]), i_episode)
        print('\rEpisode {}\tAverage Score: {:.5f}\tCurrent Score: {} eps {:.2f}'.format(
            i_episode, np.mean(scores_window), np.array_str(current_scores), eps), end="")

        if i_episode % display_frequency == 0:
            print('\rEpisode {}\tAverage Score: {:.5f}\tScore: {} eps {:.5f}'.format(
                i_episode, np.mean(scores_window), np.array_str(current_scores), eps))


        if np.mean(scores_window)>=score_required and update_network:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.5f}'.format(i_episode-score_list_len, np.mean(scores_window)))
            break

        if i_episode==n_episodes:
            print('\nEpisode finished {:d} episodes.\tAverage Score: {:.5f}'.format(i_episode-score_list_len, np.mean(scores_window)))


    return scores