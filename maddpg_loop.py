from collections import deque
from datetime import datetime

import numpy as np
import os
import torch

from maddpg_agent import experience, ReplayBuffer
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size

NOISE_GAIN_INITIAL=5
NOISE_GAIN_END=0
NOISE_DECAY=0.001


import os
def maddpg(env, agent1, agent2, n_episodes=1000,
           train_mode=True, update_network=True, score_list_len=100, checkpoints_dir="checkpoints",
           score_required=0.5, display_frequency=10, save_temp_checkpoints=True, print_scores=True):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores_window = deque(maxlen=score_list_len)  # last 100 scores

    date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    scores_agent1= []
    scores_agent2 = []
    memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed=0)
    avg_max=0
    avg_max_episode=0
    current_max=0
    current_max_episode=0

    noise_gain=NOISE_GAIN_INITIAL
    if update_network:
        agent1.noise_gain=noise_gain
        agent2.noise_gain=noise_gain

    brain_name = env.brain_names[0]
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name] # reset the environment
        state = env_info.vector_observations            # get the current state
        episode_average_score = 0
        agents_number=len(env_info.agents)
        current_scores = np.zeros(agents_number)
        while True:
            full_state = state.flatten()
            action1 = agent1.act(torch.from_numpy(state[0]).float(), add_noise=update_network)
            action2 = agent2.act(torch.from_numpy(state[1]).float(), add_noise=update_network)
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

                    # Update epsilon noise value
                    noise_gain -= NOISE_DECAY
                    if noise_gain < NOISE_GAIN_END:
                        noise_gain= NOISE_GAIN_END

                    agent1.noise_gain=noise_gain
                    agent2.noise_gain=noise_gain

            state = next_state
            episode_average_score += np.mean(reward)
            current_scores += reward_array

            if np.any(done):
                break

        scores_window.append(current_scores.max())
        scores_agent1.append(current_scores[0])
        scores_agent2.append(current_scores[1])
        avg_value = np.mean(scores_window)

        agent1.logger.add_scalar('agent0/mean_episode_rewards', float(current_scores[0]), i_episode)
        agent2.logger.add_scalar('agent1/mean_episode_rewards', float(current_scores[1]), i_episode)

        if print_scores:
            print('\rEpisode {}\tAverage Score: {:.5f}\tCurrent Score: {:.5f} {:.5f}'.format(
                i_episode, avg_value, current_scores[0], current_scores[1]), end="")

            if i_episode % display_frequency == 0:
                print('\rEpisode {}\tAverage Score: {:.5f}\tCurrent Score: {:.5f} {:.5f}'.format(
                    i_episode, avg_value, current_scores[0], current_scores[1]))

        if avg_value >=score_required and update_network:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-score_list_len, avg_value))

            save_dict_list = get_checkpoints_content(agent1, agent2)
            torch.save(save_dict_list, os.path.join(checkpoints_dir, f'checkpoint.pt'))
            break

        if i_episode %100==0 and i_episode>0 and save_temp_checkpoints:
            save_dict_list = get_checkpoints_content(agent1, agent2)
            torch.save(save_dict_list, os.path.join(checkpoints_dir,  f'episode-{i_episode}-{avg_value:.3f}.pt'))

        if current_scores.max()>current_max and current_scores.max() >0 and save_temp_checkpoints:
            if current_max_episode != 0:
                os.remove(os.path.join(checkpoints_dir, f'checkpoint_current_max-{current_max_episode}-{current_max:.3f}.pt'))
            current_max = current_scores.max()
            current_max_episode = i_episode
            save_dict_list = get_checkpoints_content(agent1, agent2)
            torch.save(save_dict_list, os.path.join(checkpoints_dir, f'checkpoint_current_max-{current_max_episode}-{current_max:.3f}.pt'))

        if avg_value>avg_max and avg_value>0 and save_temp_checkpoints:
            if avg_max_episode != 0:
                os.remove(os.path.join(checkpoints_dir, f'checkpoint_max_avg-{avg_max_episode}-{avg_max:.3f}.pt'))
            avg_max =avg_value
            avg_max_episode = i_episode
            save_dict_list = get_checkpoints_content(agent1, agent2)
            torch.save(save_dict_list, os.path.join(checkpoints_dir, f'checkpoint_max_avg-{avg_max_episode}-{avg_max:.3f}.pt'))

        if i_episode==n_episodes and print_scores:
            print('\nEpisode finished {:d} episodes.\tAverage Score: {:.5f}'.format(i_episode-score_list_len, avg_value))


    return (scores_agent1, scores_agent2)

def get_checkpoints_content(agent1, agent2):
    save_dict1 = {'actor_params': agent1.actor_local.state_dict(),
                  'actor_optim_params': agent1.actor_optimizer.state_dict(),
                  'critic_params': agent1.critic_local.state_dict(),
                  'critic_optim_params': agent1.critic_optimizer.state_dict()}
    save_dict2 = {'actor_params': agent2.actor_local.state_dict(),
                  'actor_optim_params': agent2.actor_optimizer.state_dict(),
                  'critic_params': agent2.critic_local.state_dict(),
                  'critic_optim_params': agent2.critic_optimizer.state_dict()}
    save_dict_list = [save_dict1, save_dict2]
    save_dict_list.append(save_dict_list)
    return save_dict_list