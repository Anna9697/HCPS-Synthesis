import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from collections import deque
from env.grid_world import Env
from Learner.agent_reinforce import Reinforce

from Learner.LDBA import LDBA, LDBAUtil

def train(agent, env):
    # controler = "human"

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    time_start = time.time()
    max_score = -float('inf')

    for i_episode in range(num_episodes):
        # eps = agent.epsilon_annealing(i_episode, max_eps_episode, min_eps)
        score = run_episode(agent, env)
        if score > max_score:
            agent.save_models(0)
            max_score = score

        scores_deque.append(score)
        scores_array.append(score)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Episode: {:5} Score: {:5}  Avg.Score: {:.8f}, Time: {:02}:{:02}:{:02}'. \
                  format(i_episode, score, avg_score, dt // 3600, dt % 3600 // 60, dt % 60))

        if len(scores_deque) == scores_deque.maxlen:
            threshold = 0.02
            if np.mean(scores_deque) >= threshold:
                print('\n Environment solved in {:d} episodes!\t Average Score: {:.2f}'. \
                      format(i_episode, np.mean(scores_deque)))
                break

    return scores_array, avg_scores_array  # , loss_array_plot

def run_episode(agent, env):
    """Play an epsiode and train

    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        eps (float): eps-greedy for exploration

    Returns:
        int: reward earned in this episode
    """
    # controler = "human"
    state = env.reset()
    total_reward = 0
    temp_q = 0
    # nstate = np.append(state, temp_q)
    pre_reward = None
    total_LTL = 0
    total_env = 0

    for i in range(100):
        # env.render()
        # reward = 0
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)

        agent.store_reward(reward)

        total_reward += reward
        state = next_state

        # if done:
        #     break
    agent.learn()

    return total_reward

def main(env, agent):
   scores, avg_scores = train(agent, env)

   print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

   fig = plt.figure()
   ax = fig.add_subplot(111)
   # plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
   plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="Avg on 100 episodes")

   plt.legend(bbox_to_anchor=(1.05, 1))
   plt.ylabel('Score')
   plt.xlabel('Episodes #')
   plt.show()

   agent.save_models(1)

if __name__ == '__main__':
    # LTLpath = "resources/LDBA/LTL_model6.json"
    # LTL_model = LDBAUtil.LDBA_Util(LTLpath)
    # LDBA.make_model(LTL_model, "resources/LDBA/", "LTL_model6")
    save_path = 'dir_chk/Reinforce_HPS/GridWorld2/'

    # nn_num = len(LTL_model.locations)
    random.seed(0)

    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device: ', device)

    BATCH_SIZE = 64
    # TAU = 0.005  # 1e-3   # for soft update of target parameters

    # gammaB = 0.999
    LEARNING_RATE = 5e-5
    TARGET_UPDATE = 4

    num_episodes = 60000
    print_every = 1
    hidden_dim = 16  ## 64 ## 16
    # min_eps = 0.01
    # max_eps_episode = 500

    env = Env()
    # env.seed(0)
    # env = gym.wrappers.Monitor(env, directory="monitors", force=True)

    space_dim = 2  # n_spaces
    action_dim = 4  # n_actions  get_action
    print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

    # threshold = env.spec.reward_threshold
    # print('threshold: ', threshold)

    agent = Reinforce(alpha=0.0005, state_dim=space_dim, action_dim=action_dim,
                      fc1_dim=16, fc2_dim=16, ckpt_dir=save_path, gamma=0.99)
        # Reinforce(space_dim + 1, action_dim, seed=0)
    temp_q = 0

    main(env, agent)
    print()