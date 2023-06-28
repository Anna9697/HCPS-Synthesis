import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from collections import deque
from Learner.agent_reinforce import Reinforce

from Learner.LDBA import LDBA, LDBAUtil

def train(agent, env, LTL_model, gamma, gammaB, nn_num):
    controler = "human"

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    time_start = time.time()
    max_score = -float('inf')

    for i_episode in range(num_episodes):
        # eps = agent.epsilon_annealing(i_episode, max_eps_episode, min_eps)
        score, temp_q, LTL_score, env_score = run_episode(agent, env, LTL_model, gamma, gammaB)
        if score > max_score:
            agent.save_models(0)
            max_score = score

        scores_deque.append(score)
        scores_array.append(score)
        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Episode: {:5} Score: {:5}  Avg.Score: {:.2f}, LTL_Score: {:5}, Env_Score: {:5}, Time: {:02}:{:02}:{:02}'. \
                  format(i_episode, score, avg_score, LTL_score, env_score, dt // 3600, dt % 3600 // 60, dt % 60))

        if len(scores_deque) == scores_deque.maxlen:
            threshold = 20
            if np.mean(scores_deque) >= threshold:
                print('\n Environment solved in {:d} episodes!\t Average Score: {:.2f}'. \
                      format(i_episode, np.mean(scores_deque)))
                break

    return scores_array, avg_scores_array  # , loss_array_plot

def run_episode(agent, env, LTL_model, gamma, gammaB):
    """Play an epsiode and train

    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        eps (float): eps-greedy for exploration

    Returns:
        int: reward earned in this episode
    """
    controler = "human"
    state = env.reset()
    total_reward = 0
    temp_q = 0
    nstate = np.append(state, temp_q)
    pre_reward = None
    total_LTL = 0
    total_env = 0

    while True:
        # env.render(mode=controler)
        reward = 0
        if temp_q == 1:
            env.lander.color1 = (0, 255, 255, 1)
        if temp_q == 2:
            env.lander.color1 = (255, 255, 0, 1)
        action = agent.choose_action(state)
        next_state, next_reward, done, _ = env.step(action)
        # agent.store_reward(next_reward)
        landed = not env.lander.awake
        statement = [done, env.game_over, landed]



        # done = False
        # if env.game_over or abs(next_state[0]) >= 1.0:
        #     # done = True
        #     next_reward -= 100
        # if env.curr_step >= 600 and env.lander.awake:
        #     # done = True
        #     next_reward -= 100
        # if not env.lander.awake:
        #     # done = True
        #     next_reward += 100
        #     print("landed!!!")
        #
        next_q, LTL_reward, Gamma = LTL_model.execution(next_state, str(temp_q), statement, gamma, gammaB)
        # if done and landed and next_q == 1:
        #     next_reward += 20
        reward = get_env_reward(env, next_state, pre_reward, next_reward)
        # if pre_reward is not None:
        #     reward += 0.1 * (1.0 * next_reward - pre_reward)
        total_env += reward

        # reward += LTL_reward
        agent.store_reward(reward)
        total_LTL += LTL_reward

        nstate = np.append(state, temp_q)
        nnext_state = np.append(next_state, next_q)
        # agent.step(nstate, action, reward, nnext_state, done, Gamma)
        total_reward += (reward + LTL_reward)
        pre_reward = next_reward
        state = next_state
        temp_q = next_q

        if done:
            break
    agent.learn()

    return total_reward, temp_q, total_LTL, total_env

def get_env_reward(env, next_state, pre_reward, next_reward):
    reward = 0
    if env.game_over or abs(next_state[0]) >= 1.0:
        # done = True
        next_reward -= 100
    if env.curr_step >= 600 and env.lander.awake:
        # done = True
        next_reward -= 100
    if not env.lander.awake:
        # done = True
        next_reward += 100
        print("landed!!!")
    if pre_reward is not None:
        reward += 0.1 * (1.0 * next_reward - pre_reward)
    return reward


def main(env, agent, LTL_model, gamma, gammaB, nn_num):
   scores, avg_scores = train(agent, env, LTL_model, gamma, gammaB, nn_num)

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
    LTLpath = "resources/LDBA/LTL_model6.json"
    LTL_model = LDBAUtil.LDBA_Util(LTLpath)
    LDBA.make_model(LTL_model, "resources/LDBA/", "LTL_model6")
    save_path = 'dir_chk/Reinforce_HPS/2/'

    nn_num = len(LTL_model.locations)
    random.seed(0)

    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device: ', device)

    BATCH_SIZE = 64
    TAU = 0.005  # 1e-3   # for soft update of target parameters
    gamma = 0.999999
    gammaB = 0.999
    LEARNING_RATE = 5e-5
    TARGET_UPDATE = 4

    num_episodes = 30000
    print_every = 1
    hidden_dim = 64  ## 64 ## 16
    # min_eps = 0.01
    # max_eps_episode = 500

    env = gym.make('LunarLander-v2')
    env.seed(0)
    env = gym.wrappers.Monitor(env, directory="monitors", force=True)

    space_dim = env.observation_space.shape[0]  # n_spaces
    action_dim = env.action_space.n  # n_actions  get_action
    print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

    threshold = env.spec.reward_threshold
    print('threshold: ', threshold)

    agent = Reinforce(alpha=0.0005, state_dim=space_dim, action_dim=action_dim,
                      fc1_dim=256, fc2_dim=256, ckpt_dir=save_path, gamma=0.99)
        # Reinforce(space_dim + 1, action_dim, seed=0)
    temp_q = 0

    main(env, agent, LTL_model, gamma, gammaB, nn_num)
    print()