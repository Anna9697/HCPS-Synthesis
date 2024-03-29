import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from collections import deque
from env.grid_world import Env
from Learner.agent_adapt import Agent
from Learner.agent_reinforce import Reinforce

from Learner.LDBA import LDBA, LDBAUtil

def train(agent_h, agent_c, env, LTL_model, gamma, gammaB, nn_num):
    # controler = "human"

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []

    time_start = time.time()
    max_score = -float('inf')

    for i_episode in range(num_episodes):
        eps = agent_c.epsilon_annealing(i_episode, max_eps_episode, min_eps)
        score, temp_q = run_episode(agent_h, agent_c, env, LTL_model, gamma, gammaB, eps)
        if score > max_score:
            agent_h.save_models(0)
            agent_c.save('dir_chk/HCPS-LTL/grid2', 'GridWorldMax')
            max_score = score

        scores_deque.append(score)
        scores_array.append(score)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Episode: {:5} Score: {:5}  Avg.Score: {:.8f}, eps-greedy: {:5.2f}, Time: {:02}:{:02}:{:02}'. \
                format(i_episode, score, avg_score, eps, dt // 3600, dt % 3600 // 60, dt % 60))

        if len(scores_deque) == scores_deque.maxlen:
            threshold = 0.65
            if np.mean(scores_deque) >= threshold:
                print('\n Environment solved in {:d} episodes!\t Average Score: {:.2f}'. \
                      format(i_episode, np.mean(scores_deque)))
                break

    return scores_array, avg_scores_array  # , loss_array_plot

def run_episode(agent_h, agent_c, env, LTL_model, gamma, gammaB, episode):
    """Play an epsiode and train

    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        eps (float): eps-greedy for exploration

    Returns:
        int: reward earned in this episode
    """
    state = env.reset()
    temp_q = 0
    # pre_reward = None
    nstate = np.append(state, temp_q)
    total_reward = 0
    HorM = 0
    action = 0
    j = 0
    # print(state)

    for i in range(100):
        # reward = 0
        # print(i)
        # if temp_q == 1:
        #     env.lander.color1 = (0, 255, 255, 1)
        # if temp_q == 2:
        #     env.lander.color1 = (255, 255, 0, 1)

        # if in_critical_states(state):
        HorM = agent_c.get_action(nstate, episode)
        if HorM == 0:
            action = get_CPS_action(state, action)
        else:
            action = agent_h.choose_action(state)

        # env.render()

        next_state, next_reward, done = env.step(action)
        # landed = not env.lander.awake
        # statement = [done, env.game_over, landed]

        next_q, reward, Gamma = excution_LTL(env, next_state, str(temp_q), LTL_model, gamma, gammaB)  # 自定义LTL自动机状态转换函数
        # if done and landed and next_q == 1:
        #     next_reward += 200
        #     # total_reward += 2
        #     print("Success!!!!!")
        # if pre_reward is not None:
        #     reward += 0.01 * (Gamma * next_reward - pre_reward)
        # reward += LTL_reward


        agent_h.store_reward(reward)

        total_reward += reward

        nstate = np.append(state, temp_q)
        nnext_state = np.append(next_state, next_q)

        # if HorM == 1:
        #     agent_h.step(nstate, action, reward, nnext_state, done, Gamma)
        agent_c.step(nstate, HorM, reward, nnext_state, done, Gamma)

        # pre_reward = next_reward
        state = next_state
        temp_q = next_q

        # if done:
        #     break

    # j = (j + 1) % 10
    # if j == 0:
    agent_h.learn()

    return total_reward, temp_q

def in_critical_states(state):
    in_critical = False
    if (np.sqrt(state[2] * state[2] + state[3] * state[3]) > 0.15 and (np.sqrt(state[2]*state[2] + state[3]*state[3]) < 0.3)) or (abs(state[0] > 0.15) and abs(state[0] < 0.25)) or (abs(state[5] > 0.5) and abs(state[5] < 1.5)):
        in_critical = True
    return in_critical

def excution_LTL(env, state, temp_q, LTL_model, gamma, gammaB):
    next_q = temp_q
    # lable = 'd'
    reward = 0
    Gamma = gamma

    # test2
    # a:到达a & !c
    # b:未到达b & !c
    # c:陷阱
    # d:其他
    # if state in env.
    label = env.get_label(state)
    next_q = random.choice(LTL_model.get_nextLocations(str(temp_q), label))
    if next_q in LTL_model.accept_locations:
        reward += 1 - gammaB
        Gamma = gammaB
    # print(next_q)

    # print(label, next_q)


    return int(next_q), reward, Gamma


def get_CPS_action(state, action):
    machine_policy = np.load(file="resources/MDP/gridWorld4-5/machine_policy.npy", allow_pickle=True)
    action = machine_policy[state[1], state[0]]
    return action

def main(env, agent_h, agent_c, LTL_model, gamma, gammaB, nn_num):
   scores, avg_scores = train(agent_h, agent_c, env, LTL_model, gamma, gammaB, nn_num)

   print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

   fig = plt.figure()
   ax = fig.add_subplot(111)
   # plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
   plt.plot(np.arange(100, len(avg_scores) + 1), avg_scores[99:], label="Avg on 100 episodes")

   plt.legend(bbox_to_anchor=(1.05, 1))
   plt.ylabel('Score')
   plt.xlabel('Episodes #')
   plt.show()

   agent_h.save_models(1)
   agent_c.save('dir_chk/HCPS-LTL/grid2', 'GridWorld')

if __name__ == '__main__':
    LTLpath = "resources/LDBA/LTL_gridworld1.json"
    LTL_model = LDBAUtil.LDBA_Util(LTLpath)
    LDBA.make_model(LTL_model, "resources/LDBA/", "LTL_gridworld1")
    # save_path = 'dir_chk/Reinforce_HPS/1/'

    nn_num = len(LTL_model.locations)

    random.seed(0)

    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device: ', device)

    # BATCH_SIZE = 64
    # TAU = 0.005  # 1e-3   # for soft update of target parameters
    gamma = 0.9999
    gammaB = 0.99
    # LEARNING_RATE = 5e-5
    TARGET_UPDATE = 4

    num_episodes = 30000
    print_every = 2
    hidden_dim = 16  ## 64 ## 16
    min_eps = 0.01
    max_eps_episode = 500

    env = Env()
    # env.seed(0)
    # env = gym.wrappers.Monitor(env, directory="monitors", force=True)

    space_dim = 2  # n_spaces
    action_dim = 4  # n_actions  get_action
    print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

    # threshold = env.spec.reward_threshold
    # print('threshold: ', threshold)

    agent_h = Reinforce(alpha=0.0005, state_dim=space_dim, action_dim=action_dim,
                      fc1_dim=16, fc2_dim=16, ckpt_dir='dir_chk/Reinforce_HPS/GridHCPS-2/', gamma=0.99)
    agent_c = Agent(space_dim + 1, 2, seed=0)

    agent_h.load_models('dir_chk/Reinforce_HPS/GridWorld2/', 1)
    temp_q = 0

    main(env, agent_h, agent_c, LTL_model, gamma, gammaB, nn_num)
    print()