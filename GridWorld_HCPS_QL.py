import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from collections import deque
from env.grid_world import Env
from Learner.agent_QL import QL
from Learner.agent_reinforce import Reinforce

from Learner.LDBA import LDBA, LDBAUtil


def train(agent_h, agent_c, env, LTL_model, gamma, gammaB, nn_num):
    # controler = "human"

    scores_deque = deque(maxlen=100)
    scores_array = []
    avg_scores_array = []
    success_array = deque(maxlen=30)

    time_start = time.time()
    max_score = -float('inf')

    for i_episode in range(num_episodes):
        eps = epsilon_annealing(i_episode, max_eps_episode, min_eps)
        state = env.reset()
        temp_q = 0
        nstate = np.append(state, temp_q)
        agent_c.reset(nstate, 'd')
        score, temp_q, success = run_episode(agent_h, agent_c, env, LTL_model, gamma, gammaB, eps)
        if score > max_score:
            agent_h.save_models(0)
            agent_c.save('dir_chk/HCPS-LTL/grid_QL', 'GridWorldMax')
            max_score = score

        scores_deque.append(score)
        scores_array.append(score)
        success_array.append(success)

        avg_score = np.mean(scores_deque)
        success_rate = np.mean(success_array)
        avg_scores_array.append(avg_score)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Episode: {:5} Score: {:5}  Avg.Score: {:.8f}, Success_Rate: {:.2f}, eps-greedy: {:5.2f}, Time: {:02}:{:02}:{:02}'. \
                format(i_episode, score, avg_score, success_rate, eps, dt // 3600, dt % 3600 // 60, dt % 60))

        if len(scores_deque) == scores_deque.maxlen:
            threshold = 0.90
            if success_rate >= threshold:
                print('\n Environment solved in {:d} episodes!\t Average Score: {:.2f}'. \
                      format(i_episode, np.mean(scores_deque)))
                break

    return scores_array, avg_scores_array  # , loss_array_plot

def run_episode(agent_h, agent_c, env, LTL_model, gamma, gammaB, episode):
    state = env.reset()
    label = env.get_label(state)
    # self.LDBA.reset()
    # QL.Q_initial_value = Q_initial_value
    temp_q = 0
    nstate = np.append(state, temp_q)
    total_reward = 0
    HorM = '0'
    action = 0

    iteration = 0
    success = 0

    while iteration < 100: #and \
            #self.LDBA.automaton_state != -1: #and \
            #self.LDBA.accepting_frontier_set:
        iteration += 1
        # label = env.get_label(state)
        nstate = np.append(state, temp_q)

        HorM, HorM_index = agent_c.get_action(nstate, label, episode)

        if HorM_index > 1:
            label_action = HorM
            next_state = state
        else:
            if HorM_index == 0:
                action = get_CPS_action(state, action)
            else:
                action = agent_h.choose_action(state)
            next_state, next_reward, done = env.step(action)
            label_action = env.get_label(next_state)
        # print(str(temp_q), label_action, HorM)
        # env.render()
        next_q, reward, Gamma = excution_LTL(label_action, str(temp_q), LTL_model, gamma, gammaB)

        if HorM_index == 1:
            agent_h.store_reward(next_reward + reward)

        total_reward += reward
        nnext_state = np.append(next_state, next_q)
        label = env.get_label(next_state)
        agent_c.update(nstate, HorM, reward, nnext_state, label, Gamma, episode)

        state = next_state
        temp_q = next_q
    agent_h.learn()
    # print(temp_q, LTL_model.accept_locations, success)
    if str(temp_q) in LTL_model.accept_locations:
        success = 1

    return total_reward, temp_q, success

def epsilon_annealing(i_epsiode, max_episode, min_eps: float):
    ##  if i_epsiode --> max_episode, ret_eps --> min_eps
    ##  if i_epsiode --> 1, ret_eps --> 1
    slope = (min_eps - 1.0) / max_episode
    ret_eps = max(slope * i_epsiode + 1.0, min_eps)
    # if i_epsiode > 300:
    #     i_epsiode = 300
    # slope = math.cos(math.pi * i_epsiode / 300)
    # ret_eps = max(slope, min_eps)
    return ret_eps

def in_critical_states(state):
    in_critical = False
    if (np.sqrt(state[2] * state[2] + state[3] * state[3]) > 0.15 and (np.sqrt(state[2]*state[2] + state[3]*state[3]) < 0.3)) or (abs(state[0] > 0.15) and abs(state[0] < 0.25)) or (abs(state[5] > 0.5) and abs(state[5] < 1.5)):
        in_critical = True
    return in_critical

def excution_LTL(action, temp_q, LTL_model, gamma, gammaB):
    # next_q = temp_q
    # lable = 'd'
    reward = 0.0
    Gamma = gamma

    # test2
    # a:到达a
    # b:到达b
    # c:陷阱
    # d:其他
    # if state in env.
    # action = env.get_label(state)
    # next_q = random.choice(LTL_model.get_nextLocations(str(temp_q), label))
    # print(str(temp_q), action, LTL_model.get_nextLocations(str(temp_q), action))
    if not LTL_model.get_nextLocations(temp_q, action):
        # print("None")
        return None, None, None
    next_q = LTL_model.get_nextLocations(temp_q, action)[0]
    if temp_q in LTL_model.accept_locations:
        reward += 1 - gammaB
        Gamma = gammaB
    # print(next_q)
    # print(next_q, action, reward)
    # print(action, next_q)


    return int(next_q), reward, Gamma


def get_CPS_action(state, action):
    machine_policy = np.load(file="resources/MDP/gridWorld4-5/machine_policy.npy", allow_pickle=True)
    action = machine_policy[state[1], state[0]]
    return action

def main(env, agent_h, agent_c, LTL_model, gamma, gammaB, nn_num):
   scores, avg_scores = train(agent_h, agent_c, env, LTL_model, gamma, gammaB, nn_num)

   print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

   fig = plt.figure()
   # ax = fig.add_subplot(111)
   # plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
   plt.ylim((0, 0.9))
   plt.plot(np.arange(100, len(avg_scores) + 1), avg_scores[99:])

   plt.legend(bbox_to_anchor=(1.05, 1))
   plt.ylabel('Avg reward on 100 episodes')
   plt.xlabel('Episodes #')
   plt.show()

   agent_h.save_models(1)
   agent_c.save('dir_chk/HCPS-LTL/grid_QL', 'GridWorld')

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

    num_episodes = 10000
    print_every = 2
    hidden_dim = 16  ## 64 ## 16
    min_eps = 0.001
    max_eps_episode = 1000

    env = Env()
    # env.seed(0)
    # env = gym.wrappers.Monitor(env, directory="monitors", force=True)

    space_dim = 2  # n_spaces
    action_dim = 4  # n_actions  get_action
    print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

    # threshold = env.spec.reward_threshold
    # print('threshold: ', threshold)

    agent_h = Reinforce(alpha=0.0001, state_dim=space_dim, action_dim=action_dim,
                      fc1_dim=16, fc2_dim=16, ckpt_dir='dir_chk/Reinforce_HPS/GridHCPS-2/', gamma=0.99)
    agent_c = QL(env, LTL_model, seed = 0)

    agent_h.load_models('dir_chk/Reinforce_HPS/GridWorld2/', 1)
    temp_q = 0

    main(env, agent_h, agent_c, LTL_model, gamma, gammaB, nn_num)
    print()