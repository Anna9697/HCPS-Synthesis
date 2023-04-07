import gym
from Learner.agent_adapt import Agent
from Learner.agent_reinforce import Reinforce
import time
import random
from Learner.LDBA import LDBA, LDBAUtil
from collections import deque
import numpy as np
from env.grid_world import Env

filepath = "resources/LDBA/LTL_gridworld1.json"
LTL_model = LDBAUtil.LDBA_Util(filepath)
LDBA.make_model(LTL_model, "resources/LDBA/", "LTL_gridworld1")
random.seed(0)

nn_num = len(LTL_model.locations)

env = Env()
# env.seed(0)

state_dim = 2 # n_spaces =
action_dim = 4 # n_actions =
hidden_dim = 16
agent_h = Reinforce(alpha=0.00001, state_dim=state_dim, action_dim=action_dim,
                    fc1_dim=hidden_dim, fc2_dim=hidden_dim, ckpt_dir='dir_chk/Reinforce_HPS/GridHCPS-2/', gamma=0.99)
agent_c = Agent(state_dim + 1, 2, seed=0)
print('input_dim: ', state_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

agent_h.load_models('dir_chk/Reinforce_HPS/GridWorld2/', 1)
agent_c.load_nn('dir_chk/HCPS-LTL/grid2/', 'GridWorld')


def get_CPS_action(state, action):
    machine_policy = np.load(file="resources/MDP/gridWorld4-5/machine_policy.npy", allow_pickle=True)
    action = machine_policy[state[1], state[0]]
    return action

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


def play(env, n_episodes):
    scores_deque = deque(maxlen=100)
    success_time = 0

    for i_episode in range(1, n_episodes + 1):
        s = env.reset()
        # controler = "human"

        total_reward = 0
        time_start = time.time()
        timesteps = 0
        temp_q = 0
        action = 0
        # pre_reward = None
        nstate = np.append(s, temp_q)
        HorM = 0

        human_time = 0

        for i in range(100):
            reward = 0

            # if temp_q == 1:
            #     env.lander.color1 = (0, 255, 255, 1)
            # if temp_q == 2:
            #     env.lander.color1 = (255, 255, 0, 1)

            # env.render()
            ########################################################
            # if in_critical_states(s):
            HorM = agent_c.get_action(nstate, 0)
            if HorM == 0:
                action = get_CPS_action(s, action)
            else:
                action = agent_h.choose_action(s)
                human_time += 1
            s2, r, done = env.step(action)
            # landed = not env.lander.awake
            # statement = [done, env.game_over, landed]
            next_q, reward, Gamma = excution_LTL(env, s2, str(temp_q), LTL_model, 0.9999,
                                                 0.99)  # 自定义LTL自动机状态转换函数

            # if pre_reward is not None:
            #     reward += 0.01 * (Gamma * r - pre_reward)
            # reward += reward

            nstate = np.append(s, temp_q)

            s = s2
            temp_q = next_q
            pre_reward = r
            total_reward += reward
            timesteps += 1

            # if done:
            #     break

        delta = (int)(time.time() - time_start)
        if done and (next_q == 1 or next_q == 2):
            #     r += 200
            success_time += 1
            print("Success!!!!!")

        scores_deque.append(total_reward)


        print('Episode {}\tAverage Score: {:.2f}, \t Timesteps: {} \t Human Choice: {} \tTime: {:02}:{:02}:{:02}' \
              .format(i_episode, np.mean(scores_deque), timesteps, human_time/timesteps, \
                      delta // 3600, delta % 3600 // 60, delta % 60))
    print(success_time)

play(env=env, n_episodes=1000)