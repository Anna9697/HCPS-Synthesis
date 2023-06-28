import gym
from Learner.agent_adapt import Agent
import time
import random
from Learner.LDBA import LDBA, LDBAUtil

filepath = "resources/LDBA/LTL_model6.json"
LTL_model = LDBAUtil.LDBA_Util(filepath)
LDBA.make_model(LTL_model, "resources/LDBA/", "/LTL_model6")
random.seed(0)

nn_num = len(LTL_model.locations)

env = gym.make('LunarLander-v2')
env.seed(0)

state_dim =  env.observation_space.shape[0] # n_spaces =
action_dim = env.action_space.n # n_actions =
hidden_dim = 64
# print('input_dim: ', state_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)
agent_c = Agent(state_dim + 1, 2 + len(LTL_model.epsilons), hidden_dim, seed=0)
# agent_c.load_nn('dir_chk/HCPS-LTL/8/', 'LunarLanderMachine-v0')
# agent_c.load_nn('dir_chk/HCPS-LTL/LDGBA/', 'LunarLanderMachine-v0')
agent_c.load_nn('dir_chk/HCPS-LTL/zeta/', 'LunarLanderMachine-v0')

def in_critical_states(state):
    in_critical = False
    if (np.sqrt(state[2] * state[2] + state[3] * state[3]) >= 0.0 and (np.sqrt(state[2]*state[2] + state[3]*state[3]) < 0.25)) or (abs(state[0] >= 0.0) and abs(state[0] < 0.25)) or (abs(state[5] >= 0.0) and abs(state[5] < 1.5)):
        in_critical = True
    return in_critical


def get_CPS_action(state, action):
    if np.sqrt(state[2]*state[2] + state[3]*state[3]) <= 0.2 and abs(state[4]) <= 0.2 and abs(state[5]) <= 0.3: #稳定
        if abs(state[0]) < 0.2:  #在区域内
            action = 0
        else:          #在区域外
            if state[3] < -0.2 : #超速下降
                action = 2
            else:    #正常下降或上升
                if state[0] < 0: #在范围左边
                    action = 3
                else:  #在范围右边
                    action = 1
    else :    #不稳定
        #速度f，角度t，角速度t
        if np.sqrt(state[2]*state[2] + state[3]*state[3]) > 0.2 and abs(state[4]) <= 0.2 and abs(state[5]) <= 0.3:
            if state[3] > 0:  #向上移动，调节左右
                if state[2] > 0: #向右移动
                    action = 1
                else:
                    action = 3
            else: #整体向下
                if abs(state[2]) > abs(state[3]): #左右移动速度更大
                    if state[2] > 0: #向右移动
                        action = 1
                    else:
                        action = 3
                else:  #向下速度更大
                    action = 2
        #速度t，角度f，角速度t
        if np.sqrt(state[2]*state[2] + state[3]*state[3]) <= 0.2 and abs(state[4]) > 0.2 and abs(state[5]) <= 0.3:
            if state[4] > 0: #左偏
                action = 3
            else:
                action = 1

        #速度f，角度f，角速度t
        if np.sqrt(state[2]*state[2] + state[3]*state[3]) > 0.2 and abs(state[4]) > 0.2 and abs(state[5]) <= 0.3:
            if state[4] > 0: #角度偏左
                if state[2] >= 0 and state[3] >= 0: #向右上移动
                    action = 1
                elif state[2] >= 0 and state[3] < 0: #向右下移动
                    action = 2
                elif state[2] < 0 and state[3] >= 0: #向左上移动
                    action = 3
                elif state[2] < 0 and state[3] < 0: #向左下移动
                    if abs(state[2]) > abs(state[3]):
                        action = 3
                    else:
                        action = 2
            else:  #角度偏右
                if state[2] >= 0 and state[3] >= 0: #向右上移动
                    action = 1
                elif state[2] >= 0 and state[3] < 0: #向右下移动
                    action = 1
                elif state[2] < 0 and state[3] >= 0: #向左上移动
                    action = 3
                elif state[2] < 0 and state[3] < 0: #向左下移动
                    action = 2
        #角速度f
        if abs(state[5]) > 0.3:
            if state[5] > 0: #逆时针旋转
                action = 3
            else:  #顺时针旋转
                action = 1

    return action


from collections import deque
import numpy as np



def play(env, n_episodes):
    scores_deque = deque(maxlen=100)
    success_time = 0

    for i_episode in range(1, n_episodes + 1):
        s = env.reset()
        controler = "human"
        statement = [False, False, False]
        label = LTL_model.get_lable(s, statement)

        total_reward = 0
        time_start = time.time()
        timesteps = 0
        temp_q = 0
        action = 0
        pre_reward = None
        HorM = 0

        while True:
            reward = 0
            nstate = np.append(s, temp_q)
            # action = get_CPS_action(s, action)

            if temp_q == 1:
                env.lander.color1 = (0, 255, 255, 1)
            if temp_q == 2:
                env.lander.color1 = (255, 255, 0, 1)

            # env.render(mode=controler)
            ########################################################
            if in_critical_states(s):
                if temp_q == 0 and label in LTL_model.allow_e:
                    HorM = agent_c.get_action(nstate, 0.05, True)
                else:
                    HorM = agent_c.get_action(nstate, 0.05)
            if HorM > 1:
                label_action = LTL_model.epsilons[str(temp_q)][HorM - 2]
                # print(label_action)
                s2 = s
                done = False
            else:
                action = get_CPS_action(s, action)
                s2, r, done, _ = env.step(action)
                landed = not env.lander.awake
                statement = [done, env.game_over, landed]
            # print(s2[0],s2[1], s2[2],s2[3], s2[4],s2[5])
                label_action = LTL_model.get_lable(s2, statement)
            label = LTL_model.get_lable(s2, statement)
            next_q, LTL_reward, Gamma = LTL_model.execution(str(temp_q), label_action, 0.999999, 0.999)  # 自定义LTL自动机状态转换函数
            if done and landed and next_q == 1:
                LTL_reward += 20
                success_time += 1
                print("Success!!!!!")
            if pre_reward is not None:
                reward += 0.1 * (Gamma * r - pre_reward)
            reward += LTL_reward

            nstate = np.append(s, temp_q)

            s = s2
            temp_q = next_q
            pre_reward = r
            total_reward += reward
            timesteps += 1

            if done:
                break

        delta = (int)(time.time() - time_start)

        scores_deque.append(total_reward)


        print('Episode {}\tAverage Score: {:.2f}, \t Timesteps: {} \tTime: {:02}:{:02}:{:02}' \
              .format(i_episode, np.mean(scores_deque), timesteps, \
                      delta // 3600, delta % 3600 // 60, delta % 60))
    print(success_time)

play(env=env, n_episodes=1000)