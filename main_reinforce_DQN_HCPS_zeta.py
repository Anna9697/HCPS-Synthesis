import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from collections import deque
from Learner.agent_adapt import Agent
from Learner.agent_reinforce import Reinforce

from Learner.LDBA import LDBA, LDBAUtil

def train(agent_h, agent_c, env, LTL_model, gamma, zeta, nn_num):
    controler = "human"

    scores_deque = deque(maxlen=100)
    success_array = deque(maxlen=30)
    scores_array = []
    avg_scores_array = []

    time_start = time.time()
    max_score = -float('inf')

    for i_episode in range(num_episodes):
        eps = agent_c.epsilon_annealing(i_episode, max_eps_episode, min_eps)
        score, temp_q, success = run_episode(agent_h, agent_c, env, LTL_model, gamma, zeta, eps)
        if score > max_score:
            agent_h.save_models(0)
            agent_c.save('dir_chk/HCPS-LTL/zeta', 'LunarLanderMax')
            max_score = score

        scores_deque.append(score)
        scores_array.append(score)
        success_array.append(success)

        avg_score = np.mean(scores_deque)
        avg_scores_array.append(avg_score)
        success_rate = np.mean(success_array)

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            print('Episode: {:5} Score: {:5}  Avg.Score: {:.5f}, Success_Rate: {:.2f}, eps-greedy: {:5.2f}, Time: {:02}:{:02}:{:02}'. \
                format(i_episode, score, avg_score, success_rate, eps, dt // 3600, dt % 3600 // 60, dt % 60))

        if len(scores_deque) == scores_deque.maxlen:
            threshold = 30
            if np.mean(scores_deque) >= threshold:
            # if success_rate >= 0.73:
                print('\n Environment solved in {:d} episodes!\t Average Score: {:.2f}'. \
                      format(i_episode, np.mean(scores_deque)))
                break

    return scores_array, avg_scores_array  # , loss_array_plot

def run_episode(agent_h, agent_c, env, LTL_model, gamma, zeta, episode):
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
    statement = [False, False, False]
    label = LTL_model.get_lable(state, statement)
    temp_q = 0
    pre_reward = None
    nstate = np.append(state, temp_q)
    total_reward = 0
    HorM = 0
    # HorM_index = 0
    action = 0
    next_reward = 0
    success = 0


    while True:
        reward = 0
        # env_reward = 0
        nstate = np.append(state, temp_q)
        if temp_q == 1:
            env.lander.color1 = (0, 255, 255, 1)
        if temp_q == 2:
            env.lander.color1 = (255, 255, 0, 1)


        if in_critical_states(state):
            if temp_q == 0 and label in LTL_model.allow_e:
                HorM = agent_c.get_action(nstate, episode, True)
            else:
                HorM = agent_c.get_action(nstate, episode)

        if HorM > 1:
            label_action = LTL_model.epsilons[str(temp_q)][HorM - 2]
            # print(label_action)
            next_state = state
            done = False
        else:
            if HorM == 0:
                action = get_CPS_action(state, action)
            else:
                action = agent_h.choose_action(state)
        # env.render(mode=controler)
            next_state, next_reward, done, _ = env.step(action)
            landed = not env.lander.awake
            statement = [done, env.game_over, landed]
            label_action = LTL_model.get_lable(next_state, statement)
        label = LTL_model.get_lable(next_state, statement)
        # label = label_action

        next_q, LTL_reward = LTL_model.execution_zeta(str(temp_q), label_action, zeta)  # 自定义LTL自动机状态转换函数
        if done and landed and next_q == 1:
        # if LTL_reward > 0:
        #     done = True
            reward += 20
            # total_reward += 2
            success = 1
            print("Success!!!!!")
        if LTL_reward > 0:
            reward += 10
            done = True
            print("Zeta!!!!!")
        env_reward = get_env_reward(env, next_state, pre_reward, next_reward, gamma)
        reward += (LTL_reward + env_reward)
        # if pre_reward is not None:
        #     reward += 0.0001 * (Gamma * next_reward - pre_reward)
        #     reward += 0.1 * (1.0 * next_reward - pre_reward)
        # reward += LTL_reward

        agent_h.store_reward(reward)

        total_reward += reward


        nnext_state = np.append(next_state, next_q)

        # if HorM == 1:
        #     agent_h.step(nstate, action, reward, nnext_state, done, Gamma)
        agent_c.step(nstate, HorM, reward, nnext_state, done, gamma)

        # if LTL_reward > 0:
            # env.gameover = True
            # done = True
            # break

        pre_reward = next_reward
        state = next_state
        temp_q = next_q

        if done:
            env.close()
            break

    agent_h.learn()

    return total_reward, temp_q, success

def get_env_reward(env, next_state, pre_reward, next_reward, Gamma):
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
        reward += 0.1 * (Gamma * next_reward - pre_reward)
    return reward

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

def main(env, agent_h, agent_c, LTL_model, gamma, zeta, nn_num):
   scores, avg_scores = train(agent_h, agent_c, env, LTL_model, gamma, zeta, nn_num)

   print('length of scores: ', len(scores), ', len of avg_scores: ', len(avg_scores))

   fig = plt.figure()
   ax = fig.add_subplot(111)
   plt.plot(np.arange(1, len(scores) + 1), scores, label="Score")
   plt.plot(np.arange(1, len(avg_scores) + 1), avg_scores, label="Avg on 100 episodes")

   plt.legend(bbox_to_anchor=(1.05, 1))
   plt.ylabel('Score')
   plt.xlabel('Episodes #')
   plt.show()

   agent_h.save_models(1)
   agent_c.save('dir_chk/HCPS-LTL/zeta', 'LunarLanderMachine-v0')

if __name__ == '__main__':
    LTLpath = "resources/LDBA/LTL_model6.json"
    LTL_model = LDBAUtil.LDBA_Util(LTLpath)
    LDBA.make_model(LTL_model, "resources/LDBA/", "LTL_model6")
    # save_path = 'dir_chk/Reinforce_HPS/1/'

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
    zeta = 0.99
    LEARNING_RATE = 5e-5
    TARGET_UPDATE = 4

    num_episodes = 1512
    print_every = 2
    hidden_dim = 64  ## 64 ## 16
    min_eps = 0.05
    max_eps_episode = 500

    env = gym.make('LunarLander-v2')
    env.seed(8)
    env = gym.wrappers.Monitor(env, directory="monitors", force=True)

    space_dim = env.observation_space.shape[0]  # n_spaces
    action_dim = env.action_space.n  # n_actions  get_action
    print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

    threshold = env.spec.reward_threshold
    print('threshold: ', threshold)

    agent_h = Reinforce(alpha=0.0001, state_dim=space_dim, action_dim=action_dim,
                      fc1_dim=256, fc2_dim=256, ckpt_dir='dir_chk/Reinforce_HPS/HCPS-zeta/', gamma=0.99)
    agent_c = Agent(space_dim + 1, 2 + len(LTL_model.epsilons), hidden_dim, seed=0)

    agent_h.load_models('dir_chk/Reinforce_HPS/2/', 1)
    temp_q = 0

    main(env, agent_h, agent_c, LTL_model, gamma, zeta, nn_num)
    print()