import gym
from Learner.agent_reinforce import Reinforce
import time
import random
from Learner.LDBA import LDBA, LDBAUtil
from collections import deque
import numpy as np

filepath = "resources/LDBA/LTL_model6.json"
LTL_model = LDBAUtil.LDBA_Util(filepath)
LDBA.make_model(LTL_model, "resources/LDBA/", "LTL_model6")
random.seed(0)

nn_num = len(LTL_model.locations)

env = gym.make('LunarLander-v2')
env.seed(0)

state_dim =  env.observation_space.shape[0] # n_spaces =
action_dim = env.action_space.n # n_actions =
hidden_dim = 16
agent_h = Reinforce(alpha=0.0005, state_dim=state_dim, action_dim=action_dim,
                    fc1_dim=256, fc2_dim=256, ckpt_dir='dir_chk/Reinforce_HPS/1/', gamma=0.99)
print('input_dim: ', state_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

agent_h.load_models('dir_chk/Reinforce_HPS/HCPS-4/', 1)
# agent_h.load_models('dir_chk/Reinforce_HPS/2/', 1)




def play(env, n_episodes):

    scores_deque = deque(maxlen=100)
    success_time = 0

    for i_episode in range(1, n_episodes + 1):
        s = env.reset()
        controler = "human"

        total_reward = 0
        time_start = time.time()
        timesteps = 0
        temp_q = 0
        action = 0
        pre_reward = None
        nstate = np.append(s, temp_q)

        while True:
            reward = 0

            if temp_q == 1:
                env.lander.color1 = (0, 255, 255, 1)
            if temp_q == 2:
                env.lander.color1 = (255, 255, 0, 1)

            # env.render(mode=controler)
            ########################################################
            action = agent_h.choose_action(s)
            s2, r, done, _ = env.step(action)
            landed = not env.lander.awake
            statement = [done, env.game_over, landed]
            next_q, LTL_reward, Gamma = LTL_model.execution(s2, str(temp_q), statement, 0.999999, 0.999)  # 自定义LTL自动机状态转换函数
            if done and landed and next_q == 1:
                r += 200
                success_time += 1
                print("Success!!!!!")
            if pre_reward is not None:
                reward += 0.01 * (Gamma * r - pre_reward)
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