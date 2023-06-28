import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import json
from collections import deque
from env.grid_world import Env
from Learner.agent_adapt import Agent
from Learner.agent_reinforce import Reinforce
from Learner.agent_QL import QL
from Learner.LDBA import LDBA, LDBAUtil



if __name__ == '__main__':
    # random.seed(0)
    # LTLpath = "resources/LDBA/LTL_gridworld1.json"
    # LTL_model = LDBAUtil.LDBA_Util(LTLpath)

    use_cuda = torch.cuda.is_available()
    FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    device = torch.device("cuda" if use_cuda else "cpu")
    print('device: ', device)

    hidden_dim = 16  ## 64 ## 16

    env = Env()
    # env.seed(0)
    # env = gym.wrappers.Monitor(env, directory="monitors", force=True)

    space_dim = 2  # n_spaces
    action_dim = 4  # n_actions  get_action
    print('input_dim: ', space_dim, ', output_dim: ', action_dim, ', hidden_dim: ', hidden_dim)

    # threshold = env.spec.reward_threshold
    # print('threshold: ', threshold)

    agent_h_before = Reinforce(alpha=0.0005, state_dim=space_dim, action_dim=action_dim,
                      fc1_dim=16, fc2_dim=16, ckpt_dir='dir_chk/Reinforce_HPS/GridWorld1/', gamma=0.99)
    agent_h_after = Reinforce(alpha=0.0005, state_dim=space_dim, action_dim=action_dim,
                               fc1_dim=16, fc2_dim=16, ckpt_dir='dir_chk/Reinforce_HPS/GridHCPS-2/', gamma=0.99)
    # agent_c = Agent(space_dim + 1, 4,seed = 1)
    # agent_c = QL(env, LTL_model, seed = 0)

    agent_h_before.load_models('dir_chk/Reinforce_HPS/GridWorld2/', 1)
    # agent_h_after.load_models('dir_chk/Reinforce_HPS/GridHCPS-2/', 1)
    agent_h_after.load_models('dir_chk/Reinforce_HPS/GridHCPS-2-LDGBA/', 1)
    # agent_h_after.load_models('dir_chk/Reinforce_HPS/GridHCPS-2-zeta/', 1)
    # agent_c.load_nn('dir_chk/HCPS-LTL/grid2/', 'GridWorld')

    # with open("dir_chk/HCPS-LTL/grid_QL/GridWorld_LTL_local.json", 'r', encoding='UTF-8') as f:
    #     switch_policy = json.load(f)
    with open("dir_chk/HCPS-LTL/grid_QL_LDGBA/GridWorld_LTL_local.json", 'r', encoding='UTF-8') as f:
        switch_policy = json.load(f)
    # with open("dir_chk/HCPS-LTL/grid_QL_zeta/GridWorld_LTL_local.json", 'r', encoding='UTF-8') as f:
    #     switch_policy = json.load(f)

    for agent in [agent_h_before, agent_h_after]:
        print('---human policy---')
        for i in range(5):
            for j in range(4):
                state = torch.tensor([[j, i]], dtype=torch.float).to(device)
                probabilities = agent.policy.forward(state).tolist()[0]
                print([j, i], ':', probabilities)
    print('---switch policy---')
    # print(sorted(switch_policy.items(), key=lambda d: d[0], reverse=False))
    for temp_q in range(4):
        for i in range(5):
            for j in range(4):
                key = str(np.array([j, i, temp_q]))
                # print(str(np.array([j, i, temp_q])))
                if key in switch_policy.keys():
                    print([j, i, temp_q], ':', switch_policy[key])
                else:
                    print([j, i, temp_q], ':', None)


