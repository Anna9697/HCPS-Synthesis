BUFFER_SIZE =int(1e5)
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
TAU=1e-3
UPDATE_EVERY = 4

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from Learner.model_adapt import QNetwork
from torch.distributions import Categorical
import torch.nn.functional as F
from Learner.replay_buffer_adapt import ReplayBuffer
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PolicyNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, seed=0):
        super(PolicyNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.prob = nn.Linear(fc2_dim, action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(device)
        # self.seed = random.seed(seed)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        prob = torch.softmax(self.prob(x), dim=-1)

        return prob

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

class Reinforce:
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, ckpt_dir, gamma=0.99):
        self.gamma = gamma
        self.checkpoint_dir = ckpt_dir
        self.reward_memory = []
        self.log_prob_memory = []

        self.policy = PolicyNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                    fc1_dim=fc1_dim, fc2_dim=fc2_dim)

    def choose_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(device)
        probabilities = self.policy.forward(state)
        dist = Categorical(probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_prob_memory.append(log_prob)

        return action.item()

    def store_reward(self, reward):
        self.reward_memory.append(reward)

    def learn(self):
        # ---------------------------------------------------------------
        # 法一
        # ---------------------------------------------------------------
        G_list = []
        G_t = 0
        for item in self.reward_memory[::-1]:
            G_t = self.gamma * G_t + item
            G_list.append(G_t)
        G_list.reverse()
        G_tensor = torch.tensor(G_list, dtype=torch.float).to(device)

        # ---------------------------------------------------------------
        # 法二
        # ---------------------------------------------------------------
        # G = np.zeros_like(self.reward_memory, dtype=np.float64)
        # for t in range(len(self.reward_memory)):
        #     G_sum = 0
        #     discount = 1
        #     for k in range(t, len(self.reward_memory)):
        #         G_sum += self.reward_memory[k] * discount
        #         discount *= self.gamma
        #     G[t] = G_sum
        # G_tensor2 = T.tensor(G, dtype=T.float).to(device)

        loss = 0
        for g, log_prob in zip(G_tensor, self.log_prob_memory):
            loss += -g * log_prob


        if not isinstance(loss, int):
            self.policy.optimizer.zero_grad()
            loss.backward()
            self.policy.optimizer.step()
        self.reward_memory.clear()
        self.log_prob_memory.clear()

    def save_models(self, episode):
        self.policy.save_checkpoint(self.checkpoint_dir + 'Reinforce_policy_{}.pth'.format(episode))
        print('Saved the policy network successfully!')

    def load_models(self, path, episode):
        self.policy.load_checkpoint(path + 'Reinforce_policy_{}.pth'.format(episode))
        print('Loaded the policy network successfully!')

    # def save(self, directory, filename):
    #     # for i in range(nn_num):
    #     torch.save(self.qnetwork_local.state_dict(), '%s/%s_LTL_local.pth' % (directory, filename))
    #     torch.save(self.qnetwork_target.state_dict(), '%s/%s_LTL_target.pth' % (directory, filename))
    #
    # def load_nn(self, directory, filename):
    #     self.qnetwork_local.load_state_dict(torch.load('%s/%s_LTL_local.pth' % (directory, filename)))
    #     self.qnetwork_target.load_state_dict(torch.load('%s/%s_LTL_target.pth' % (directory, filename)))



# class Agent():
#
#     def __init__(self, state_size, action_size, seed):
#         """Initialize an Agent object.
#
#         Params
#         ======
#             state_size (int): dimension of each state
#             action_size (int): dimension of each action
#             seed (int): random seed
#         """
#         self.device = torch.device("cuda" if use_cuda else "cpu")
#         self.state_size = state_size
#         self.action_size = action_size
#         self.seed = random.seed(seed)
#
#         # Q-Network
#         self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
#         self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
#         self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
#
#         # Replay memory
#         self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
#         # Initialize time step (for updating every UPDATE_EVERY steps)
#         self.t_step = 0
#
#     def step(self, state, action, reward, next_state, done, GAMMA):
#         # Save experience in replay memory
#         self.memory.add(state, action, reward, next_state, done)
#
#         # Learn every UPDATE_EVERY time steps.
#         self.t_step = (self.t_step + 1) % UPDATE_EVERY
#         if self.t_step == 0:
#             # If enough samples are available in memory, get random subset and learn
#             if len(self.memory) > BATCH_SIZE:
#                 experiences = self.memory.sample()
#                 self.learn(experiences, GAMMA)
#
#
#     def get_action(self, state, eps, check_eps=True):
#         """Returns actions for given state as per current policy.
#
#         Params
#         ======
#             state (array_like): current state
#             eps (float): epsilon, for epsilon-greedy action selection
#         """
#         # device = torch.device("cuda" if use_cuda else "cpu")
#         state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
#         self.qnetwork_local.eval()
#         with torch.no_grad():
#             action_values = self.qnetwork_local(state)
#         self.qnetwork_local.train()
#
#         # Epsilon-greedy action selection
#         if random.random() > eps:
#             return np.argmax(action_values.cpu().data.numpy())
#         else:
#             return random.choice(np.arange(self.action_size))
#
#     def learn(self, experiences, gamma):
#         """Update value parameters using given batch of experience tuples.
#
#         Params
#         ======
#             experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
#             gamma (float): discount factor
#         """
#         # Obtain random minibatch of tuples from D
#         states, actions, rewards, next_states, dones = experiences
#
#         ## Compute and minimize the loss
#         ### Extract next maximum estimated value from target network
#         q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
#         ### Calculate target value from bellman equation
#         q_targets = rewards + gamma * q_targets_next * (1 - dones)
#         ### Calculate expected value from local network
#         q_expected = self.qnetwork_local(states).gather(1, actions)
#
#         ### Loss calculation (we used Mean squared error)
#         loss = F.mse_loss(q_expected, q_targets)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
#
#         # ------------------- update target network ------------------- #
#         self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
#
#
#     def soft_update(self, local_model, target_model, tau):
#         """ tau (float): interpolation parameter"""
#
#         for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
#             target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
#
#     def hard_update(self, local, target):
#         for target_param, param in zip(target.parameters(), local.parameters()):
#             target_param.data.copy_(param.data)
#
#     def save(self, directory, filename):
#         # for i in range(nn_num):
#         torch.save(self.qnetwork_local.state_dict(), '%s/%s_LTL_local.pth' % (directory, filename))
#         torch.save(self.qnetwork_target.state_dict(), '%s/%s_LTL_target.pth' % (directory, filename))
#
#     def load_nn(self, directory, filename):
#         self.qnetwork_local.load_state_dict(torch.load('%s/%s_LTL_local.pth' % (directory, filename)))
#         self.qnetwork_target.load_state_dict(torch.load('%s/%s_LTL_target.pth' % (directory, filename)))
#
#     def epsilon_annealing(self, i_epsiode, max_episode, min_eps: float):
#         ##  if i_epsiode --> max_episode, ret_eps --> min_eps
#         ##  if i_epsiode --> 1, ret_eps --> 1
#         slope = (min_eps - 1.0) / max_episode
#         ret_eps = max(slope * i_epsiode + 1.0, min_eps)
#         # if i_epsiode > 300:
#         #     i_epsiode = 300
#         # slope = math.cos(math.pi * i_epsiode / 300)
#         # ret_eps = max(slope, min_eps)
#         return ret_eps


            

