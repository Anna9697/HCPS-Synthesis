import numpy as np
import random
import json

class QL:
    """
more details can be find in: https://arxiv.org/abs/2010.06797
    Attributes
    ----------
    MDP : an object from ./environments
        MDP object has to have the following properties
        (1) countably finite state and action spaces
        (2) a "step(action)" function describing the dynamics
        (3) a "state_label(state)" function that maps states to labels
        (4) a "reset()" function that resets the state to an initial state
        (5) current state of the MDP is "current_state"
        (6) action space is "action_space" and all actions are enabled in each state
    LDBA : an object from ./automata
        an automaton

    reward : array, shape=(n_pairs,n_qs,n_rows,n_cols)
        The reward function of the star-MDP. self.reward[state] = 1-discountB if 'state' belongs to B, 0 otherwise.

    discount : float
        The discount factor.

    discountB : float
        The discount factor applied to accepting states.

    """

    def __init__(self, env = None, LDBA = None, learning_rate = 0.9, seed = 0):
        if env is None:
            raise Exception("QL expects Env as an input")
        self.env = env
        if LDBA is None:
            raise Exception("LCRL expects LDBA as an input")
        self.LDBA = LDBA
        # self.epsilon_transitions_exists = 'epsilon' in self.LDBA.assignment.keys()
        #self.gamma = discount_factor
        # self.discount = gamma
        # self.discountB = gammaB
        self.alpha = learning_rate
        # self.epsilon = epsilon
        self.path_length = []
        self.Q = {}
        self.Q_initial_value = 0.0
        # ##### testing area ##### #
        self.test = False
        self.seed = random.seed(seed)

    def reset(self, nstate, label):
        # label = self.env.get_label(nstate[0:-1])
        action_space = self.action_space_augmentation(nstate[-1], label)
        self.Q[str(nstate)] = {}
        for action_index in range(len(action_space)):
            self.Q[str(nstate)][action_space[action_index]] = self.Q_initial_value = 0.0

    def get_action(self, nstate, label, eps):
        # label = self.env.get_label(nstate[0:-1])
        HorM_space = self.action_space_augmentation(nstate[-1], label)
        # print(nstate, label)
        Qs = []
        for action_index in range(len(HorM_space)):
            Qs.append(self.Q[str(nstate)][HorM_space[action_index]])
        # print(np.where(Qs == np.max(Qs))[0])
        if random.random() > eps:
            HorM_index = random.choice(np.where(Qs == np.max(Qs))[0])
        else:
            HorM_index = random.choice(range(len(HorM_space)))
        HorM = HorM_space[HorM_index]
        return HorM, HorM_index


    def update(self, nstate, action, reward, nnext_state, label, gamma, eps):
        # label = self.env.get_label(nstate[0:-1])
        HorM_space = self.action_space_augmentation(nnext_state[-1], label)
        Qs_prime = []
        if str(nnext_state) not in self.Q.keys():
            self.Q[str(nnext_state)] = {}
            for action_index in range(len(HorM_space)):
                self.Q[str(nnext_state)][HorM_space[action_index]] = 0
                Qs_prime.append(0)
        else:
            for action_index in range(len(HorM_space)):
                Qs_prime.append(self.Q[str(nnext_state)][HorM_space[action_index]])
        # print(self.Q[str(nstate)][action], reward, np.max(Qs_prime))
        self.Q[str(nstate)][action] = (1 - eps) * self.Q[str(nstate)][action] + eps * (
                    reward + gamma * np.max(Qs_prime))
        # print(self.Q[str(nstate)][action])

    # dependent reward and discount function
#     def reward(self, reward_flag):
#         if reward_flag > 0:
#             R = 1-self.discountB
#             gamma = self.discountB
#             return R, gamma
#         elif reward_flag < 0:
#             gamma = self.discount
#             return 0, gamma
#         else:
#             gamma = self.discount
#             return 0, gamma

    def action_space_augmentation(self, q, label):
        if str(q) in self.LDBA.epsilons.keys() and label in self.LDBA.allow_e:
            product_MDP_action_space = ['0', '1'] + self.LDBA.epsilons[str(q)]
        else:
            product_MDP_action_space = ['0', '1']
        return product_MDP_action_space
    def save(self, directory, filename):
        with open('%s/%s_LTL_local.json' % (directory, filename), "w", encoding='utf-8') as f:  ## 设置'utf-8'编码
            f.write(json.dumps(self.Q, ensure_ascii=False, indent=4))