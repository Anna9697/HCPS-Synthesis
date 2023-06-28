import json
import copy
import random
import numpy as np
from graphviz import Digraph

class LDGBA(object):
    def __init__(self, locations, actions, epsilons, trans, finale_conditons, accept_locations, current_accepts, allow_e_actions):
        self.locations = locations
        self.actions = actions
        self.epsilons = epsilons
        self.transitions = trans
        self.finale_conditons = finale_conditons
        self.accept_locations = accept_locations
        self.allow_e = allow_e_actions
        self.current_accept = current_accepts

    def add_Transitions(self, trans):
        for tran in trans:
            if tran not in trans:
                self.transitions.append(tran)

    def get_InitLocation(self):
        for l in self.locations:
            if l.init:
                return l
    def get_AcceptLocations(self):
        accept_locations = []
        for l in self.locations:
            if l.accept:
                accept_locations.append(l)
        return accept_locations
    
    def get_nextLocations(self, location, action):
        next_locations = []
        for tran in self.transitions:
            if tran.source == location and tran.action == action:
                next_locations.append(tran.target)
        return next_locations

    def make_model(data, filePath, fileName):
        dot = Digraph()
        init_location = data.get_InitLocation()
        dot.node(name='', label='', shape='plaintext')
        dot.edge('', str(init_location.location), ' start', style='dashed')
        for l in data.locations:
            if l.accept:
                dot.node(name=str(l.location), label=str(l.location), shape='doublecircle')
            else:
                dot.node(name=str(l.location), label=str(l.location))
        for tran in data.transitions:
            tranLabel = " " + str(tran.action) + " " + str(tran.reward)
            dot.edge(str(tran.source), str(tran.target), tranLabel)
        newFilePath = filePath + fileName
        dot.render(newFilePath, view=False)

    def get_lable(self, state, statement):
        # next_q = 0
        # action = 'c'
        # reward = 0
        # # label = 'c'

        in_area = abs(state[0]) <= 0.2
        in_speed = np.sqrt(state[2] * state[2] + state[3] * state[3]) <= 0.2
        in_aspeed = abs(state[5]) <= 1.0
        # oob = abs(state[0] - 2.0 / 5) >= 1.0 or state[1] >= 1.7
        # crash = (state[6] or state[7]) and not in_speed

        # print(np.sqrt(state[2] * state[2] + state[3] * state[3]),  abs(state[5]))

        # test4
        # a:在范围内且平稳
        # b:不在范围内或不平稳
        # c:出界/坠毁/超时
        if not statement[0]:
            if in_speed and in_area and in_aspeed:
                # if np.sqrt(state[2] * state[2] + state[3] * state[3]) < 0.2 and abs(state[4]) <= 0.5 and abs(
                #         state[5]) <= 1.0 and (state[3] >= -0.2 and state[3] <= -0.05) and abs(state[0]) <= 0.2:
                lable = 'a'
            # elif not crash:
            #     lable = 'c'
            else:
                lable = 'c'
        else:
            if statement[1]:
                lable = 'b'
        # else:
            elif statement[2] and in_speed and in_area and in_aspeed:
                lable = 'a'
                # reward += 1000
                # print("Success!!!!!")
            else:
                lable = 'b'

        return lable

    def execution(self, temp_q, label_action, gamma, gammaB):
        # next_q = temp_q
        # lable = 'c'
        reward = 0.0
        Gamma = gamma

        # lable = self.get_lable(state, statement)
        # print(temp_q, label_action)
        if not self.get_nextLocations(temp_q, label_action):
            # print("None")
            return None, None, None

        next_q = self.get_nextLocations(temp_q, label_action)[0]

        if self.get_reward_label(next_q):
            reward += 1-gammaB
            Gamma = gammaB

        return int(next_q), reward, Gamma

    def get_reward_label(self, q):
        indeces_to_remove = []
        for accepts in self.current_accept:
            # print(accepts)
            if q in accepts:
                indeces_to_remove.append(accepts)
                # Acc_set.remove(con)
                self.current_accept.remove(accepts)
                # print(Acc_set)
                # return 1.0
        # self.current_accept =
        if not self.current_accept:
            self.current_accept = copy.deepcopy(self.finale_conditons)
            # print(self.current_accept, self.finale_conditons)
        # print(q,indeces_to_remove)
        if indeces_to_remove:
            # print("True")
            return True
        else:
            return False

        # if not Acc_set:
        #     Acc_set = copy.deepcopy(self.finale_conditons)
            # print("Acc_set:",Acc_set)
        # return 0.0



class Location(object):
    def __init__(self, location_id, location, b_init, b_accept, location_value):
        self.id = location_id
        self.location = location
        self.init = b_init
        self.accept = b_accept
        self.value = location_value

class Transition(object):
    def __init__(self, tran_id, source, action, reward, target):
        self.id = tran_id
        self.source = source
        self.action = action
        self.reward = reward
        self.target = target

    def is_passing_tran(self, tran):
        if self.source == tran.source and self.action == tran.action:
            return True
        else:
            return False


class LDGBAUtil(object):
    def LDBA_Util(filepath):
        with open(filepath, 'r') as json_model:
            model = json.load(json_model)

        actions = model["actions"]
        epsilons = model["epsilons"]
        l_list = model["locations"]
        tran_list = model["transitions"]
        init_locations = model["init"]
        accept_conditions = model["accept"]
        allow_e_actions = model["allow_e"]

        transitions = []
        locations = []
        finale_conditons = []
        accept_locations = []
        current_accepts = []

        for acc in accept_conditions:
            # print(accept_conditions[acc])
            # print(acc)
            finale_conditons.append(accept_conditions[acc])
            current_accepts.append(accept_conditions[acc])
            for l in accept_conditions[acc]:
                if l not in accept_locations:
                    accept_locations.append(l)


        for tran in tran_list:
            tran_id = str(tran)
            source = tran_list[tran][0]
            action = tran_list[tran][1]
            reward = tran_list[tran][2]
            target = tran_list[tran][3]
            transitions.append(Transition(tran_id, source, action, reward, target))
        for l in l_list:
            l_id = str(l)
            b_init = False
            b_aceept = False
            if l in init_locations:
                b_init = True
            if l in accept_locations:
                b_aceept = True
            locations.append(Location(l_id, l, b_init, b_aceept, 0))
        
        return LDGBA(locations, actions, epsilons, transitions, finale_conditons, accept_locations, current_accepts, allow_e_actions)