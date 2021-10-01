import random
import numpy as np
from smac.env.pettingzoo import StarCraft2PZEnv
import typing
from copy import deepcopy
from utils.ReplayBuffer import ReplayBuffer, Episode

import torch
import torch.nn as nn
import torch.nn.functional as F

# Receive current individual Observation o_t ^a
# Receive last agent action u_t ^a

class RNNAgent(nn.Module):
    def __init__(self, input_shape, hidden_dim, num_actions):
        super(RNNAgent, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

# Take rnn agent network ouputs as input
class QMIX(nn.Module):
    def __init__(self, num_agents, state_dim, embed_dim, hyper_embed_dim):
        super(QMIX, self).__init__()
        self.n_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.hyper_embed = hyper_embed_dim

        # Hypernetwork
        # Takes the state s as input and generates the weights of one layer of the mixing network
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim,  self.hyper_embed),
                                       nn.ReLU(),
                                       nn.Linear( self.hyper_embed, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim,  self.hyper_embed),
                                           nn.ReLU(),
                                           nn.Linear(self.hyper_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    # output of rnn agents is vector. We need to reshape
    def forward(self, agent_qs, states):
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)

        # First layer
        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)

        # Reshape with view
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)

        # Batch matrix matrix product
        # (b x m x n) X (b x n x p) -> (b x m x p)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)

        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + v

        bs = agent_qs.size(0)
        q_total = y.view(bs, -1, 1)
        return q_total

# 의문점 :
# 1. 에이전트 q 네트워크의 가중치를 어떻게 공유시킬까?
# 2. torch.tensor에 vector stack 어떻게 하는가?
# 3. train에서는 obs isn't given. How can I calculate Q

def train_qmix(agent_num: int, batch_size:int, memory:ReplayBuffer,
               mix_net:QMIX, target_net:QMIX, agent_qnet:RNNAgent):
    batches = memory.sample(batch_size)
    for _, episode_batch in enumerate(batches):
        Q_values = []
        for t, transition in enumerate(episode_batch):
            state = transition['state']
            joint_action = transition['joint_action']
            reward = transition['reward']
            next_state = transition['next_state']
            q_vector = torch.tensor(agent_qnet[agent].forward()
                                    for agent in range(agent_num))



def target_update():
    pass

def random_policy(env, agents: list, obs_dict: dict, dones: dict):
    # 기본적으로 dones에 존재하지 않는 에이전트들의 행동은 None으로 처리
    actions = {agent: None for agent in agents}

    # dones에 존재하는 에이전트들에게 랜덤한 액션을 할당한다.
    for agent in dones.keys():
        # 에이전트가 done이라면 None Action을 할당한다.
        if dones[agent]:
            action = None
        # 에이전트가 done이 아니라면 action_mask에 있는 행동들 중에서 하나를
        # 랜덤으로 선택한다.
        elif isinstance(obs_dict[agent], dict) and "action_mask" in obs_dict[agent]:
            action = random.choice(np.flatnonzero(obs_dict[agent]["action_mask"]))
        else:
            action = env.actions_space[agent].sample()

    return actions

def print_messages(num_epi, steps, total_rewards):
    print(f"{num_epi} episode is finisished")
    print(f"Current steps : {steps}")
    print(f"Total rewards of episode : {total_rewards}")

def get_env_info(env):
    """
    :param env:
    :return: agents_list, obs_size, action_size
    """

    agents_list = env.agents  # ['marine_0', 'marine_1', 'marine_2']
    first_agent = agents_list[0]  # marine_0

    obs_dict = env.observation_spaces[
        first_agent]  # Dict(action_mask:Box(0, 1, (8,), int8), observation:Box(-1.0, 1.0, (30,), float32))
    obs_shape = obs_dict['observation'].shape  # (30, )
    OBS_SIZE = obs_shape[0]  # 30
    ACTION_SIZE = env.action_spaces[first_agent].n  # 8

    return agents_list, OBS_SIZE, ACTION_SIZE


def run():
    # Environment Initialization
    env = StarCraft2PZEnv.parallel_env(map_name="3m") # if map_name == "3m"
    agents_list, obs_size, actions_size = get_env_info(env)
    init_state = env.state() # (48, )
    agents_num = len(agents_list)
    # Network Initialization
    # 에이전트들의 obs_space, actions_space가 모두 같다.
    # 우선 dictionary를 통해 네트워크를 각 에이전트마다 구성해보자.
    # 언젠가는 네트워크를 공유해서 짜서 비교실험을 해야한다.
    q_networks = {agent: RNNAgent(input_shape=obs_size+1, hidden_dim=32, num_actions=actions_size).cuda()
               for agent in agents_list}

    mixing_network = QMIX(num_agents=agents_num, state_dim=init_state.shape[0], embed_dim=32, hyper_embed_dim=64)
    target_network = QMIX(num_agents=agents_num, state_dim=init_state.shape[0], embed_dim=32, hyper_embed_dim=64)


    # Define max_step, max_epi
    max_step = 10000
    max_epi = 50
    steps = 0
    num_epi = 0
    time_limit = 100
    memory_size = 50000

    # Epsilon Initialization
    start_eps = 1.0
    end_eps = 0.05
    eps = start_eps
    step_drop = (start_eps - end_eps) / max_epi

    # Replay buffer initialzation
    memory = ReplayBuffer(memory_size)

    while steps < max_step and num_epi < max_epi:

        # Initialize information of episode
        t = 0
        state = env.state()

        obs_dict = env.reset()
        # obs_dict의 경우 2중 딕셔너리로 구성되어 있다.
        # obs_dict는 agent가 key이고 item은 dict이다.
        # obs_dict = {agent0 : item_dict, ...}
        # item dict는 key가 observation, action_mask다.
        # item_dict = {'observation' : (30,), 'action_mask' : (8,)

        # Initialize first trajectories and joint actions
        # tau는 agent의 action-observation history
        dones = {a: False for a in agents_list}
        joint_actions = random_policy(env, agents_list, obs_dict,dones) # u_0
        pre_trajectories = dict()
        for a in agents_list:
            pre_trajectories[a] = (obs_dict[a]['observation'], obs_dict[a]['action_mask'],
                                   joint_actions[a])

        hidden_states = {a : q_networks[a].init_hidden() for a in agents_list} # taus : hidden state for RNN

        total_rewards = 0
        Q = dict()

        episode = Episode(time_limit)

        # Schedule epsiolon
        if (eps > end_eps):
            eps -= step_drop

        while True:
            env.render()
            for a in dones.keys():
                obs = obs_dict[a]['observation']
                action = joint_actions[a]
                if action==None:
                    action = np.array([0])
                input = torch.from_numpy(np.append(obs, action))
                input = input.cuda().float()
                input = input.unsqueeze(0)
                Q[a], hidden_states[a] = q_networks[a].forward(input, hidden_states[a])
                coin = random.random()
                action_mask = obs_dict[a]["action_mask"]
                if dones[a]:
                    action = None
                elif coin < eps:
                    action = random.choice(np.flatnonzero(action_mask))
                else: # dones[a] == False and coin >= eps
                    action = torch.argmax(Q[a])
                    if action not in action_mask:
                        action = 0
                joint_actions[a] = action

            # step environment
            obs_dict, rewards, dones, _ = env.step(joint_actions)
            print(dones)
            t += 1
            steps += 1
            r = sum(rewards.values())
            total_rewards += r
            next_state = env.state()

            # Add (state, joint_action, reward, next_state) transition in episodebuffer
            episode.add(state, joint_actions, r, next_state, dones, obs_dict)
            state = next_state

            if False not in dones.values() and t < time_limit:
                break

        # Put episode in replay buffer
        memory.put(episode.trajectory)
        print(len(episode.trajectory))
        print(len(memory))
        print_messages(num_epi, steps, total_rewards)
        num_epi += 1

run()

