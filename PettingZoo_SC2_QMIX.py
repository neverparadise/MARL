
import random
import numpy as np
from smac.env.pettingzoo import StarCraft2PZEnv

import torch
import torch.optim as optim
from model.QMIX import *
from utils.ReplayBuffer import ReplayBuffer
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'joint_action', 'next_state', 'reward'))


def print_messages(num_epi, steps, total_rewards):
    print("# {} episode is finished".format(num_epi))
    print("Currnet steps : {}".format(steps))
    print("Episode total rewards", total_rewards)

def train(memory, batch_size, q_network, q_target, optimizer):
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    print(batch)



def main():
    """
    Runs an env object with random actions.
    """
    env = StarCraft2PZEnv.env(map_name="3m")
    max_steps = 100000
    steps = 0
    max_epi = 100
    total_rewards = 0
    done = False
    num_epi = 0
    env.reset()

    s = env.state()
    memory = ReplayBuffer(50000)

    num_agents = len(env.agents)
    obs_size = None
    act_size = None
    agent_nets = dict()
    for agent in env.agents:
        obs_size = env.observation_spaces[agent]['observation'].shape[0]
        act_size = (env.action_spaces[agent].n)
        agent_nets[agent] = RNNAgent(obs_size+1, 32,  act_size).cuda()

    #print(agent_net)

    # Epsilon scheduling
    startEpsilon = 1.0
    endEpsilon = 0.05
    epsilon = startEpsilon
    stepDrop = (startEpsilon - endEpsilon) / max_epi


    while steps < max_steps and num_epi < max_epi:
        total_rewards = 0
        env.reset()
        t = 0
        # Initialize episode
        dones = {agent: False for agent in env.agents}
        taus = dict()
        pre_actions = dict()

        pre_state =  np.zeros(obs_size)
        state = np.zeros(obs_size)
        u = np.zeros(1)
        state_count = 0

        # Initialize taus
        for agent in env.agents:
            taus[agent] = agent_nets[agent].init_hidden()
            pre_actions[agent] = np.array([0])

        # Schedule epsilon
        if (epsilon > endEpsilon):
            epsilon -= stepDrop


        for agent in env.agent_iter():
            print(env.agents)
            env.render()
            obs, reward, done, _ = env.last()
            obs_tmp = (obs['observation'])
            act_tmp = (pre_actions[agent])
            input = torch.from_numpy(np.append(obs_tmp, act_tmp))
            input = input.cuda().float()
            input = input.unsqueeze(0)
            Q, taus[agent] = agent_nets[agent].forward(input, taus[agent])

            coin = random.random()


            if done:
                action = None
                dones[agent] = True
            elif coin < epsilon:
                action = random.choice(np.flatnonzero(obs["action_mask"]))
            else:
                action = torch.argmax(Q)
                if action not in obs["action_mask"]:
                    reward = -1
                    action = 0
            total_rewards += reward

            state = np.vstack((state, obs['observation']))
            u = np.vstack((u, action))
            state_count += 1
            env.step(action)
            steps += 1
            pre_actions[agent] = action

            if(state_count % num_agents == 0 and state_count != 0):
                # Add transitions to replay buffer
                memory.put(pre_state, u, reward, state)
                t = t + 1
                print(pre_state.shape)
                print(state.shape)
                print(len(memory))

                #if(len(memory) > 500):
                #    train(memory, 32)

                pre_state = state
                state = np.zeros(obs_size)
                u = np.zeros(1)

            #if False not in dones.values():
            #    break
             # Put state, joint actions, total_rewards next_state in replay_buffer


        print_messages(num_epi, steps, total_rewards)
        num_epi += 1

    env.close()


if __name__ == "__main__":
    main()