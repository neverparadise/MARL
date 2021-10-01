import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class QMIX(nn.Module):
    def __init__(self, num_agents):
        super(QMIX, self).__init__()
        self.num_agents = num_agents
        self.state_dim = None
        self.obs_dim = None



import random
from smac.env.pettingzoo import StarCraft2PZEnv


def main():
    """
    Runs an env object with random actions.
    """
    env = StarCraft2PZEnv.env(map_name="8m")

    episodes = 10

    total_reward = 0
    done = False
    completed_episodes = 0

    while completed_episodes < episodes:
        env.reset()
        print(env.agents)
        print(env.action_spaces)
        for agent in env.agent_iter():
            env.render()

            obs, reward, done, _ = env.last()
            total_reward += reward
            if done:
                action = None
            elif isinstance(obs, dict) and "action_mask" in obs:
                action = random.choice(np.flatnonzero(obs["action_mask"]))
            else:
                action = env.action_spaces[agent].sample()
            env.step(action)

        completed_episodes += 1

    env.close()

    print("Average total reward", total_reward / episodes)


if __name__ == "__main__":
    main()