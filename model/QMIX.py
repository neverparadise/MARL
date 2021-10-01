import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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