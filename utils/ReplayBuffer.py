from collections import namedtuple, deque
from itertools import count
import math
import random
from copy import deepcopy

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def put(self, episode):
        epi = deepcopy(episode)
        self.memory.append(epi)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Episode:
    def __init__(self, episode_limit):
        self.trajectory = deque(maxlen=episode_limit)

    def add(self, s, u, r, ns, dones, obs_dict):
        dict = {'state': s, 'action': u, 'reward': r,
                'next_state': ns , 'dones': dones, 'all_obs':obs_dict}
        self.trajectory.append(deepcopy(dict))

    def __len__(self):
        return len(self.trajectory)
