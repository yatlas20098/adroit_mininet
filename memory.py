from collections import namedtuple, deque
import random
import numpy as np

Transition = namedtuple('Transition', ('state', 'terminated', 'truncated', 'awake', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self._memory = deque([], maxlen=capacity)

    def push(self, *args):
        self._memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self._memory, batch_size)

    def __len__(self):
        return len(self._memory)

    def clear(self):
        self._memory = [] 
