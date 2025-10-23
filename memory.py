from collections import namedtuple, deque
import numpy as np

Transition = namedtuple('Transition', ('state', 'terminated', 'truncated', 'awake', 'action', 'probs', 'value', 'next_state', 'reward'))

class PPOMemory(object):
    def __init__(self, batch_size):
        self._transitions = [] 
        self._batch_size = batch_size

    def generate_batches(self):
        n_states = len(self._transitions)
        batch_start = np.arange(0, n_states, self._batch_size)
        indicies = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indicies)
        batches = [indicies[i:i+self._batch_size] for i in batch_start]
        return self._transitions, batches

    def push(self, *args):
        """Save a transition"""
        self._transitions.append(Transition(*args))

    def __len__(self):
        return len(self._transitions)

    def clear(self):
        self._transitions = [] 
