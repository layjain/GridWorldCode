import numpy as np
import random
import os

class ReplayMemory:

    def __init__(self, config):
        self.config = config
        self.actions = np.empty((self.config.mem_size), dtype=np.int32)
        self.rewards = np.empty((self.config.mem_size), dtype=np.int32)
        self.colors = np.empty((self.config.mem_size, self.config.num_colors), dtype=np.float32)
        self.terminals = np.empty((self.config.mem_size,), dtype=np.float16)
        self.count = 0
        self.current = 0
        self.dir_save = config.dir_save + "memory/"

        if not os.path.exists(self.dir_save):
            os.makedirs(self.dir_save)

    def save(self):
        np.save(self.dir_save + "colors.npy", self.colors)
        np.save(self.dir_save + "actions.npy", self.actions)
        np.save(self.dir_save + "rewards.npy", self.rewards)
        np.save(self.dir_save + "terminals.npy", self.terminals)

    def load(self):
        self.colors = np.load(self.dir_save + "colors.npy")
        self.actions = np.load(self.dir_save + "actions.npy")
        self.rewards = np.load(self.dir_save + "rewards.npy")
        self.terminals = np.load(self.dir_save + "terminals.npy")



class DQNReplayMemory(ReplayMemory):

    def __init__(self, config):
        super(DQNReplayMemory, self).__init__(config)

        self.pre = np.empty((self.config.batch_size, self.config.history_len, self.config.num_colors), dtype=np.float32)
        self.post = np.empty((self.config.batch_size, self.config.history_len, self.config.num_colors), dtype=np.float32)

    def getState(self, index):

        index = index % self.count
        if index >= self.config.history_len - 1:
            a = self.colors[(index - (self.config.history_len - 1)):(index + 1), ...]
            return a
        else:
            indices = [(index - i) % self.count for i in reversed(range(self.config.history_len))]
            return self.colors[indices, ...]

    def add(self, color, reward, action, terminal):
        assert len(color) == self.config.num_colors

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.colors[self.current] = color
        self.terminals[self.current] = float(terminal)
        self.count = max(self.count, self.current + 1)
        ###(LJ) self.count can never exceed self.config.mem_size
        ###self.count keeps track of how much memory has already been filled
        self.current = (self.current + 1) % self.config.mem_size
        ###self.current tells the current index in the memory

    def sample_batch(self):
        ### we choose random (s, a, r, s, a) tuples from replay memory (where one state is 4=hist_len observations)
        assert self.count > self.config.history_len

        ###(LJ): Choose indices from memory which dont include the current index/terminal states
        ### *** Repetition allowed
        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(self.config.history_len, self.count-1)
                if index >= self.current and index - self.config.history_len < self.current:
                    continue

                if self.terminals[(index - self.config.history_len): index].any():
                    continue
                break
            self.pre[len(indices)] = self.getState(index - 1)
            self.post[len(indices)] = self.getState(index)
            indices.append(index)

        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminals = self.terminals[indices]

        return self.pre, actions, rewards, self.post, terminals

class DRQNReplayMemory(ReplayMemory):

    def __init__(self, config):
        super(DRQNReplayMemory, self).__init__(config)

        self.timesteps = np.empty((self.config.mem_size), dtype=np.int32)
        self.states = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update + 1, self.config.num_colors), dtype=np.float32)
        self.actions_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))
        self.rewards_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))
        self.terminals_out = np.empty((self.config.batch_size, self.config.min_history + self.config.states_to_update +1))

    def add(self, color, reward, action, terminal, t):
        assert len(color) == (self.config.num_colors)

        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.colors[self.current] = color
        self.timesteps[self.current] = t
        self.terminals[self.current] = float(terminal)
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.config.mem_size


    def getState(self, index):
        a = self.colors[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a

    def get_scalars(self, index):
        t = self.terminals[index - (self.config.min_history + self.config.states_to_update + 1): index]
        a = self.actions[index - (self.config.min_history + self.config.states_to_update + 1): index]
        r = self.rewards[index - (self.config.min_history + self.config.states_to_update + 1): index]
        return a, t, r

    def sample_batch(self):
        ###(LJ) we choose random [(s, a, r,) sequences ] from replay memory 
        ### ... (where one sequence is [hist_len+states_to_update+1] observations)
        assert self.count > self.config.min_history + self.config.states_to_update

        indices = []
        while len(indices) < self.config.batch_size:

            while True:
                index = random.randint(self.config.min_history, self.count-1)
                if index >= self.current and index - self.config.min_history < self.current:
                    ### does not include current
                    continue
                if index < self.config.min_history + self.config.states_to_update + 1:
                    ### for getState to work
                    continue
                if self.timesteps[index] < self.config.min_history + self.config.states_to_update:
                    ### so that getState extracts the past colors of the same episode
                    continue
                break
            self.states[len(indices)] = self.getState(index)
            self.actions_out[len(indices)], self.terminals_out[len(indices)], self.rewards_out[len(indices)] = self.get_scalars(index)
            indices.append(index)


        return self.states, self.actions_out, self.rewards_out, self.terminals_out
