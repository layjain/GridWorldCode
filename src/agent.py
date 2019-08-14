import numpy as np
from src.env_wrapper import GridWorldWrapper

class BaseAgent():

    def __init__(self, config):
        self.config = config
        self.env_wrapper = GridWorldWrapper()
        self.rewards = 0
        self.epsilon = config.epsilon_start
        if config.reward_clipping == None:
            self.min_reward = -float('inf')
            self.max_reward = float('inf')
            
        self.replay_memory = None
        self.history = None
        self.net = None
        if self.config.restore:
            self.load()
        else:
            self.i = 0



    def save(self):
        self.replay_memory.save()
        self.net.save_session()
        print('with i being '+str(self.i))
        np.save(self.config.dir_save+'step.npy', self.i)

    def load(self):
        self.replay_memory.load()
        self.net.restore_session()
        self.i = np.load(self.config.dir_save+'step.npy')