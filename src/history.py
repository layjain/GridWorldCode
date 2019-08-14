import numpy as np
import os

class History():
    '''
    History just stores the current state
    '''
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.history_len = config.history_len
        self.num_colors = config.num_colors
        self.history = np.zeros((self.history_len, self.num_colors), dtype=np.float32)

    def add(self, color):
        #color is represented as one-hot encoded vector
        self.history[:-1] = self.history[1:]
        self.history[-1] = color

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history