'''
Test for env_wrapper
'''
import time
from src.config import base_config
conf=base_config()

from src.env_wrapper import GridWorldWrapper


env = GridWorldWrapper(conf)

env.render()
for a in [0,1,2,3, 1, 1,1,1,1,2,2,2]:
	print('taking action:', a)
	env.act(a)
	env.render()
	print('reward:',env.reward, 'terminal?:',env.terminal, 'color:', env.color)
	