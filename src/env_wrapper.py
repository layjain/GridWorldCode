import numpy as np
import time
import matplotlib.pyplot as plt
#from src.config import base_config

class Grid(object):
	'''
	Generate a grid object, with associated reward, color, coordinates
	'''

	def __init__(self, config):
		self.W, self.H = config.grid_dimensions
		self.num_colors = config.num_colors
		self.action_space = [0,1,2,3]
		self.empty_color = [0 for _ in range(self.num_colors)]
		self.max_steps = config.max_steps
		all_colors = list(range(self.num_colors))
		self.info={}
		#self.all_colors = list(map(index_to_color,all_colors))
		seed = config.grid_maker_random_seed
		np.random.seed(seed)
		self.coords = [(a,b) for a in range(self.W) for b in range(self.H)]
		self.coords_set = set(self.coords)
		self.colors = {coord : self.index_to_color(np.random.choice(all_colors)) for coord in self.coords}
		self.rewards = {coord : float(np.random.uniform(config.min_reward, config.max_reward)) for coord in self.coords}
		###STATE:
		self.loc = config.start_coords
		self.color = self.colors[self.loc]
		self.reward = 0
		self.num_steps = 0
		###
		self.action_changes =[(-1, 0), (0, 1), (1, 0), (0, -1)] #translations

	def index_to_color(self, color_index):
		color = self.empty_color[:]
		color[color_index] = 1
		return color

	def step(self, action):
		change = self.action_changes[action]
		self.num_steps+=1
		loc = self.loc
		new_loc = (loc[0] + change[0], loc[1] + change[1])
		if new_loc in self.coords_set:
			self.loc = new_loc
			self.reward = self.rewards[new_loc]
			self.color = self.colors[new_loc]
			#REWARD GIVEN WHILE ENTERIMNG THE NEW LOCATION
			#TRYOUT: Exiting
			return [self.color, self.reward, self.terminal, self.info]
		else:
			# ZERO REWARD FOR TAKING ACTION THAT LEADS TO NO-WHERE
			# DISCUSS AND TRYOUT OTHER VARIANTS
			# CHANGE NUM_STEPS??
			# self.reward = 0
			self.color = self.colors[loc]
			return [self.color, self.reward, self.terminal, self.info]

	def color2str(self, c):
		for e in range(len(c)):
			if c[e] == 1:
				break
		return str(e/len(c))

	def render(self):
		fig = plt.figure()
		ax = fig.gca()
		ax.set_xticks(np.arange(0, self.W, 1))
		ax.set_yticks(np.arange(0, self.H, 1))
		plt.grid()
		for c in self.coords:
			plt.plot(c[0], c[1], color = self.color2str(self.color))
		plt.plot(self.loc[0], self.loc[1],'ro', label = 'LOC')
		plt.legend()
		plt.title('STEP: '+str(self.num_steps))
		plt.show()

	@property
	def terminal(self):
		return self.num_steps >= self.max_steps


class GridWorldWrapper(object):
	
	def __init__(self, config):
		self.W, self.H = config.grid_dimensions
		self.coords = config.start_coords
		self.env = Grid(config)
		#Let: 0 action means nothing

	def new_game(self):
		self.coords = self.config.start_coords
		self.reward=0
		self.action=0

	def new_random_game(self):
		self.new_game()

	def _step(self, action):
		self.action = action
		_, self.reward, self.terminal, self.info = self.env.step(action)

	def random_step(self):
		np.random.seed()
		return np.random.choice(self.env.action_space)

	def act(self, action):
		self._step(action)

	def new_play_game(self):
		self.new_game()

	def render(self):
		self.env.render()

	@property
	def color(self):
		return self.env.color