class base_config():
	#TRYOUT: the state fed to LSTM comprise of only 1 screen for DRQN unlike and then LSTM is unrolled
	#....... We can feed a bunch of screens into LSTM as a single 'state' as done in DQN
	reward_clipping = None
	#TRYOUT: Changing the Networks
	#TRYOUT: Clipping the Rewards
	#TRYOUT: A function of Rewards
	#TRYOUT: Removing repetition from sample_batch in replay_memory
	#....... because over-repeating some episode can be a problem; Infact, what about in the very start?
	#....... thats why train starts at 20K?
	#TRYOUT: Variants as discussed in env_wrapper.py
	#Note: Need to change hard-coded numbers in dqn.py based on no. of train steps
	num_colors = 4
	grid_dimensions = (16,16)
	start_coords=(8,8) #Must be tuple-hashable
	grid_maker_random_seed = 0
	min_reward = 0
	max_reward = 1.0
	reward_distribution = 'uniform'
	reward_noise = None

	train_steps = 10000000
	batch_size = 64
	history_len = 1 #4 for Atari
	epsilon_start = 1.0
	epsilon_end = 0.02
	max_steps = 100
	epsilon_decay_episodes_fraction=0.1
	epsilon_decay_episodes = train_steps*epsilon_decay_episodes_fraction #This is a float
	train_freq = 8
	### TRYOUT: 8 was for Atari. Must change to a lower value?
	update_freq = 10000
	train_start = 20000
	dir_save = "saved_session/"
	restore = False
	epsilon_decay = float((epsilon_start - epsilon_end))/float(epsilon_decay_episodes)
	#Epsilon decays by a constant amount after each episode
	#We can also TRYOUT: let epsilon decay by a constant Factor
	#random_start = 10
	test_step = 5000
	network_type = "dqn"


	gamma = 0.99
	learning_rate_minimum = 0.00025
	lr_method = "rmsprop"
	learning_rate = 0.00025
	lr_decay = 0.97
	keep_prob = 0.8
	num_episodes_for_play_scores_summary=100

	num_lstm_layers = 1
	lstm_size = 64 #original 512
	min_history = 16
	states_to_update = 4
	mem_size = 800000