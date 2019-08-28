from src.agent import BaseAgent
from src.replay_memory import DRQNReplayMemory
#same replay memory functionality should be used
from src.networks.eunn_net import EUNN_NETWORK
import numpy as np
from tqdm import tqdm

class EUNNAgent(BaseAgent):

    def __init__(self, config):
        super(EUNNAgent, self).__init__(config)
        self.replay_memory = DRQNReplayMemory(config)
        self.net = EUNN_NETWORK(4, config)
        self.net.build()
        self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_min_reward","ep_avg_reward" ,"ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

    def observe(self, t):
        reward = max(self.min_reward, min(self.max_reward, self.env_wrapper.reward))
        self.color = self.env_wrapper.color
        self.replay_memory.add(self.color, reward, self.env_wrapper.action, self.env_wrapper.terminal, t)
        if self.i < self.config.epsilon_decay_episodes:
            self.epsilon -= self.config.epsilon_decay
        if self.i % self.config.train_freq == 0 and self.i > self.config.train_start:
            states, action, reward, terminal = self.replay_memory.sample_batch()
            q, loss= self.net.train_on_batch_target(states, action, reward, terminal, self.i)
            self.total_q += q
            self.total_loss += loss
            self.update_count += 1
        if self.i % self.config.update_freq == 0:
            self.net.update_target()

    def policy(self, state):
        self.random = False
        if np.random.rand() < self.epsilon:
            self.random = True
            return self.env_wrapper.random_step()
        else:
            a, self.lstm_state_c, self.lstm_state_h = self.net.sess.run([self.net.q_action, self.net.state_output_c, self.net.state_output_h],{
                self.net.state : [[state]],
                self.net.c_state_train: self.lstm_state_c,
                self.net.h_state_train: self.lstm_state_h
            })
            return a[0]


    def train(self, steps):
        f=open('eunn2.txt', 'w')
        render = False
        self.env_wrapper.new_random_game()
        num_game, self.update_count, ep_reward = 0,0,0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        ep_rewards, actions = [], []
        t = 0
        print(self.net.number_of_trainable_parameters())
        self.color = self.env_wrapper.color
        self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single

        for self.i in tqdm(range(self.i, steps)):
            state = self.color
            #NOTE (LJ): In DRQN, state is just one observation
            #(LJ) In DQN, state is hist_len observations, stacked.
            action = self.policy(state)
            self.env_wrapper.act(action)
            if self.random:
                self.lstm_state_c, self.lstm_state_h = self.net.sess.run([self.net.state_output_c, self.net.state_output_h], {
                    self.net.state: [[state]],
                    self.net.c_state_train : self.lstm_state_c,
                    self.net.h_state_train: self.lstm_state_h
                })
            self.observe(t)
            if self.env_wrapper.terminal:
                t = 0
                self.env_wrapper.new_random_game()
                self.color = self.env_wrapper.color
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
                self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single
            else:
                ep_reward += self.env_wrapper.reward
                t += 1
            actions.append(action)
            total_reward += self.env_wrapper.reward
            #print(self.i,action,total_reward, self.env_wrapper.terminal)

            if self.i >= self.config.train_start:
                if self.i % self.config.test_step == self.config.test_step - 1:
                    avg_reward = total_reward / self.config.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    sum_dict = {
                        'average_reward': avg_reward,
                        'average_loss': avg_loss,
                        'average_q': avg_q,
                        'ep_max_reward': max_ep_reward,
                        'ep_min_reward': min_ep_reward,
                        'ep_avg_reward': avg_ep_reward,
                        'ep_num_game': num_game,
                        'learning_rate': self.net.learning_rate,
                        'ep_rewards': ep_rewards,
                        'ep_actions': actions
                    }
                    self.net.inject_summary(sum_dict, self.i)
                    num_game = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []
                    f.write(str(avg_ep_reward))

            if self.i % 50000 == 0 and self.i > 0:
                j = 0
                print('saving..')
                self.save()
                play_score = self.play(episodes=self.config.num_episodes_for_play_scores_summary, net_path=self.net.dir_model)
                print('play_score:', play_score)
                self.net.inject_summary({'play_score':play_score}, self.i)
            if self.i % 100000 == 0:
                j = 0
                render = True

            if render:
                #self.env_wrapper.env.render()
                j += 1
                if j == 1000:
                    render = False
        f.close()

    def play(self, episodes, net_path, verbose=False):
        self.net.restore_session(path=net_path)
        self.env_wrapper.new_game()
        self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single
        i = 0
        episode_steps = 0
        #(LJ):ADDED Episode Reward info
        actions_list=[]
        episode_reward=0
        all_rewards=[]
        while i < episodes:
            a, self.lstm_state_c, self.lstm_state_h = self.net.sess.run([self.net.q_action, self.net.state_output_c, self.net.state_output_h],{
                self.net.state : [[self.env_wrapper.color]],
                self.net.c_state_train: self.lstm_state_c,
                self.net.h_state_train: self.lstm_state_h
            })
            action = a[0]
            actions_list.append(action)

            if episode_steps==0:
                print('coords at the start:', self.env_wrapper.env.loc)
            self.env_wrapper.act_play(action)
            episode_steps += 1
            episode_reward+=self.env_wrapper.reward
            if episode_steps > self.config.max_steps:
                self.env_wrapper.terminal = True
            if self.env_wrapper.terminal:
                if verbose:
                    print('episode terminated in '+str(episode_steps)+' steps with reward '+str(episode_reward))
                all_rewards.append(episode_reward)
                if verbose:
                    print('ACTIONS TAKEN:')
                    print(actions_list)
                actions_list=[]
                episode_steps = 0
                episode_reward=0
                i += 1
                self.env_wrapper.new_play_game()
                self.lstm_state_c, self.lstm_state_h = self.net.initial_zero_state_single, self.net.initial_zero_state_single
        if verbose:
            print('ALL REWARDS:')
            print(all_rewards)
            print('AVERAGE')
            print(sum(all_rewards)/len(all_rewards))
        return sum(all_rewards)/len(all_rewards)
