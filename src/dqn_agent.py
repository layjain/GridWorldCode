from src.agent import BaseAgent
from src.history import History
from src.replay_memory import DQNReplayMemory
from src.networks.dqn import DQN
import numpy as np
from tqdm import tqdm

class DQNAgent(BaseAgent):

    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.history = History(config)
        self.replay_memory = DQNReplayMemory(config)
        self.net = DQN(4, config)
        self.net.build()
        self.net.add_summary(["average_reward", "average_loss", "average_q", "ep_max_reward", "ep_avg_reward","ep_min_reward", "ep_num_game", "learning_rate"], ["ep_rewards", "ep_actions"])

    def observe(self):
        reward = max(self.min_reward, min(self.max_reward, self.env_wrapper.reward))
        color = self.env_wrapper.color
        self.history.add(color)
        self.replay_memory.add(color, reward, self.env_wrapper.action, self.env_wrapper.terminal)
        if self.i < self.config.epsilon_decay_episodes:
            self.epsilon -= self.config.epsilon_decay
        if self.i % self.config.train_freq == 0 and self.i > self.config.train_start:
            #print('----> i',self.i)
            ### training starts only after train_start=20K steps, mem_size is 800K, train_freq is 8,
            ### I guess thats why its okay to not worry about sampling with repititions
            state, action, reward, state_, terminal = self.replay_memory.sample_batch()
            q, loss = self.net.train_on_batch_target(state, action, reward, state_, terminal, self.i)
            ### self.i is passed to implement lr decay
            self.total_q += q
            self.total_loss += loss
            self.update_count += 1
        if self.i % self.config.update_freq == 0:
            self.net.update_target()

    def policy(self):
        if np.random.rand() < self.epsilon:
            return self.env_wrapper.random_step()
        else:
            state = self.history.get()
            a = self.net.q_action.eval({
                self.net.state : [state]
            }, session=self.net.sess)
            return a[0]


    def train(self, steps):
        f=open('dqn2.txt', 'w')
        render = False
        self.env_wrapper.new_random_game()
        num_game, self.update_count, ep_reward = 0,0,0.
        total_reward, self.total_loss, self.total_q = 0.,0.,0.
        ep_rewards, actions = [], []
        t = 0
        print(self.net.number_of_trainable_parameters())

        for _ in range(self.config.history_len):
            self.history.add(self.env_wrapper.color)
            ### So, the first state is just the first color, repeated a number of times
        for self.i in tqdm(range(self.i, steps)):
            #take action, observe
            action = self.policy()
            self.env_wrapper.act(action)
            self.observe()
            if self.env_wrapper.terminal:
                t = 0
                self.env_wrapper.new_random_game()
                num_game += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += self.env_wrapper.reward
                t += 1
            actions.append(action)
            total_reward += self.env_wrapper.reward
            #print(self.i,action,total_reward, self.env_wrapper.terminal)
            #total_reward, max_ep_reward, min_ep_reward, avg_ep_reward keep track of reward earned every self.config.test_step=5000 steps
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
                        'ep_num_game': num_game,
                        'ep_avg_reward': avg_ep_reward,
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
                #play_score = self.play(episodes=self.config.num_episodes_for_play_scores_summary, net_path=self.net.dir_model)
                #self.net.inject_summary({'play_score':play_score}, self.i)
            if self.i % 100000 == 0:
                j = 0
                render = True

            if render:
                #self.env_wrapper.env.render()
                j += 1
                if j == 1000:
                    render = False
        f.close()

    def play(self, episodes, net_path, verbose=False, print_average=True):
        d=[]
        self.net.restore_session(path=net_path)
        self.env_wrapper.new_game()
        i = 0
        for _ in range(self.config.history_len):
            self.history.add(self.env_wrapper.color)
        episode_steps = 0
        ###EDIT (LJ): added rewards calculation
        episode_reward = 0
        actions_list=[]
        while i < episodes:
            #Chose Action:
            a = self.net.q_action.eval({
                self.net.state : [self.history.get()]
            }, session=self.net.sess)
            action = a[0]
            actions_list.append(action)
            #Take Action
            self.env_wrapper.act_play(action)
            self.history.add(self.env_wrapper.color)
            episode_steps += 1
            episode_reward+=self.env_wrapper.reward
            if episode_steps > self.config.max_steps:
                self.env_wrapper.terminal = True
            if self.env_wrapper.terminal:
                if verbose:
                    print('episode terminated in '+str(episode_steps)+' steps with reward '+str(episode_reward))
                    print('ACTIONS TAKEN:')
                    print(actions_list)
                actions_list=[]
                d.append(episode_reward)
                episode_steps = 0
                episode_reward = 0
                i += 1
                self.env_wrapper.new_play_game()
                for _ in range(self.config.history_len):
                    color = self.env_wrapper.color
                    self.history.add(color)
        if verbose:
            print('ALL, AVERAGE:',[d, sum(d)/len(d)])
        if print_average:
            print('AVERAGE:',sum(d)/len(d))
        return sum(d)/len(d)
