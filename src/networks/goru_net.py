import numpy as np
import os
import tensorflow as tf
import shutil
from functools import reduce
from tensorflow.python import debug as tf_debug
from src.utils import conv2d_layer, fully_connected_layer, stateful_goru, huber_loss
from src.networks.base import BaseModel


class GORU_NETWORK(BaseModel):

    def __init__(self, n_actions, config):
        super(GORU_NETWORK, self).__init__(config, "goru")
        self.n_actions = n_actions
        self.num_colors = config.num_colors
        self.num_lstm_layers = config.num_lstm_layers
        self.lstm_size = config.lstm_size
        self.min_history = config.min_history
        self.states_to_update = config.states_to_update

    def add_placeholders(self):
        self.w = {}
        self.w_target = {}
        self.state = tf.placeholder(tf.float32, shape=[None, 1, self.num_colors],
                                    name="input_state")
        self.action = tf.placeholder(tf.int32, shape=[None], name="action_input")
        self.reward = tf.placeholder(tf.int32, shape=[None], name="reward")
        self.state_target = tf.placeholder(tf.float32,
                                           shape=[None, 1, self.num_colors],
                                           name="input_target")
        # create placeholder to fill in lstm state
        self.c_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_c")
        self.h_state_train = tf.placeholder(tf.float32, [None, self.lstm_size], name="train_h")
        self.lstm_state_train = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_train, self.h_state_train)



        self.c_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_c")
        self.h_state_target = tf.placeholder(tf.float32, [None, self.lstm_size], name="target_h")
        self.lstm_state_target = tf.nn.rnn_cell.LSTMStateTuple(self.c_state_target, self.h_state_target)

        # initial zero state to be used when starting episode
        self.initial_zero_state_batch = np.zeros((self.batch_size, self.lstm_size))
        self.initial_zero_state_single = np.zeros((1, self.lstm_size))

        self.initial_zero_complete = np.zeros((self.num_lstm_layers, 2, self.batch_size, self.lstm_size))

        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")
        self.target_val = tf.placeholder(dtype=tf.float32, shape=[None], name="target_val")
        self.terminal = tf.placeholder(dtype=tf.float32, shape=[None], name="terminal")
        self.target_val_tf = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions])

    def add_logits_op_train(self):
        x=self.state
        out, state = stateful_goru(x=x, num_layers=self.num_lstm_layers, lstm_size=self.lstm_size, state_input=tuple([self.lstm_state_train]),
                                               scope_name="lstm_train")
        self.state_output_c = state[0][0]
        self.state_output_h = state[0][1]
        shape = out.get_shape().as_list() #[None,1,512]

        out = tf.reshape(out, [tf.shape(out)[0], shape[2]])
        w, b, out = fully_connected_layer(out, self.n_actions, scope_name="out_train", activation=None)

        self.w["wout"] = w
        self.w["bout"] = b

        self.q_out = out
        self.q_action = tf.argmax(self.q_out, axis=1)

    def add_logits_op_target(self):
        x = self.state_target #shape: [None, 1, num_colors]

        out, state = stateful_goru(x, self.num_lstm_layers, self.lstm_size,
                                                      tuple([self.lstm_state_target]), scope_name="lstm_target")
        self.state_output_target_c = state[0][0]
        self.state_output_target_h = state[0][1]
        shape = out.get_shape().as_list()

        out = tf.reshape(out, [tf.shape(out)[0], shape[2]])

        w, b, out = fully_connected_layer(out, self.n_actions, scope_name="out_target", activation=None)

        self.w_target["wout"] = w
        self.w_target["bout"] = b

        self.q_target_out = out
        self.q_target_action = tf.argmax(self.q_target_out, axis=1)

    def train_on_batch_target(self, states, action, reward, terminal, steps):
        states = states #(batch_size, min_hist+states_to_update+1, num_colors)
        q, loss = np.zeros((self.batch_size, self.n_actions)), 0
        states = np.transpose(states, [1, 0, 2])
        action = np.transpose(action, [1, 0])
        reward = np.transpose(reward, [1, 0])
        terminal = np.transpose(terminal, [1, 0])
        states = np.reshape(states, [states.shape[0], states.shape[1], 1, states.shape[2]])
        #states.shape should be (min_hist+states_to_update+1, batch_size, 1, num_colors)
        lstm_state_c, lstm_state_h = self.initial_zero_state_batch, self.initial_zero_state_batch
        #LJ: initial unroll target lstm: for 1 state, starting from zero state
        lstm_state_target_c, lstm_state_target_h = self.sess.run(
            [self.state_output_target_c, self.state_output_target_h],
            {
                self.state_target: states[0],
                self.c_state_target: self.initial_zero_state_batch,
                self.h_state_target: self.initial_zero_state_batch
            }
        )
        for i in range(self.min_history):
            #LJ: unroll both the networks for min_history steps
            j = i + 1
            lstm_state_c, lstm_state_h, lstm_state_target_c, lstm_state_target_h = self.sess.run(
                [self.state_output_c, self.state_output_h, self.state_output_target_c, self.state_output_target_h],
                {
                    self.state: states[i],
                    self.state_target: states[j],
                    self.c_state_target: lstm_state_target_c,
                    self.h_state_target: lstm_state_target_h,
                    self.c_state_train: lstm_state_c,
                    self.h_state_train: lstm_state_h
                }
            )
        for i in range(self.min_history, self.min_history + self.states_to_update):
            #LJ: First, Unroll the target network
            j = i + 1
            target_val, lstm_state_target_c, lstm_state_target_h = self.sess.run(
                [self.q_target_out, self.state_output_target_c, self.state_output_target_h],
                {
                    self.state_target: states[j],
                    self.c_state_target: lstm_state_target_c,
                    self.h_state_target: lstm_state_target_h
                }
            )
            max_target = np.max(target_val, axis=1)
            #LJ: to figure out max{a}[Q(a, theta_prime)]
            target = (1. - terminal[i]) * self.gamma * max_target + reward[i]
            _, q_, train_loss_, lstm_state_c, lstm_state_h = self.sess.run(
                [self.train_op, self.q_out, self.loss, self.state_output_c, self.state_output_h],
                feed_dict={
                    self.state: states[i],
                    self.c_state_train: lstm_state_c,
                    self.h_state_train: lstm_state_h,
                    self.action: action[i],
                    self.target_val: target,
                    self.lr: self.learning_rate
                }
            )
            ###LJ: train_op trains
            q += q_
            loss += train_loss_
        if steps % 20000 == 0 and steps > 50000:
            self.learning_rate *= self.lr_decay  # decay learning rate
            if self.learning_rate < self.learning_rate_minimum:
                self.learning_rate = self.learning_rate_minimum
        self.train_steps += 1
        return q.mean(), loss / (self.states_to_update)



    def add_loss_op_target(self):
        action_one_hot = tf.one_hot(self.action, self.n_actions, 1.0, 0.0, name='action_one_hot')
        train = tf.reduce_sum(self.q_out * action_one_hot, reduction_indices=1, name='q_acted')
        self.delta = train - self.target_val
        self.loss = tf.reduce_mean(huber_loss(self.delta))

        avg_q = tf.reduce_mean(self.q_out, 0)
        q_summary = []
        for i in range(self.n_actions):
            q_summary.append(tf.summary.histogram('q/{}'.format(i), avg_q[i]))
        self.avg_q_summary = tf.summary.merge(q_summary, 'q_summary')
        self.loss_summary = tf.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        self.add_logits_op_train()
        self.add_logits_op_target()
        self.add_loss_op_target()
        self.add_train_op(self.lr_method, self.lr, self.loss, clip=10)
        self.initialize_session()
        self.init_update()

    def update_target(self):
        for name in self.w:
            self.target_w_assign[name].eval({self.target_w_in[name]: self.w[name].eval(session=self.sess)},
                                            session=self.sess)
        for var in self.lstm_vars:
            self.target_w_assign[var.name].eval({self.target_w_in[var.name]: var.eval(session=self.sess)},
                                                session=self.sess)

    def init_update(self):
        self.target_w_in = {}
        self.target_w_assign = {}
        for name in self.w:
            self.target_w_in[name] = tf.placeholder(tf.float32, self.w_target[name].get_shape().as_list(), name=name)
            self.target_w_assign[name] = self.w_target[name].assign(self.target_w_in[name])

        self.lstm_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm_train")
        lstm_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="lstm_target")

        for i, var in enumerate(self.lstm_vars):
            self.target_w_in[var.name] = tf.placeholder(tf.float32, var.get_shape().as_list())
            self.target_w_assign[var.name] = lstm_target_vars[i].assign(self.target_w_in[var.name])
