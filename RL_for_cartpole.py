'''
Today I present to you A.I. for games. The challenge is to balance a pole by
moving the cart it is sitting on left and right.

We can solve this challenge by letting it play randomly and save the top few
episodes where it performed the best. Then train a neural network on those
few episodes.

As the training progress, the agent will gradually switch from random playing
to using the neural network.
'''
import argparse
import random
import gym
import tensorflow as tf
import numpy as np

class RL:
  def __init__(self,observation_size=4,action_size=2):
    '''
    Initializing Variables for this class
    :param observation_size: The size of the observation space
    Default : 4 (for cartpole)
    [Ex: 6]
    :param action_size: The size of the action space
    Default : 2 (for cartpole)
    [Ex: 3]
    '''
    self.action_size= action_size
    self.observation = tf.placeholder(tf.float32,[None,observation_size])
    self.labeled_moves = tf.placeholder(tf.float32,[None,action_size])

  def network(self,hidden_size=100):
    '''
    The deep neural network model where we will be using.
    :param hidden_size: Number of nodes in the hidden layers.
    Default : 100
    [Ex: 64]
    :return:Tensor Output of the network
    '''
    fc1 = tf.layers.dense(self.observation,hidden_size,activation=tf.nn.relu)
    fc2 = tf.layers.dense(fc1, hidden_size, activation=tf.nn.relu)
    d1 = tf.nn.dropout(fc2,0.8)
    fc3 = tf.layers.dense(d1, hidden_size, activation=tf.nn.relu)
    fc4 = tf.layers.dense(fc3, self.action_size, activation=None)
    return fc4

def np_softmax(x):
  '''
  External softmax function for the network as I wasn't able to integrate
  it directly into the model.
  :param x: List of Numbers
  [1,2,3]
  :return: List of numbers after softmax
  [0.09003057, 0.24472847, 0.66524096]
  '''
  return np.exp(x) / np.sum(np.exp(x),axis=0)

def main(args):
  dict_args = vars(args)

  # Let's define all of our variables
  test_episodes = dict_args['test_episodes']  # Number of testing episodes
  train_step = dict_args['train_step']  # Number of train episode
  explore_proba_decay = dict_args['explore_proba']  # Explore decay rate, how fast we want the agent to switch
  # from random to using the network. (explore_proba_decay ** steps)
  # Bear in mind that using
  # a small value may cause the model to never converge.

  select_top_best = dict_args['select_top_best']  # Select the top k examples where the model did the best
  sample = dict_args['sample']  # How much we want to sample in each train steps
  epoch = 15  # Training the neural network on the data collected
  curr_longest = 0  # sum of reward from best top k examples

  # Initialize the OpenAI gym
  env = gym.make('CartPole-v0')
  act_space = env.action_space.n
  ob_space = env.observation_space.shape[0]
  observation = env.reset()

  # Declare the deep neural network
  net = RL(observation_size=ob_space, action_size=act_space)
  output = net.network()
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,
                                                                   labels=net.labeled_moves))
  optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
  saver = tf.train.Saver()
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(train_step):
      data = []
      labels = []
      print('Step:', step)

      # Play (N = sample) times and select the top k examples
      for _ in range(sample):
        done = False
        total_reward = 0
        one_datapoint = []
        one_action = []

        while done == False:
          # Select between explore and use network by chance
          val = random.uniform(0, 1)

          if val < explore_proba_decay ** step:
            act = env.action_space.sample()

          elif val > explore_proba_decay ** step:
            obs_reshaped = np.reshape(observation, (-1, ob_space))
            final_layer = sess.run(output, feed_dict={net.observation: obs_reshaped})[0]
            layer_score = np_softmax(final_layer)
            act = np.argmax(layer_score)

          # Ensure that the state BEFORE committing the action is saved
          # rather than after
          one_datapoint.append(observation)

          observation, reward, done, info = env.step(act)

          act_one_hot = np.zeros([act_space])
          act_one_hot[act] = 1
          one_action.append(act_one_hot)

          total_reward += reward
          if dict_args['watch_it_train'] == True:
            env.render()

        observation = env.reset()

        data.append(one_datapoint)
        labels.append(one_action)

      data.sort(key=len)
      labels.sort(key=len)

      li_of_len = [len(x) for x in data[-select_top_best:]]

      # If the top k selected isn't any better than the previous
      # ones. We will omit it.
      if sum(li_of_len) < curr_longest:
        print("Long not found, Continue")
        print("Top", select_top_best, "Examples", li_of_len)
        continue
      # Else if we found better performing data,
      # we'll train it on the deep neural network
      else:
        print("Top", select_top_best, "Examples", li_of_len)
        print(li_of_len)
        curr_longest = sum(li_of_len)
        training_data = []
        training_label = []

        for datas in data[-select_top_best:]:
          training_data.extend(datas)

        for label in labels[-select_top_best:]:
          training_label.extend(label)

        for _ in range(epoch):
          a, c = sess.run([optimizer, loss], feed_dict={net.observation: training_data,
                                                        net.labeled_moves: training_label})

    # Once we've completed our training, we can watch how it performs.
    for i in range(test_episodes):
      state = env.reset()
      done = False
      while done == False:
        obs_reshaped = np.reshape(state, (-1, ob_space))
        final_layer = sess.run(output, feed_dict={net.observation: obs_reshaped})[0]
        layer_score = np_softmax(final_layer)
        act = np.argmax(layer_score)
        state, reward, done, info = env.step(act)
        env.render()
    saver.save(sess, 'model/RL')

if __name__ == '__main__':

  parser = argparse.ArgumentParser('RL For Cartpole Game')

  parser.add_argument('--watch_it_train',
                      type=bool,
                      default=False)

  parser.add_argument('--explore_proba',
                      type=int,
                      default=0.98)

  parser.add_argument('--train_step',
                      type=int,
                      default=100)

  parser.add_argument('--test_episodes',
                      type=int,
                      default=20)

  parser.add_argument('--select_top_best',
                      type=int,
                      default=3)

  parser.add_argument('--sample',
                      type=int,
                      default=15)

  args = parser.parse_args()

  main(args)

