""" TF model for CartPole """

import collections
import random

import numpy as np
from tensorflow.keras import layers, models, optimizers


class Model:

    def __init__(self, observation_space, action_space, learning_rate=0.001, discount=0.95, batch_size=20,
                 initial_exploration_rate=1.0):
        """
        Init function
        :param observation_space: shape of the observation space
        :param action_space: shape of the action space
        :param learning_rate: float - model learning rate
        :param discount: float - discount rate for reward update
        :param batch_size: int - size of the batch used for training
        :param initial_exploration_rate - float - used for telling the model how to explore
        """
        self.exploration_rate = initial_exploration_rate

        self.learning_rate = learning_rate
        self.discount = discount
        self.batch_size = batch_size
        self.action_space = action_space
        self.observation_space = observation_space
        self.action_space = action_space
        self.memory = collections.deque(maxlen=10000)
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24,
                               input_shape=(self.observation_space,),
                               activation='relu'))
        model.add(layers.Dense(24,
                               activation='relu'))
        model.add(layers.Dense(24,
                               activation='relu'))
        model.add(layers.Dense(self.action_space,
                               activation='linear'))
        model.compile(loss='mse',
                      optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def get_action(self, state):
        """
        Predict an action given a state
        :param state: [position, velocity, pole angle, velocity at pole tip] - state of the cart pole
        :return: float - action indicating whether to go left or right
        """
        # if the exploration rate is high, the model's prediction is no better than a random one
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        print('predicted q_values:      ', q_values)
        return np.argmax(q_values[0])

    def learn(self):
        """ Make the model learn from stored memory """
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + self.discount * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.update_exploration_rate()

    def update_exploration_rate(self):
        """ Updates the exploration rate """
        exploration_min = 0.01
        decay_rate = 0.995
        self.exploration_rate = self.exploration_rate * decay_rate
        self.exploration_rate = max(exploration_min, self.exploration_rate)

    def store(self, current_state, action, reward, next_state, done):
        """ Store the given tuple in memory """
        self.memory.append((current_state, action, reward, next_state, done))
