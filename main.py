""" Main file """

import gym
import numpy as np
import rl

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    model = rl.Model(observation_space, action_space)
    while True:
        current_state = env.reset()
        current_state = np.reshape(current_state, (1, observation_space))
        while True:
            env.render()
            action = model.get_action(current_state)
            next_state, reward, done, info = env.step(action)
            print('state info:      ', next_state)

            # penalize for being far from the center and a big pole angle
            reward = reward - 2 * (abs(next_state[0]) + abs(next_state[2])) if not done else -reward
            print(reward)
            print()
            next_state = np.reshape(next_state, (1, observation_space))
            model.store(current_state, action, reward, next_state, done)
            current_state = next_state
            if done:
                break
            model.learn()
