import time

import sys

import numpy as np

import gym


# environment

env = gym.make('Taxi-v3')

if sys.argv[-1] == '--train':
    S = env.observation_space.n

    A = env.action_space.n


    # hyperparams

    episodes = 1000000

    epsilon = 1

    gamma = 0.6

    alpha = 0.1


    # Q-table

    values = np.zeros((S, A))


    try:
        # Q-learning

        for i in range(episodes):
            # state

            s = env.reset()

            score = 0

            done = False

            while not done:
                # debug

                # time.sleep(0.01)

                # env.render()

                if np.random.random() < epsilon:
                    a = env.action_space.sample()
                else:
                    a = values[s].argmax()

                s_, r, done, _ = env.step(a)

                # update

                values[s, a] = (1 - alpha) * values[s, a] + alpha * (r + gamma * values[s_].max())

                s = s_

                # decrease

                epsilon = max(epsilon - 0.0000001, 0.1)

                # gamma = max(gamma - 0.0000001, 0.1)

                # alpha = max(alpha - 0.0000001, 0.0001)

                score += r

            # debug

            # env.close()

            print('Episode %d - Epsilon %f - Gamma %f - Alpha %f - Score %d' % (i, epsilon, gamma, alpha, score))
    finally:
        np.save('taxi.npy', values)
else:
    s = env.reset()

    if sys.argv[-1] == '--test':
        values = np.load('taxi.npy')

        policy = lambda s: values[s].argmax()

    if sys.argv[-1] == '--random':
        policy = lambda s: env.action_space.sample()

    done = False

    score = 0

    penalty = 0

    while not done:
        env.render()

        a = policy(s)

        s_, r, done, _ = env.step(a)

        s = s_

        score += r

        if r == -10:
            penalty += 1

    env.close()

    print('Score %d Penalty %d' % (score, penalty))