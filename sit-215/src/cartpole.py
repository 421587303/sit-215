import sys

import random

import copy

import collections

import torch

import torch.nn as nn

import gym


env = gym.make('CartPole-v1')


if sys.argv[-1] == '--train':
    net = nn.Sequential(
        nn.Linear( 4, 24),
        nn.ReLU(),

        nn.Linear(24, 24),
        nn.ReLU(),

        nn.Linear(24,  2),
    )

    online = net

    target = copy.deepcopy(net)

    memory = collections.deque(maxlen=1000)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(online.parameters())

    e = 1

    i = 0

    try:
        episode = 0

        while True:
            done = False

            state = env.reset()

            score = 0

            while not done:
                # debug

                # env.render()

                if i % 10 == 0:
                    target.load_state_dict(online.state_dict())

                # e-greedy strategy

                if random.random() < e:
                    action = random.randrange(2)
                else:
                    s = torch.Tensor(state).unsqueeze(0)

                    action = online(s).argmax(1).item()

                # decrease

                e = max(.01, e - 9e-4)

                state_, reward, done, info = env.step(action)

                # customize reward function

                if done and score < 480:
                    reward = -10
                if done and score < 300:
                    reward = -20
                if done and score < 100:
                    reward = -30

                memory.append((state, action, reward, state_))

                # replay

                if len(memory) > 32:
                    s, a, r, x = zip(*random.sample(memory, 32))

                    s = torch.Tensor(s)

                    a = torch.Tensor(a)

                    r = torch.Tensor(r)

                    x = torch.Tensor(x)

                    z = online(s)

                    y = z.clone().detach()

                    y.scatter_(1, a.long().unsqueeze(1), r.unsqueeze(1) + .95 * target(x).gather(1, online(x).argmax(1).unsqueeze(1)))

                    loss = criterion(z, y)

                    optimizer.zero_grad()

                    loss.backward()

                    optimizer.step()

                state = state_

                i += 1

                score += 1

            episode += 1

            print('Episode %d - Score %d - Epsilon %f' % (episode, score, e))
    finally:
        torch.save(net.state_dict(), 'cartpole.pt')
else:
    s = env.reset()

    if sys.argv[-1] == '--test':
        state_dict = torch.load('cartpole.pt')

        net = nn.Sequential(
            nn.Linear( 4, 24),
            nn.ReLU(),

            nn.Linear(24, 24),
            nn.ReLU(),

            nn.Linear(24,  2),
        )

        net.load_state_dict(state_dict)

        policy = lambda s: net(torch.Tensor(s).unsqueeze(0)).argmax(1).item()

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