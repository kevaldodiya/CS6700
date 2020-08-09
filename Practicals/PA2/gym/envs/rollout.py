#!/usr/bin/env python

import gym
import chakra
import vishamC
import numpy as np
import matplotlib.pyplot as plt
import time
import click
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def policy_gradient(lr=0.1, gamma=0.9, max_itr=10, batch_size=20, rtype='chakra'):
    if rtype == 'chakra':
        env = gym.make('chakra-v0')
    else:
        env = gym.make('vishamC-v0')
    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]

    # initialization
    theta = np.random.normal(scale=0.01, size=(obs_dim + 1, action_dim))
    # theta [3x2]

    rewards = []
    for i in range(max_itr):
        # collect batch_size no. of trajectories
        trajectories = []
        total_r = 0.
        for _ in range(batch_size):
            state = env.reset()
            tr = []
            done = False

            while not done:
                # select action
                action = np.random.multivariate_normal(mean=np.dot(theta.T, np.append(state,1.)), cov=np.identity(action_dim))
                s, r, done, _ = env.step(action)
                total_r += r
                tr.append((state, action, r))
                state = s
            trajectories.append(tr)
        rewards.append(total_r/batch_size)

        b = 0.
        t = 0
        grad = np.zeros(shape=theta.shape)
        for trajectory in trajectories:
            for j in range(len(trajectory)):
                t += 1
                s, a, r = trajectory[j]
                R = sum([gamma**p*val[2] for p, val in enumerate(trajectory[j:])])
                s = np.append(s, 1).reshape((s.shape[0]+1, 1))
                mu = np.dot(theta.T, np.append(state,1.)).reshape((1, a.shape[0]))
                grad += np.dot(s, (a - mu))*(2*(R-b))

        # normalize grad
        grad = grad / (np.linalg.norm(grad)+1e-8)

        # update parameter
        theta += lr * grad

    if rtype == 'chakra':
        np.save('theta_c', theta)
    else:
        np.save('theta_v', theta)
    return rewards

def chakra_get_action(theta, ob, rng=np.random):
    # ob_1 = include_bias(ob)
    ob_1 = np.array(ob.tolist()+[1.])
    mean = theta.dot(ob_1)
    return rng.normal(loc=mean, scale=1.)    

@click.command()
@click.argument("env_id", type=str, default="chakra")
def main(env_id):
    # Register the environment
    rng = np.random.RandomState(42)

    if env_id == 'chakra':
        # from rlpa2 import chakra
        # env = gym.make('chakra')
        from chakra import chakra
        env = chakra()
        get_action = chakra_get_action
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
    else:
        raise ValueError(
            "Unsupported environment: must be 'chakra' ")

    env.seed(42)

    # Initialize parameters
    theta = rng.normal(scale=0.01, size=(action_dim, obs_dim + 1))

    while True:
        ob = env.reset()
        done = False
        # Only render the first trajectory
        # Collect a new trajectory
        rewards = []
        while not done:
            action = get_action(theta, ob, rng=rng)
            next_ob, rew, done, _ = env.step(action)
            ob = next_ob
            env.render()
            rewards.append(rew)

        print("Episode reward: %.2f" % np.sum(rewards))


def plot():
    max_episode = 100
    rewards = []
    bs = [0.001, 0.01, 0.1, 1., 10.]
    # bs = [1, 2, 5, 10, 20, 50]
    for batch_size in bs:
        tr = 0
        for i in tqdm(range(10)):
            r= policy_gradient(lr=batch_size, batch_size=5, max_itr=max_episode, rtype='visham')
            tr += r[-1]
        rewards.append(tr/10.)
        print(tr/10.)
        # break

    r_y = rewards
    plt.title('policy gradient')
    plt.xlabel('batch_size')
    plt.ylabel('average rewards')
    plt.plot(bs,r_y)
    plt.savefig('pg_lr_v.png')
    plt.clf()


def plot2(model='chakra'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = []
    Y = []
    Z = []

    if model == 'chakra':
        theta = np.load('theta_c.npy')
        env = gym.make('chakra-v0')
    else:
        theta = np.load('theta_v.npy')
        env = gym.make('vishamC-v0')

    action_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    max_itr = 500
    for i in range(max_itr):
        # collect batch_size no. of trajectories
        trajectories = []
        state = env.reset()
        done = False

        while not done:
            # select action
            action = np.random.multivariate_normal(mean=np.dot(theta.T, np.append(state,1.)), cov=np.identity(action_dim))
            s, r, done, _ = env.step(action)
            state = s
            trajectories.append((state, r))
        
        tr = 0.
        for s, r in reversed(trajectories):
            tr += r
            X.append(s[0])
            Y.append(s[1])
            Z.append(tr)

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    

if __name__ == "__main__":
    # plot2('chakra')
    plot()
    # main()
    # policy_gradient(max_itr=1000, batch_size=5)
    # policy_gradient(max_itr=100, batch_size=10, rtype='visham')

    # env = gym.make('chakra-v0')
    # action_dim = env.action_space.shape[0]
    # obs_dim = env.observation_space.shape[0]
    # theta = np.load('chakra.npy')
    # end = 0
    # while not end:
    #   state = env.reset()
    #   env.render()
    #   done = False
    #   rewards = []
    #   while not done:
    #       action = np.random.multivariate_normal(mean=np.dot(theta.T, np.append(state,1.)), cov=np.identity(action_dim))
    #       s, r, done, _ = env.step(action)
    #       rewards.append(r)
    #       env.render()
    #       time.sleep(0.1) 
    #   end = int(input())
    #   print('rewards: %d' %(np.sum(rewards)))
    # env.close()