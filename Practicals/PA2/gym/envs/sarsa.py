import gym
import Grid
import numpy as np
import matplotlib.pyplot as plt

def stochAction(action):
	np.random.seed(seed =None)
	temp =[]
	for i in env.totalActions:
		temp.append(i)

	if np.random.uniform(size=1) < 0.9 :
		return action
	else:
		np.random.seed(seed =None)
		temp.remove(action)
		return temp[int(np.random.randint(low = 0,high = len(temp)-1,size = 1))]

def maxAction(Q,state):
	value = []
	for a in env.totalActions:
		value.append(Q[state,a])
	ind = np.argmax(value)
	return env.totalActions[ind]

def chooseAction(Q,state,eps):
	np.random.seed(seed =None)
	if np.random.uniform(size=1) < (1 - eps) :
		return maxAction(Q,state)
	else:
		np.random.seed(seed =None)
		return env.totalActions[int(np.random.randint(low = 0,high = len(env.totalActions)-1,size =1))]


if __name__ == '__main__':
	env = gym.make('Grid-v0')	
	num = 1500
	runs = 1
	runReward = np.zeros(num)
	runCount = np.zeros(num)

	for j in range(runs):
		alpha = 0.1
		gamma = 0.9
		eps = 0.7
		Q = {}
		for state in range(env.m*env.n):
			for action in env.totalActions:
				Q[state,action] = 0

		totalRewards = np.zeros(num)
		totalCount = np.zeros(num)

		for i in range(num):
			if i % 100 == 0:
				print("game finishes ",i)
			done = False
			epReward = 0
			count =0
			obs = env.reset()
			policy = []
			action = chooseAction(Q,env.currentPosition,eps)
			action = stochAction(action)

			while not done:
				obs1,reward,done,info = env.step(action)
				k = [obs,action]
				policy.append(k)
				if info == True:
					k = [obs1-1,'R']
					policy.append(k)
				epReward = epReward + reward
				count  = count +1
				action1 = chooseAction(Q,env.currentPosition,eps)
				action1 = stochAction(action1)

				Q[obs,action] = Q[obs,action] + alpha*(reward + gamma*Q[obs1,action1] - Q[obs,action])
				obs = obs1
				action = action1

			if eps - 2/num>0:
				eps = eps - 2/num
			else:
				eps =0
			totalRewards[i] = epReward
			totalCount[i] = count

		runReward = np.add(runReward,totalRewards)
		runCount = np.add(runCount,totalCount)

	runReward = np.divide(runReward,runs)
	runCount = np.divide(runCount,runs)

	plt.figure(1)
	plt.plot(runReward)
	plt.title("SARSA avg reward Vs runs Goal B")
	plt.xlabel("runs")
	plt.ylabel("avg reward")

	plt.figure(2)
	plt.plot(runCount)
	plt.title("SARSA avg steps Vs runs goal B")
	plt.xlabel("runs")
	plt.ylabel("avg steps")
	env.printOptPolicy()
	plt.show()
	env.printOptPolicy()
	env.render(policy)
