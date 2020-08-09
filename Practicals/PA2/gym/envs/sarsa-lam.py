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
	num = 2000
	runs = 1
	
	lam = [0.9] #[0,0.3,0.5,0.9,0.99,1]
	xaxis = np.arange(1,num+1)
	xaxis = xaxis.transpose()
	trackReward =[]
	trackStep =[]
	photo =0
	for k in range(len(lam)):
		runReward = np.zeros(num)
		runCount = np.zeros(num)
		for j in range(runs):
			alpha = 0.1
			gamma = 0.9
			eps = 0.7
			lambada = lam[k]
			Q = {}
			for state in range(env.m*env.n):
				for action in env.totalActions:
					Q[state,action] = 0
			E = {}
			for state in range(env.m*env.n):
				for action in env.totalActions:
					E[state,action] = 0
			totalRewards = np.zeros(num)
			totalCount = np.zeros(num)

			for i in range(num):
				if i % 100 == 0:
					print("game finishes ",i)
				
				done = False
				epReward = 0
				count =0
				obs = env.reset()
				action = chooseAction(Q,env.currentPosition,eps)
				action = stochAction(action)
				policy = []
				while not done:
					obs1,reward,done,info = env.step(action)
					t = [obs,action]
					policy.append(t)
					if info == True:
						t = [obs1-1,'R']
						policy.append(t)
					epReward = epReward + reward
					count  = count +1
					action1 = chooseAction(Q,env.currentPosition,eps)
					action1 = stochAction(action1)

					delta = reward +gamma*Q[obs1,action1] - Q[obs,action]
					E[obs,action] = E[obs,action] + 1
					for state in range(12*12):
						for a in env.totalActions:
							Q[state,a] = Q[state,a] + alpha*delta*E[state,a]
							E[state,a] = gamma*lambada*E[state,a]

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
		trackStep.append(runCount)
		trackReward.append(runReward)
	env.printOptPolicy()
	plt.show()
'''
		plt.figure()
		plt.plot(xaxis,runReward,'r')
		plt.title("SARSA lam avg reward Vs runs")
		plt.xlabel("runs")
		plt.ylabel("avg reward")
		photo = photo + 1
		string1 = "SARSAlam" +str(lam[k])+"reC"
		plt.savefig(string1+".png")

		plt.figure()
		plt.plot(xaxis,runCount,'r')
		plt.title("SARSA lam steps Vs runs")
		plt.xlabel("runs")
		plt.ylabel("avg steps")
		photo = photo + 1
		string1 = "SARSA" +str(lam[k])+"stepC"
		plt.savefig(string1+".png")
		env.printOptPolicy(policy)
		photo = photo + 1
		string1 = "SARSAlam" +str(lam[k])+"goalCpolicy"
		plt.savefig(string1+".png")
		env.render(policy)

	plt.figure()
	i=0
	colors = ['b', 'r', 'g', 'm', 'y','k', 'c']
	for var1 in trackReward:
		temp =[]
		
		for var2 in range(num):
			temp.append(var1[var2])
		plt.plot(xaxis,temp,colors[i],label ="lam = "+str(lam[i]))
		i = i +1
	plt.title("SARSA lam avg reward Vs runs comparison")
	plt.xlabel("runs")
	plt.ylabel("avg reward")
	plt.legend(bbox_to_anchor=(0.8, 1.10), loc='upper left', borderaxespad=0.)

	photo = photo + 1
	string1 = "SARSAlamComparisonCre"
	plt.savefig(string1+".png")
	plt.figure()
	i=0
	for var1 in trackStep:
		temp =[]
		for var2 in range(num):
			temp.append(var1[var2])
		
		plt.plot(xaxis,temp,colors[i],label ="lam = "+str(lam[i]))
		i = i+1

	plt.title("SARSA lam avg step Vs runs comparison")
	plt.xlabel("runs")
	plt.ylabel("avg step")
	plt.legend(bbox_to_anchor=(0.8, 1.10), loc='upper left', borderaxespad=0.)
	photo = photo + 1
	string1 = "SARSAlamComparisonCstep"
	plt.savefig(string1+".png")
'''
