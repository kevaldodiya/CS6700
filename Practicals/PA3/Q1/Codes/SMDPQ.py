import gym
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class SMDPQ():

	def __init__(self, env, episodes, flag):

		self.env,self.Q,self.intra,self.episodes,self.gamma,self.alpha,self.epsilon = env , np.zeros([6, 11, 11]) , flag , episodes, 0.9, 0.1,0.1
    

	def getOpt(self, state):
		temp =  np.random.uniform(0,1)
		ch = np.random.choice([0,1,2,3,4,5])
		maxi = np.argmax(self.Q[:,state[0],state[1]], axis=0)
		if   self.epsilon >temp:
			return ch  # 4-5 are multistep option except options are primitive actions
		else: 
			return maxi


	def policy(self, state, tardoor):

		if self.env.isAtDoor(state):
			x1,y1 = state
			x2,y2 = tardoor 
            # if agent at a doorway then it should move into one direction
			x = 3 if x2>x1 else 0
			y = 1 if y2>y1 else 2

			tempAction = self.env.actions[x][0] + x1
			xtemp = self.env.getReward([tempAction, y1]) 
			action = y if xtemp < 0 else x

		else:
            # inside the room
			x1,y1 = state
			x2,y2 = tardoor
			x = 3 if x2>x1 else 0
			y = 1 if y2>y1 else 2
				
			if x1 == x2:
				action = y

			elif y1 == y2:
				action = x

			else:
				first = [state[0]+self.env.actions[x][0],y1]
				sec = [x1,state[1]+self.env.actions[y][1]]
				distx = abs(x2-x1)
				disty = abs(y2-y1)
				if (self.env.getReward(first)>=0) and (disty < distx) :
					action = x
				elif (disty > distx) and (self.env.getReward(sec)>=0) :
					action = y            
				else:
					pos = [state[0]+self.env.actions[x][0],y1]
					action = x if self.env.getReward(pos) >=0 else y

		return action

	def runOption(self, state,tarDoor,opt):
		
		if opt <= 3:
			st = 0
			pr = [0.1/3, 0.1/3, 0.1/3, 0.1/3]
			pr[opt] = 0.9 
            # select action by above mentioned probabity 

			action = np.random.choice([0,1,2,3],1,p = pr)[0]
			st  = st + 1
			nextstate, totreward, done, k = env.step(state, action, tarDoor)
			if self.intra:
				ne = totreward + self.gamma * np.amax(self.Q[:,nextstate[0],nextstate[1]]) - self.Q[action][state[0], state[1]]
				self.Q[action][state[0], state[1]] = self.Q[action][state[0], state[1]] + self.alpha*ne

                    
		else: 
            #performing multistep option
			st = 0
			pr = [0.03333, 0.03333, 0.03333, 0.03333]
			totreward = 0
			flag =1
			while state != tarDoor:
				nextstate, reward, done, terminate = self.env.step(state, self.policy(state, tarDoor), tarDoor)
				totreward , state , st  = totreward + reward, nextstate , st + 1
				if self.intra:
                    # update the Q value using the same option
					if  terminate:
						ne = reward + self.gamma * np.amax(self.Q[:,nextstate[0],nextstate[1]]) - self.Q[opt][state[0], state[1]]
						self.Q[opt][state[0], state[1]] = self.Q[opt][state[0], state[1]] + self.alpha*ne
 
                    # update using the max option possible in that state
					else:
						ne = reward + self.gamma * self.Q[opt,nextstate[0],nextstate[1]] - self.Q[opt][state[0], state[1]]
						self.Q[opt][state[0], state[1]] = self.Q[opt][state[0], state[1]] + self.alpha*ne						
    			
				if done:
					break
				
				if terminate:
					break

				

		return nextstate, totreward, done, st


	def smdpQ(self, env):
        
		rs,st = np.zeros([self.episodes]) , np.zeros([self.episodes])
		for episode in range(self.episodes):
			state = self.env.reset()

			while True:
                # here i choose 6 options 4 are primitive actions and 2 hallways options for each states
				temp = state
				option = self.getOpt(state)
				if option > 3:
					t = self.getTarDoor(state, option)
					tarDoor = t
					temp = tarDoor 
				else:
					tarDoor = None

				nextstate, reward, done, k = self.runOption(state,tarDoor,option)
                
				if not self.intra :
					ga = pow(self.gamma , k)
					ne_re = ga*np.amax(self.Q[:,nextstate[0],nextstate[1]])
					self.Q[option][state[0], state[1]] = (1 - self.alpha)*self.Q[option][state[0], state[1]] + self.alpha*( reward + ne_re )


				st[episode],state,rs[episode] = st[episode]+ 1 ,nextstate, rs[episode] + reward

				if done:
					break

		return  st, rs, self.Q


	def getTarDoor(self, state, option):
		if option > 3:

			d1,d2 = self.env.doorways[1]
			d3,d4 = self.env.doorways[3]
        
			if state == d1:
				if option == 4 :
					tardoor = self.env.doorways[4][0]
				else:
					tardoor = self.env.doorways[1][1]

			if state == d2 :
				if option == 4:
					tardoor = self.env.doorways[1][0]
				else:
					tardoor = self.env.doorways[2][1]	
			if state == d3 :
				if option == 4:
					tardoor = self.env.doorways[2][0]
				else:
					tardoor = self.env.doorways[3][1]

			if state == d4 :
				if option == 4:
					tardoor = self.env.doorways[3][0]
				else:
					tardoor = self.env.doorways[4][1] 

			if state != d1 and state != d2 and state != d3 and state != d4:
				tardoor = self.env.doorways[self.env.getRoom(state)][option-4]
				
		else:
			room = self.env.getRoom(state)
			d1,d2 = self.env.doorways[1]
			d3,d4 = self.env.doorways[3]

			if d1 == state:
				tardoor = [9,5]
			elif d2 == state or d3 == state:
				tardoor = [6,8]
			elif d4 == state:
				tardoor = [6,8]
			else:
				doors = self.env.doorways[room]
				if room == 1:
					if(  (abs(doors[0][0] - state[0]) + abs(doors[0][1] - state[1]) ) < (abs(doors[1][0] - state[0]) + abs(doors[1][1] - state[1])) ):
						tardoor = [5,1]
					else:
						tardoor = [2,5]

				elif room == 2 or room == 4:
					tardoor = [6,8]

				elif room == 3 :
					tardoor = [9,5]
					

		return tardoor
    
 


    

if __name__=='__main__':

	env = gym.make('grid4d-v0')
	episodes, runs , intra_opt = 1000, 50, False
	obj = SMDPQ(env, episodes, intra_opt)
	eps = [episodes]
	stp,avg,qtemp = np.zeros(eps) , np.zeros(eps) , np.zeros([11,11])
	act = 4

	for i in range(runs):
		print("run =", i)
		steps, rewards, Q = obj.smdpQ(env)	
		#qtemp = qtemp + Q[act,:,:]
		qtemp = qtemp + np.sum(Q, axis=0)
		avg = avg + rewards
		stp = stp + steps

	avg = (1/runs)*avg
	stp = (1/runs)*stp

	epi = range(episodes)
	plt.figure(1)
	plt.plot(epi, avg,color = 'c')
	plt.title('SMDP Q learning : Average reward with 50 experiments ')
	plt.ylabel('Avg_reward')
	plt.xlabel('episodes')

	plt.figure(2)
	plt.plot(epi, stp,color = 'g')
	plt.title('SMDP Q learning : Average steps with 50 experiments ' )
	plt.ylabel('steps')
	plt.xlabel('episodes')
	plt.show()
	ax = sns.heatmap(qtemp, linewidth=0.5)
	plt.show()