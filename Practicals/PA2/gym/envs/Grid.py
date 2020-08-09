import gym
import numpy as np
import logging

logger = logging.getLogger(__name__)
class GridWorld(gym.Env):
	
	def __init__(self,m,n,starting,terminals,holes):
		self.m = m
		self.n = n
		self.grid = [['-']*self.n]*self.m
		self.rewardArray = np.zeros(m,n)
		self.terminals = terminals
		self.currentPosition = starting[np.random.randint(low = 0,high = len(starting)-1,size=1)]
		x,y = self.getCoordinates(self.currentPosition)
		grid[x][y] = 'S'

		for var1 in self.terminals: #terminals is array of terminal states
			x,y = self.getCoordinates(var1)
			grid[x][y] = 'T'
			rewardArray[x][y] = 10
		self.actionSpace = {'U' :-self.n,'D':self.n,'R': 1,'D': -1}
		self.totalActions = ['U','D','R','L']

		for var1 in self.holes :
			x,y = self.getCoordinates(var1[0])
			if var1[1] == -3:
				self.grid[x][y] = "BH"
				self.rewardArray[x][y] = -3
			elif var1[1] == -2 :
				self.grid[x][y] = "GH"
				self.rewardArray[x][y] = -2
			else:
				self.grid[x][y] = "H"
				self.rewardArray[x][y] = -1
	def offgrid(self,state):
		x.y = self.getCoordinates(state)
		if x >= self.m or y>=self.n or y<0 or x<0:
			return True
		else:
			return  False
	def getCoordinates(self,state):
		x = state // self.n
		y = state % self.n
		return x,y

	def setState(self,state):
		self.currentPosition = state
		x,y = self.getCoordinates(state)
		self.grid[x][y] = 'O'

	def isTerminal(self,state):
		if state in self.terminals:
			return True
		else:
			return False

	def step(self,action):	
		newstate = self.currentPosition + actionSpace[action]
		if offgrid(newstate):
			reward = -5
			newstate = self.currentPosition
		else:
			x,y = self.getCoordinates(newstate)
			reward = rewardArray[x][y]

		self.setState(newstate)
		if np.random.uniform(size =1) > 0.5:
			newstate = self.currentPosition + 1
			if offgrid(newstate):
				reward = -5
				newstate = self.currentPosition
			else:
				x,y = self.getCoordinates(newstate)
				reward = rewardArray[x][y]
 
		self.setState(newstate)
		return self.currentPosition,reward,isTerminal(self.currentPosition),None

	def reset(self):
		self.currentPosition = starting[np.random.randint(low = 0,high = len(starting)-1,size=1)]
		self.grid = [['-']*self.n]*self.m
		x,y = self.getCoordinates(self.currentPosition)
		grid[x][y] = 'S'
		for var1 in self.terminals: #terminals is array of terminal states
			x,y = self.getCoordinates(var1)
			grid[x][y] = 'T'
        
		for var1 in self.holes:
			x,y = self.getCoordinates(var1[0])
			if var1[1] == -3:
				self.grid[x][y] ="BH"
         
			elif var1[1] == -2:
				self.grid[x][y] ="GH"
            	
			else:
				self.grid[x][y] ="H"

		return self.currentPosition

	def render(self):
		print("-------------------------------------------------------------")
		for var1 in range(self.m):
			for var2 in range(self.n):
				print(grid[var1][var2] + "  ")

			print('\n')
        
		print("\n")
		print("-------------------------------------------------------------")