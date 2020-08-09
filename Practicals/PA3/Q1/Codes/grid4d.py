import gym
from gym import  spaces
from gym import error
from gym import utils
import numpy as np
from matplotlib import pyplot as plt
from gym.utils import seeding
import seaborn as sns

class grid4d(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):

        #making gridworld
        k = [11,11]
        self.grid = np.zeros(k)

        # boundary reward
        self.grid[5,2:5] = self.grid[5,2:5]- 0.2
        self.grid[5,0] = self.grid[5,0]- 0.2 
        self.grid[0:2,5] = self.grid[0:2,5] - 0.2
        self.grid[3:9,5] = self.grid[3:9,5]- 0.2
        self.grid[6,6:8] = self.grid[6,6:8]- 0.2
        self.grid[6,9:11] = self.grid[6,9:11]- 0.2
        self.grid[10,5] = self.grid[10,5]- 0.2
        self.grid[5,5] = -0.2
        #ax = sns.heatmap(self.grid, linewidth=0.5)
        #plt.show()
        #primitive actions

        self.actions = {
                        0 : [-1,0], # UP
                        1 : [0,1],  # Right
                        2 : [0,-1], # Left
                        3 : [1,0]   # Down
                    } 

        #setting up goal position
        self.goalPos = [[6,8],[8,8]]

        
        #goal position reward

        self.grid[6,8] = 1
        #self.grid[8,8] = 1

        # rooms start positions and its dimentions
        self.rooms = {
                        1 : [[0,0], [5,5]],
                        2 : [[0,6], [6,5]],
                        3 : [[7,6], [4,5]],
                        4 : [[6,0], [5,5]]

                    }
        
        # doorways in respective rooms
        self.doorways = {
                            1 : [[5,1],[2,5]],
                            2 : [[2,5],[6,8]],
                            3 : [[6,8],[9,5]],
                            4 : [[9,5],[5,1]]
                        }

   
    #state to room conversion
    def getRoom(self, state):

        room1,room2,room3,room4 = self.rooms[1],self.rooms[2],self.rooms[3],self.rooms[4]

        if ( (state[0]>=room1[0][0]) and (state[1]>=room1[0][1]) and (state[0]<=room1[0][0] + room1[1][0])  and (state[1]<=room1[0][1] + room1[1][1]) ):
            return 1
        if ( (state[0]>=room2[0][0]) and (state[1]>=room2[0][1]) and (state[0]<=room2[0][0] + room2[1][0])  and (state[1]<=room2[0][1] + room2[1][1]) ):
            return 2
        if ( (state[0]>=room3[0][0]) and (state[1]>=room3[0][1]) and (state[0]<=room3[0][0] + room3[1][0])  and (state[1]<=room3[0][1] + room3[1][1]) ):
            return 3
        if ( (state[0]>=room4[0][0]) and (state[1]>=room4[0][1]) and (state[0]<=room4[0][0] + room4[1][0])  and (state[1]<=room4[0][1] + room4[1][1]) ):
            return 4
        else:
            return 0   # for dorways
            
    #returns start position

    def gotoStart(self):
        temp = 1
        if temp == 1 :
            firstPos,dim = self.rooms[temp]
            k = range(firstPos[0], firstPos[0] + dim[0])
            k = list(k)
            k1 = range(firstPos[1], firstPos[1] + dim[1])
            k1 = list(k1)
            startPos = [np.random.choice(k), np.random.choice(k1)]
            return startPos
        else:
            startPos = [8,2]
            return startPos
    
    
    def getDoor(self, state):   
        return self.doorways[self.getRoom(state)]


    def getReward(self, pos):
        return self.grid[pos[0],pos[1]]

    
    def isAtDoor(self, state):
        d1,d2 = self.doorways[1]
        d3,d4 = self.doorways[3]

        if state == d1 or state == d2 or state == d3 or state == d4:
            return  True
        
        else:
            return  False

    def step(self, state, action, tardoor):
        room = self.getRoom(state)
        flag = self.isAtDoor(state)
        nextState = state

        if flag: # if state is at doorway
            x,y = state
            xnext,ynext = self.actions[action][0] , self.actions[action][1]
            nextState = state if self.getReward([x+xnext,y+ynext]) == -0.2 else [x+xnext,y+ynext]
            return nextState, self.getReward([x+xnext,y+ynext]), False,False

        else:
            x1,y1 = self.rooms[room][0][0],self.rooms[room][0][1]
            x2,y2 = self.rooms[room][0][0]+self.rooms[room][1][0] - 1 , self.rooms[room][0][1]+self.rooms[room][1][1] - 1
            d1, d2 = self.getDoor(state)
            t1,t2 = state[0] + self.actions[action][0] , state[1] + self.actions[action][1]
            nextState = [t1,t2]

            if  [t1,t2] == d2 or [t1,t2] == d1 :
                terminate = True if nextState == tardoor else False
                if  self.goalPos[0] == nextState or self.goalPos[1] == nextState:
                    done = True
                else:
                    done = False
                return nextState , self.getReward(nextState) , done , terminate

            elif (t1 < x1) or (t2 < y1) or (t1> x2)  or (t2 > y2) :
                return state,False,False,-0.2

            else : 
                if self.goalPos[0] == nextState or self.goalPos[1] == nextState:
                    done = True
                else:
                    done = False 

                terminate = False
                return nextState, self.getReward(nextState), done, terminate


    def reset(self):
        return self.gotoStart()
        

    def render(self, mode='human'):
        print("nothing")

    def close(self):
        print("closing")


'''
if __name__=='__main__':
    obj = grid4d()
    obj.gotoStart()
'''