import datetime
import statistics
import numpy as np
import gym
from gym import wrappers
import tensorflow as tf
import os



# defined model that computes forward pass too
class MyModel(tf.keras.Model):
    def __init__(self,  totactions, totstates, hidunits):
        super(MyModel, self).__init__()

        self.inputLyr , self.hidLyrs = tf.keras.layers.InputLayer(input_shape=(totstates,)) , []

        for i in hidunits:
            temp = tf.keras.layers.Dense(i , activation = "relu" , kernel_initializer = 'RandomNormal')
            self.hidLyrs.append(temp)

        self.outputLyr = tf.keras.layers.Dense(totactions, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    #forward pass computation
    def call(self, inStates):
        temp = self.inputLyr(inStates)

        for fun in self.hidLyrs:
            temp = fun(temp)

        return self.outputLyr(temp)


class DQN:
    def __init__(self, totstates, totactions, hidunits, alpha):
        self.totactions, self.batchSize , self.gamma = totactions , 32, 0.95
        self.model = MyModel(totactions,totstates, hidunits )
        self.sarsa = {'state': [], 'action': [], 'reward': [], 'nextState': [], 'done': []}
        self.maxBufferSize , self.minBufferSize = 15000 , 100
        self.optimizer = tf.optimizers.Adam(alpha)

    def predict(self, inputs):    
        floatconv = inputs.astype("float32")   # directly computes forwad pass by just calling model.
        return self.model(np.atleast_2d(floatconv))

    def getAction(self, states, eps):
        ch = np.random.choice(self.totactions)
        act = self.predict(np.atleast_2d(states))
        act = act[0]
        act = np.argmax(act)

        if  np.random.random() < eps :
            return ch
        else:
            return act

    def copyW(self, Net):
        for i1, i2 in zip(self.model.trainable_variables, Net.model.trainable_variables):
            i1.assign(i2.numpy())


    def train(self, TargetNet):
        bufferSize  = len(self.sarsa['state'])
        if self.minBufferSize > bufferSize :
            return 0

        iden = np.random.randint(low=0, high = bufferSize, size=self.batchSize)
        possStates = [self.sarsa['state'][i] for i in iden]
        states = np.asarray(possStates)

        possActions = [self.sarsa['action'][i] for i in iden]
        actions = np.asarray(possActions)

        possReward = [self.sarsa['reward'][i] for i in iden]
        rewards = np.asarray(possReward)

        possNextstate = [self.sarsa['nextState'][i] for i in iden]
        nextStates = np.asarray(possNextstate)

        possDone = [self.sarsa['done'][i] for i in iden]
        dones = np.asarray(possDone)

        nextQ = np.max(TargetNet.predict(nextStates), axis=1)
        nextRe = np.where(dones, rewards, rewards+self.gamma*nextQ)
        #print( tf.one_hot(actions, self.totactions))
        #print(self.predict(states))
        #print("finished")
        predstates = self.predict(states)
        oneHot = tf.one_hot(actions, depth = self.totactions)
        with tf.GradientTape() as tape:
            predNextRe = tf.math.reduce_sum(predstates * oneHot, axis=1)
            loss = tf.math.reduce_mean(tf.square(nextRe - predNextRe))

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    

    def addSarsa(self, exp):
        
        if self.maxBufferSize <= len(self.sarsa['state']) : 
            # pop the first element 
            for key in self.sarsa.keys():
                self.sarsa[key].pop(0)
        else:      
            for key, value in exp.items():
                self.sarsa[key].append(value)

    


def play(env, TrainNet, TargetNet, eps):
    rewards,stps,done,losses,copytime = 0,0,False,[],25
    obs = env.reset()
    while not done:
        action = TrainNet.getAction(obs, eps)
        prevobs = obs
        obs, r, done, k = env.step(action)
        rewards  = rewards +  r
        if done:
            r = -10
            env.reset()

        TrainNet.addSarsa( {'state': prevobs, 'action': action, 'reward': r, 'nextState': obs, 'done': done} )
        TrainNet.train(TargetNet)
        #print(loss)
        stps = stps + 1
        if (stps % copytime == 0 ) :
            TargetNet.copyW(TrainNet)

    return rewards

def main():
    env = gym.make('CartPole-v0')
    totactions,totstates = env.action_space.n , env.observation_space.shape[0]
    hidunits,alpha = [160, 160,160] , 0.01
    summary_writer = tf.summary.create_file_writer('Q2/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    TrainNet, TargetNet= DQN(totstates, totactions, hidunits, alpha),DQN(totstates, totactions, hidunits, alpha)
    episodes = 50000
    totrewards,eps,mineps,decay = np.empty(episodes) ,  1 , 0.01 , 0.9997

    for i in range(episodes):
        if i % 50 == 0:
            print("episode:", i)

        if eps <= mineps:
            eps = mineps
        else:
            eps = eps*decay

        gamere = play(env, TrainNet, TargetNet, eps)
        totrewards[i] = gamere
        if i % 50 == 0:
            print("reward:", gamere)

        avgre = totrewards[max(0, i - 100):(i + 1)].mean()

        with summary_writer.as_default():
            tf.summary.scalar('avg reward(100)', avgre, step=i)
            tf.summary.scalar('episode reward', gamere, step=i)
        
    env.close()


if __name__ == '__main__':
    main()