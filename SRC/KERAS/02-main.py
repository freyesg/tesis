# import the gym stuff
import gym
# import other stuff
import random
import numpy as np
# import own classes
from simulated_annealing import SA


"""
import gym
env = gym.make('Copy-v0')
print env.reset()
print "bla bla bla bla bla bla"
print env.render()
print "bla bla bla bla bla bla"
"""

env = gym.make('CartPole-v0')

epochs = 10#10000
steps = 200
scoreTarget = 200
starting_temp = 1
final_temp = 0.001

entrada, salida = len(env.observation_space.high), env.action_space.n
sa = SA(entrada, salida, 10, env, steps, epochs, scoreTarget = scoreTarget, starting_temp = starting_temp, final_temp = final_temp, max_change= 1.0)

# network size for the agents
# [50 20 1]
#sa.initAgent([4])
sa.initAgent([50, 20, 1])

sa.sa()
#print type(env)
#print env.observation_space
#print env.action_space
