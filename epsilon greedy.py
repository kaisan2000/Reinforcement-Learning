#create a Q dictionary
import numpy as np
from random import randint
Q={}

for i in range(5):
    for j in range(5):
        state = (i,j)
        for action in range(4):
            Q = [state, action]


gamma = 0.99
epsilon = 0.1

qsa = [0.3,0.2,0.9,-2]

def pi(state):
    if np.random.rand()<epsilon:
         return np.argmax(qsa)
    else:
        return randint(1,4)


print(pi(state))     
     

         