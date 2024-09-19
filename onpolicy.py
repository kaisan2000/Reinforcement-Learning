import numpy as np
import maze
from random import randint
env = maze.Maze()
env.reset()
env.render()

Q = {}

gamma = 0.99
epsilon = 0.1
num_episodes = 200

for i in range(5):
    for j in range(5):
        state = (i,j)
        for action in range(4):
            Q[state , action] = np.random.rand(0,1)


Q[(4,4),action] = 0 

qsa = [0.3,0.1,0.2,0.4]

def pi(state):
    if np.random.rand()>epsilon:
        return np.argmax(qsa)
    else:
        return randint(1,4)
    

G = {}

for i_episode in range(num_episodes):
    done = False
    transitions = []

    while not done:
        action = pi(state)
        next_state,reward,done = env.step(action)   
        state = next_state

G = 0

for state_t,action_t,reward_t in reversed(transitions):
    G = reward_t + gamma*G

    if not (state_t,action_t) in G[state,action]:
        G[(state_t,action_t)] = []
    G[(state_t,action_t)].append(G)

    Q[state_t,action_t] = np.mean(G[(state_t,action_t)])     

print(Q[state_t,action_t])

        
