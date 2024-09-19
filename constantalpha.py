import numpy as np
import maze
import matplotlib.pyplot as plt
from random import randint
env = maze.Maze()
env.reset()
env.render()

print(f"Observation space shape: {env.observation_space.nvec}")
print(f"Number of actions: {env.action_space.n}")
Q = {}


action_values = np.zeros(shape=(5,5,4))

for i in range(5):
    for j in range(5):
        state = (i,j)
        for action in range(4):
            Q[state,action] = np.random.rand(0,1)



qsa = [0.3,2,1,4]

def policy(state,epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(qsa)
        

        
gamma = 0.99
epsilon = 0.1
num_episodes = 20
alpha = 0.1 

def on_policy_mc_control(policy,action_values,episodes,gamma,epsilon,alpha):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        transitions = []

        while not done:
            action = policy(state,epsilon)
            next_state,reward,done = env.step(action)
            transitions.append([state,action,reward])
            state = next_state

            G = 0
            for state_t,action_t,reward_t in reversed(transitions):
                G = reward_t + gamma*G

                qsa = action_values[state_t][action_t]
                action_values[state_t][action_t] += alpha*(G-qsa)



