import numpy as np
import maze
from random import randint
import time

env = maze.Maze()
env.reset()
env.render()

a_v = np.zeros((5,5,4))

def t_pi(state,epsilon = 0.1):
    
   return np.argmax( a_v[tuple(state)])
     

def e_pi(state):
    return np.random.randint(4)

def q_learning(a_v,e_pi,t_pi,episodes,alpha = 0.1,gamma = 0.99):

    for episode in range(20):
        state = env.reset()
        print(f"Episode{episode + 1}/{episodes}")               #show episode number
        done = False
        step_count = 0

        while not done:
            action = e_pi(state)
            next_state,reward,done,_ = env.step(action)
            next_action = t_pi(next_state)

            qsa = a_v[tuple(state)][action]
            next_qsa = a_v[tuple(next_state)][next_action]
            a_v[tuple(state)][action] = qsa + alpha * (reward + gamma *next_qsa - qsa)


            env.render()
            time.sleep(0.01)


            state = next_state
            action = next_action
            step_count +=1

        print(f"Episode{episode + 1} finished after {step_count} steps.\n")    


#run q_learning method
q_learning(a_v,e_pi,t_pi,episodes = 20)

print("Learned Q-values:")
print(a_v)
