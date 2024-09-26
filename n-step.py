import numpy as np
import maze
from random import randint
import time

env = maze.Maze()
env.reset()
env.render()

a_v = np.zeros((5,5,4))

def pi(state,epsilon = 0.1):
    if np.random.random()<epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(a_v[tuple(state)])
    

def n_step(a_v,pi,episodes,alpha=0.1,gamma=0.99,epsilon=0.1,n=6):

    for episode in range(20):
        state = env.reset()
        action = pi(state,epsilon)
        done = False
        transitions = []
        print(f"Episide{episode + 1}/{episodes}")
        step_count = 0
        t = 0

        while t-n < len(transitions):

            if not done:
                next_state,reward,done,_ = env.step(action)
                next_action = pi(next_state,epsilon)
                transitions.append([state,action,reward])

            if t>=n:
                G = (1-done)*a_v[tuple(next_state)][next_action]

                for state_t,action_t,reward_t in reversed(transitions[t-n:]):
                    G = reward_t + gamma*G

                a_v[tuple(state_t)][action_t] += alpha * (G-a_v[tuple(state_t)][action_t])

            env.render()
            time.sleep(0.01)


            t = t+1
            state = next_state
            action = next_action
            step_count +=1

        print(f"Episode{episode + 1} finished after {step_count} steps.\n")


   
#run n-step method
n_step(a_v,pi,episodes = 20)
print("Learned Q-values")
print(a_v)

    