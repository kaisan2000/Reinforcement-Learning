import numpy as np
import maze
from random import randint
import time

env = maze.Maze()
env.reset()
env.render()

a_v = np.zeros((5,5,4))



def pi(state,epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(a_v[tuple(state)])
    


def sarsa(a_v,pi,episodes,alpha=0.1,epsilon=0.1,gamma=0.99):

    for episode in range(20):
        state = env.reset()
        print(f"Episode{episode + 1}/{episodes}")     #show episode number
        action = pi(state,epsilon)
        done = False
        step_count = 0

        while not done:
            next_state,reward,done,_ = env.step(action)   #take a step in env
            next_action = pi(next_state,epsilon)

            qsa = a_v[tuple(state)][action]    #update q-value
            next_qsa = a_v[tuple(next_state)][next_action]
            a_v[tuple(state)][action] = qsa + alpha * (reward + gamma *next_qsa - qsa)

            # print(f"Step{step_count}: State{state},Action{action},Reward{reward},Next State{next_state}")

            env.render()
            time.sleep(0.01)
            
            state = next_state
            action = next_action
            step_count += 1

            

        print(f"Episode {episode + 1} finished after {step_count} steps.\n")    


#run sarsa

sarsa(a_v,pi,episodes=20)

print("Learned Q-values:")
print(a_v)

    
              