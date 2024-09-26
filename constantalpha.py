import numpy as np
import maze

from random import randint
import time
env = maze.Maze()
env.reset()
env.render()

print(f"Observation space shape: {env.observation_space.nvec}")
print(f"Number of actions: {env.action_space.n}")
Q = {}
a_v = np.zeros(shape=(5,5,4))

for i in range(5):
    for j in range(5):
        state = (i,j)
        for action in range(4):
            qsa =  Q[tuple(state),action] = np.random.rand()





def policy(state,epsilon=0.1):
    if np.random.random() < epsilon:
        return np.random.randint(4)
    else:
        return action == np.argmax(qsa)
        

        
gamma = 0.99
epsilon = 0.1
num_episodes = 20
alpha = 0.1 

def on_policy_mc_control(policy,a_v,episodes,gamma=0.99,epsilon=0.1,alpha=0.1):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        transitions = []
        step_count = 0

        print(f"Episode{episode + 1}/{num_episodes}")

        while not done:
            action = policy(state,epsilon)
            next_state,reward,done,_ = env.step(action)
            transitions.append([tuple(state),action,reward])

            env.render()
            time.sleep(0.01)

            state = next_state
            step_count +=1

            G = 0
            for state_t,action_t,reward_t in reversed(transitions):
                G = reward_t + gamma*G

                qsa = a_v[tuple(state_t)][action_t]
                a_v[tuple(state_t)][action_t] += alpha*(G-qsa)

        print(f"Episode{episode + 1} finished after {step_count} steps.\n")


on_policy_mc_control(policy,a_v,episodes = 20)



print("Learned Q-values:")
print(a_v)
                       



