import numpy as np
import maze
import time

from random import randint
env = maze.Maze()
env.reset()
env.render()

Q = {}

gamma = 0.99
epsilon = 0.1
num_episodes = 20

for i in range(5):
    for j in range(5):
        state = (i,j)
        Q[state] = {}
        for action in range(4):
            Q[state][action] = -3* np.random.rand()






def pi(state,epsilon=0.1):
    if np.random.rand()>epsilon:
        action = np.argmax([Q[state][a] for a in range(4)])
        return action 
    else:
        return np.random.randint(4)      # return random action with epsilon probability
    



def on_policy(pi,Q,episodes=20,gamma=0.99,epsilon=0.1): 

    returns = {}
 

    for i_episode in range(episodes):
        state = env.reset()
        done = False
        transitions = []

        print(f"Episode{i_episode + 1}/{episodes}")

        #run one episode
    

        while not done:
            action = pi(state,epsilon)
            next_state,reward,done, info = env.step(action) 
            transitions.append([state,action,reward]) 

            env.render()
            time.sleep(0.01)

            state = next_state

        G = 0

        for state_t,action_t,reward_t in reversed(transitions):
            G = reward_t + gamma*G     #update return

            


            if not (state_t,action_t) in returns:
                returns[(state_t,action_t)] = []
            returns[(state_t,action_t)].append(G)

            Q[state_t,action_t] = np.mean(returns[(state_t,action_t)])   

    return Q

Q = on_policy(pi,Q,episodes = 20)

print("Learned Q-values:")

for state in Q:
    print(f"State {state} : {Q[state]}")
