from unityagents import UnityEnvironment
import numpy as np
from collections import deque 
from maddpg import MADDPG 
from ddpg_agent import DDPGAgent
from model import Actor
import random
import torch
import numpy as np
import sys

def main(argv):

    unity_env_name = argv[0]

    # please do not modify the line below
    env = UnityEnvironment(file_name=unity_env_name)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.vector_observations              # get the current state
    score = 0.0                                          # initialize the score

    # load the weights from file
    maddpg = MADDPG(state_size=state_size, action_size=action_size, random_seed=0)


    
    agent_first = DDPGAgent(state_size, action_size, 10)
    agent_first.actor_local.load_state_dict(torch.load('first_agentcheckpoint_actor.pth', map_location='cpu'))

    agent_second = DDPGAgent(state_size, action_size, 10)
    agent_second.actor_local.load_state_dict(torch.load('second_agentcheckpoint_actor.pth', map_location='cpu'))

    maddpg.agents = [agent_first, agent_second]

    while True:
        actions = maddpg.act(state)
        env_info = env.step(actions)[brain_name]        # send the action to the environment
        next_states = env_info.vector_observations   # get the next state
        rewards = env_info.rewards                   # get the reward
        dones = env_info.local_done                  # see if episode has finished
        score += np.max(rewards)                                # update the score
        states = next_states                             # roll over the state to next time step
        #if np.any(dones):                                       # exit loop if episode finished
        #    break
    print("Score: {}".format(score))

if __name__ == "__main__": 
    print(sys.argv[1:])
    main(sys.argv[1:])

