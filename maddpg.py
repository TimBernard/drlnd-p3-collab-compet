import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from ddpg_agent import DDPGAgent

import torch
import torch.nn.functional as F
import torch.optim as optim


# Current Hyperparemeters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 2e-4         # learning rate of the actor 
LR_CRITIC = 2e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG(): 
    """Coordinates Multi Agent Deep Deterministic Policy Gradient Algorithm""" 

    def __init__(self, state_size, action_size, random_seed): 
        """ Initialize multiple agents, replay buffer... """
        self.agents = [DDPGAgent(state_size, action_size, random_seed), 
                      DDPGAgent(state_size, action_size, random_seed)]

        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def act(self, states): 

        actions = [agent.act(state) for agent, state in zip(self.agents,states)]
        return np.array(actions)

    def reset(self): 
        for agent in self.agents: 
            agent.noise.reset()

    def step(self, states, actions, rewards, next_states, dones):

        # Add experience tuple to replay buffer 
        self.memory.add(states, actions, rewards, next_states, dones)

        # store (S_all, A_all, R_all, S_all') tuple in Replay buffer for (combined for both agents)

        # for each agent i = 1,..N:
            # sample random minibatch of (S_all, A_all, R_all, S_all') tuple from replay buffer of size batch-size 
            # set target using reward_i, S_all, A_all, S_all'
            # update critic, which requires using aforementioned target, S_all, and A_all
            # update actor, which requires using S_all, A_all 

        if len(self.memory) > BATCH_SIZE: 
            for idx, agent in enumerate(self.agents): 
                agent.learn(experiences=self.memory.sample(), gamma=GAMMA, agent_idx=idx)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""

        # Flatten these variables to be in order 
        state = state.flatten()
        action = action.flatten()
        next_state = next_state.flatten()

        # adapt replay buffer to have (S_all, S'_all, A_all, R_all) 
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
