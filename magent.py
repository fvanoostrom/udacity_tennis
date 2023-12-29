import numpy as np
import random
import json
import copy
from collections import namedtuple, deque

from ddpg_agent import DDPGAgent

import torch

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, seed, agent_configuration):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.parse_agent_configuration(agent_configuration)
        self.num_agents = num_agents
        self.agents = [DDPGAgent(state_size=state_size, action_size=action_size, num_agents = num_agents, seed=seed, agent_configuration= agent_configuration)
 for idx in range(num_agents)]
        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

    def parse_agent_configuration(self, agent_configuration):
        self.agent_configuration = agent_configuration
        self.buffer_size = agent_configuration["buffer_size"] if agent_configuration else None
        self.batch_size = agent_configuration["batch_size"] if agent_configuration else None
        self.gamma = agent_configuration["gamma"] if agent_configuration else None
        self.tau = agent_configuration["tau"] if agent_configuration else None
        self.lr_actor = agent_configuration["lr_actor"] if agent_configuration else None
        self.lr_critic = agent_configuration["lr_critic"] if agent_configuration else None
        self.update_every = agent_configuration["update_every"] if agent_configuration else None

    def step(self, states, actions, rewards, next_states, dones, step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # save the experiences and rewards
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size and (step % self.update_every) == 0 :
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, states, add_noise=True, eps = 1.0):
        """Returns actions for given state as per current policy."""
        actions = []
        for state, agent in zip(states, self.agents):
            action = agent.act(state, add_noise, eps)
            actions.append(action)
        return actions

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        for agent in self.agents:
            agent.learn(experiences, gamma)

    def save(self, path='output/checkpoint.pt'):
        for idx, agent in enumerate(self.agents):
            agent.save(path.replace('.pt',f'_{idx}.pt'))
            
    def load(self, path='output/checkpoint.pt'):
        for idx, agent in enumerate(self.agents):
            agent.load(path.replace('.pt',f'_{idx}.pt'))


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, seed, mu=0., theta=0.15, sigma=0.08):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(shape)
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
        dx = self.theta * (self.mu - x) + self.sigma * (np.random.rand(*x.shape)-0.5)
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
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
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