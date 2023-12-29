import numpy as np
import random
import json
import os
import copy
from collections import namedtuple, deque

from dnn_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from base_agent import BaseAgent

class DDPGAgent(BaseAgent):
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, seed, agent_configuration):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        super().__init__(state_size, action_size, seed, agent_configuration)
        self.parse_agent_configuration(agent_configuration)
        self.num_agents = num_agents


        # Actor Network (w/ Target Network)
        self.actor_local = QNetwork(state_size, action_size, seed, agent_configuration["actor"]["layers"]).to(device)
        self.actor_target = QNetwork(state_size, action_size, seed, agent_configuration["actor"]["layers"]).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = QNetwork(state_size, action_size, seed, agent_configuration["critic"]["layers"]).to(device)
        self.critic_target = QNetwork(state_size, action_size, seed, agent_configuration["critic"]["layers"]).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic, weight_decay=0)

        #Copy the weights from local to target networks. Value of 1.0 means it is a hard copy
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)
        
        # Noise process
        self.noise = OUNoise((self.num_agents, self.action_size), seed)

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

    def act(self, state, add_noise=True, eps = 1.0):
        """Returns actions for given state as per current policy."""

        # create a pytorch tensor from the received numpy array
        state = torch.from_numpy(state).float().to(device)

        # set the actor to evaluation and turn off gradients computation while computing the action
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
            
        # turn back on the training state
        self.actor_local.train()
        # add noise
        if add_noise:
            action += self.noise.sample()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.clip(action, -1, 1)
        else:
            #take a random action between -1 and 1
            action = np.random.random(size=action.shape) * 2 - 1
            # return np.random.rand(self.action_size) * 2 -1

            
        # # create a valid return value by keeping it between -1 and 1
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

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
                
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models       
        actions_next = self.actor_target(next_states)
        # Calculate the value of the next actions given the next_states using the target critic network.
        Q_targets_next = self.critic_target(next_states, actions_next).detach()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # compute the expected value from the local critic network
        Q_expected = self.critic_local(states, actions)
        # calculate the difference
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        # compute the best action using the local actor
        actions_pred = self.actor_local(states)
        # compute the loss 
        # the loss is set to - since we perform ascent in stead of descent.
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, path='output/checkpoint.pt'):
        index = path.find('.pt')
        torch.save(self.actor_local.state_dict(), path[:index]+'_actor_local'+path[index:])
        torch.save(self.critic_local.state_dict(), path[:index]+'_critic_local'+path[index:])
        torch.save(self.actor_target.state_dict(), path[:index]+'_actor_target'+path[index:])
        torch.save(self.critic_target.state_dict(), path[:index]+'_critic_target'+path[index:])
        with open(path.replace('.pt','_config.json'), 'w') as f:
            json.dump(self.agent_configuration, f, indent=2) 
            
    def load(self, path='output/checkpoint.pt'):
        index = path.find('.pt')
        self.actor_local.load_state_dict(torch.load(path[:index]+'_actor_local'+path[index:]))
        self.actor_target.load_state_dict(torch.load(path[:index]+'_actor_target'+path[index:]))
        self.critic_local.load_state_dict(torch.load(path[:index]+'_critic_local'+path[index:]))
        self.critic_target.load_state_dict(torch.load(path[:index]+'_critic_target'+path[index:]))


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