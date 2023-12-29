import json
import os

from rl_environment import RLEnvironment
from base_agent import BaseAgent
from ddpg_agent import DDPGAgent
agents ={"base": BaseAgent, "random": BaseAgent, "ddpg": DDPGAgent}

with open("results/config.json", 'r') as f:
    configuration = json.load(f)


env = RLEnvironment(file_name='Tennis_Windows_x86_64/Tennis.exe')
num_agents = 2

agent = DDPGAgent(state_size=24, action_size=2, num_agents = num_agents, seed=2, agent_configuration = configuration["agent"])
agent.load('results/checkpoint.pt')
env.run_episode(agent, train_mode = False, eps = 0.6, exit_on_done = False)
env.close()
