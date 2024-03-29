import json
import os

from rl_environment import RLEnvironment
from magent import MAgent

with open("results/checkpoint_0_config.json", 'r') as f:
    configuration = json.load(f)


env = RLEnvironment(file_name='Tennis_Windows_x86_64/Tennis.exe')
num_agents = 2

agent = MAgent(state_size=24, action_size=2, num_agents = num_agents, seed=2, agent_configuration = configuration)
agent.load('results/checkpoint.pt')
env.run_episode(agent, train_mode = False, eps = 0.6, exit_on_done = False)
env.close()
