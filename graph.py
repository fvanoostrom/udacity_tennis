# from unityagents import UnityEnvironment
# from datetime import datetime
import json
import os
import matplotlib.pyplot as plt

# from rl_environment import RLEnvironment
# from base_agent import BaseAgent
# from dqn_agent import DQNAgent
# from double_dqn_agent import DoubleDQNAgent
# from ddpg_agent import DDPGAgent
# from agent import Agent

# import numpy as np


# open all previous results
if os.path.isfile("results/results.json"):
    with open("results/results.json", 'r') as f:
        results = json.load(f)
# if it does not exist create an empty array
else:
    results = []

# plot the results and compare them with other results
def calculate_moving_average(numbers, window_size):
    return [round(sum(numbers[max(0,i-window_size):i]) / 
                  min(i,window_size),
                  2)
            for i in range(1, len(numbers)+1)]

# plt.plot(calculate_moving_average(scores,100), color='red', label='current result')

#filter the results on the 5 best final score (highest average of last 100 episodes) 
results_displayed = sorted(results, key=lambda r: r['date'], reverse=True)[:8]
cmap = plt.cm.get_cmap('hsv', len(results_displayed)+2)
#plot these results by taking all the scores and calculating the moving average.
plt.figure(figsize=(12,10))
for i, result in enumerate(results_displayed):
    label = result['name']
    plt.plot(calculate_moving_average(result['scores'],100), color=cmap(i+1), alpha=1.0, linestyle='--', label=label)
plt.legend()
plt.ylabel("Score")
plt.xlabel("Episode #")
# plt.show()
plt.savefig("results/" + "results" + ".png")
