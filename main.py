from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import numpy as np

from rl_environment import RLEnvironment
from base_agent import BaseAgent
from ddpg_agent import DDPGAgent

agents ={"base": BaseAgent, "random": BaseAgent, "ddpg": DDPGAgent}

configuration = {
                "min_episodes" : 100,
                "max_episodes" : 1000,
                "max_time" : 1000, 
                "target_score" : 30.0,
                "agent" : {
                        "type" : "ddpg",
                        "buffer_size" : int(1e6),  # replay buffer size
                        "batch_size" : 1024,         # minibatch size
                        "gamma" : 0.95,            # discount factor
                        "tau" : 1e-3,              # for soft update of target parameters
                        "lr_critic" : 1e-4,               # learning rate 
                        "lr_actor" : 1e-3,               # learning rate 
                        "update_every" : 10,        # how often to update the network
                        "actor" :{"layers": [{"type":"linear", "arguments": (33,128)},
                                             {"type":"batchnorm", "arguments":(128,)},
                                             {"type":"relu", "arguments":()},
                                             {"type":"linear", "arguments": (128,256)},
                                             {"type":"relu", "arguments":()},
                                             {"type":"linear", "arguments": (256,4)},
                                             {"type":"tanh", "arguments":()}]},
                        "critic" :{"layers": [{"type":"linear", "arguments": (33,128)},
                                             {"type":"batchnorm", "arguments":(128,)},
                                             {"type":"relu", "arguments":()},
                                             {"type":"linear", "arguments": (132,256)},
                                             {"type":"relu", "arguments":()},
                                             {"type":"linear", "arguments": (256,1)},
                                             {"type":"tanh", "arguments":()}]}
                                             }
}

env = RLEnvironment(file_name='Reacher_Windows_x86_64_20/Reacher.exe')
num_agents = 1

env.print_env_info()

start_date = datetime.now()
name = "linear_batchnorm_relu_128_256"
name = start_date.strftime("%Y%m%d_%H%M%S") + "_" + name
model_path = 'output/model_' + name + '.pt'

print(f"start:{start_date}")
agent = DDPGAgent(state_size=33, action_size=4, num_agents = num_agents, seed=2, agent_configuration = configuration["agent"])

# it is possible to load previously trained neural networks by uncommenting the following line:
# agent.load('results/checkpoint.pt')

# run the actual training
scores = env.train(agent, 100, configuration["max_episodes"], configuration["max_time"],
                configuration["target_score"],)

# calculate the duration of the training
end_date = datetime.now()
print(f"end:{end_date}")
duration = end_date - start_date
# create an object of the results and save the trained model
cur_result = {"name": name, "type":configuration["agent"]["type"],
                "date": start_date.strftime("%Y-%m-%d %H:%M:%S"), "episodes" : len(scores),
                "final_score" :  sum(scores[-100:])/len(scores[-100:]), "duration" : str(duration),
                "scores": scores, "configuration": configuration, "model_path" : model_path}

agent.save(path=model_path)

# save the result of the current run
with open("output/results_"+ name + ".json", 'w') as f:
    json.dump(cur_result, f, indent=2) 

# open all previous results
if os.path.isfile("output/results.json"):
    with open("output/results.json", 'r') as f:
        results = json.load(f)
# if it does not exist create an empty array
else:
    results = []

# save results with the current result appended
results.append(cur_result)
with open("output/results.json", 'w') as f:
    json.dump(results, f, indent=2)


# plot the results and compare them with other results
def calculate_moving_average(numbers, window_size):
    return [round(sum(numbers[max(0,i-window_size):i]) / 
                  min(i,window_size),
                  2)
            for i in range(1, len(numbers)+1)]

plt.plot(calculate_moving_average(scores,100), color='red', label='current result')

#filter the results on the 5 best final score (highest average of last 100 episodes) 
results_displayed = sorted(results, key=lambda r: r['date'], reverse=True)[:5]
cmap = plt.cm.get_cmap('hsv', len(results_displayed)+2)
#plot these results by taking all the scores and calculating the moving average.
for i, result in enumerate(results_displayed):
    label = result['type'] + '_' + result['date'] if result.get('alias') is None else result['alias']
    plt.plot(calculate_moving_average(result['scores'],100), color=cmap(i+1), alpha=1.0, linestyle='--', label=label)
plt.legend()
plt.ylabel("Score")
plt.xlabel("Episode #")
# plt.show()
plt.savefig("output/" + name + ".png")

# close the environment
env.close()
