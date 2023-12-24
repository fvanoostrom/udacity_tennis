from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from ddpg_agent import DDPGAgent

class RLEnvironment(UnityEnvironment):
    """Takes care of the Reinforcement Learning Environment"""

    def __init__(self, file_name):
        """Initialize an Agent object.
        
        Params
        ======
            file_name (string) : name of the executable file to run
        """
        super().__init__(file_name=file_name)
        self.brain_name = self.brain_names[0]
        self.brain = self.brains[self.brain_name]
        self.initiated = False
        self.num_agents = 0

    def reset(self, train_mode=True, config=None, lesson=None):
        self.env_info = super().reset(train_mode, config, lesson)
        self.brain_info = self.env_info[self.brain_name]
        self.initiated = True
        self.num_agents = len(self.brain_info.agents)
        self.action_size = self.brain.vector_action_space_size
        return self.env_info

    def print_env_info(self):
        if not self.initiated:
            self.reset()
        # number of agents in the environment
        print('Number of agents:', len(self.brain_info.agents))
        # number of actions
        print('Number of actions:', self.action_size)
        # examine the state space 
        state = self.brain_info.vector_observations[0]
        print('States look like:', state)
        state_size = len(state)
        print('States have length:', state_size)

    def run_episode(self, agent, train_mode= True, max_t=1000): 
        self.reset(train_mode=train_mode) 
        # Get the states of all agents
        states = self.brain_info.vector_observations
        # Set the score of all agents to zero
        score = np.zeros(self.num_agents)
        # Reset the agent
        agent.reset()
        # Loop over all timesteps until the max timestep is reached or the environment returns 'done'                                         
        for t in range(max_t):
            actions = agent.act(states)
            self.brain_info = self.step(actions)[self.brain_name]
            # get the next_states, rewards, if the task is done from the environment
            # and increase the score with the last reward
            next_states = self.brain_info.vector_observations
            rewards = self.brain_info.rewards
            dones = self.brain_info.local_done
            score += self.brain_info.rewards
            
            # Let the agent step. This could trigger a training
            agent.step(states, actions, rewards, next_states, dones, t)
            # set the current states to the states received by the environment
            states = next_states
            # if any agent is done, exit
            if np.any(dones):
                break
        return score

    def train(self, agent, min_episodes = 100, max_episodes=2000, max_t=1000, target_score= 1.0 ):
        scores = []
        # A queue to keep only the last 100 episodes' scores
        scores_window = deque(maxlen=100)
        for i_episode in range(1, max_episodes+1):

            score = self.run_episode(agent, train_mode=True, max_t=max_t)
            # calculate the mean of all running agents, and add them to the deque and scores
            score = score.mean()
            scores_window.append(score)
            scores.append(score)

            # compute mean of the last episodes of the window
            mean_score = np.mean(scores_window)
            
            # Print the mean of the last episode
            print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, mean_score), end="")

            if i_episode % 100 == 0:
                print('\rEpisode {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(i_episode, score, mean_score))
                
            if i_episode >= min_episodes and mean_score >= target_score :
                print('\rEnvironment solved in {} episodes, mean score: {:.2f}'.format(i_episode, mean_score))
                agent.save()
                break
                
        return scores