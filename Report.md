[//]: # (Image References)

[image1]: results/trained_agent.gif "Trained Agents"
[image2]: results/results.png "Results of multiple training runs"

Within this project we will implement a reinforcement learning agent. The goal of the agent is to reach a moving ball. The agent does so by adding force two a robotic arm, consisting of two joints.The goal of the agent is to find the right forces for the joints to reach the ball. For this purpose the agent receives each timestep 33 inputs from the environment. We will make use of the DDPG architecture. Within this project we will make use of the verion of the environment with 20 arms to speed up the gathering of experience.

![Trained agent][image1]

# Improvements
At first the agent was a copy from the reacher project. After some small adjustments this resulted in the following gameplay in which the agents seemed to not learn anything at all. It seemed to just avoid the ball alltogether. 

- To the eps was reintroduced
- lowering the discount factor from 0.95 to 0.99
- The model seemed to learn too slow. Therefore, the update frequency (after how many timesteps the model should be retrained) was set from once every 10 to every timestep. This however resulted in very long runtimes. Instead of updating the model more frequent, the learning rates where increased by a tenfold (from 1e4 to 1e3) and update frequency was again set to once every 10 timesteps. This resulted in lower running times.
- increasing the learning rate
- increasing the tau

Prioritized Experience Replay
In order to speed up the training a version of Prioritized Experience Replay was implemented. There exist multiple versions of implementing this. This version was inspired by the work of [the-computer-scientist](https://github.com/the-computer-scientist/OpenAIGym/blob/master/PrioritizedExperienceReplayInOpenAIGym.ipynb) ([youtube](https://www.youtube.com/watch?v=MqZmwQoOXw4&ab_channel=TheComputerScientist)). Within this implementation a default importance is given to each new experience. During training this importance is updated by calculting the TD error. This is the difference between expected value of the (local) critic and the received reward and the expected value of the next state (as calculated by the target actor and critic). When the importance is high the experience is more likely to be used later on during another training run.


- after some more debugging it was found that the weights of the actor did not change.

Further hyperparameter changes to reduce training time at cost of episodes


# Architecture
The architecture used is based on the architecture of the Pendulum exercise. This is a DDPG algorithm that uses deep learning to learn the right values. Within this architecture there is both an actor and a critic network. The first one for choosing the best action, and the second one for estimating the values of the action.

1. Initialize the environments
1. Get the states of all the arms
1. Use the local version of the actor to choose the right torque to apply to the joints
1. Save the states and rewards to the replay memory
1. Check if it is time to train. The agent should have gathered enough experiences in the replay memory. Furthermore it is set to only train each 'X' timestep
1. If it is time for training: 

    A. Train the local critic network

    B. Train the local actor network

    C. Soft update the target networks for both actor and critic by marginally getting these networks closer to the local (parameter tau, by default set to 1e-3)
1. Check whether the environment is done or maximum amount of timesteps has been reached. If not, increase the timestep and go back to step 2. Otherwise break the loop and go to step 8.
1. Save the scores and check whether to goal has been reached. Otherwise, run a new episode by going to step 1.

The solution is structured in this way:
- main.py: python scripts that runs the simulation
- rl_environment.py: a wrapper around the Unity environment that handles the looping over episodes and timesteps
- base_agent.py: a base class of an agent which performs random actions
- ddpg_agent.py: an implement version of the base agent with a ddpg architecture
- dnn_model.py: a class with the 'brain' of the agent: an pytorch neural network

Furthermore the solution consists of:
- report.md: this report
- README.md: instructions how to get everything started
- play.py: python script to run a trained agent
- plot.py: python script to plot a matplotlib graph
- output folder: place where the scores, configuration and matplotlit graphs are stored
- results folder: results of the project: images, trained model
- results/checkpoint_*.pt: the weights of the trained model of the actor and critic (both local and target)

# Architecture Neural Network
The agent makes use of a pytorch deep neural network with two hidden layers for all it's networks (local actor, target actor, local critic, local actor). By default we use [ReLU](https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning) activation function, and a [batchnorm layer](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739) to normalize the outputs and speed up learning.

# Trying out different configurations of hyperparameters
It is quite hard to create a well performing agent, since even agents that eventually converge towards a succesfull strategy have low scores in the beginning of the training. Since the training takes a few hours to come up with a succesfull model it takes a lot of trial and error. Therefor, the parameters from the DDPG-pendulum were used as a baseline. With the exception of the buffer size which was increased to 1e6 to make room for the increased number of agents.

In an attempt to decrease the number of required episodes the buffer size was increased to 5e6, and update frequency of training to once every 4 timesteps. However, this made the training too slow, and it had to be abandoned after 12 hours, in which it ran 500 episodes with scores reaching an average around 9. This is an improvement in terms of episodes, but the cost of computation time for each episode was far more increased. Making it less efficient altogether.

The following table gives an overview 

| Parameter     |                           Description                          | Pendulum exercise | Previous Project (banana solver) | Used Default Value for this project |
|---------------|:--------------------------------------------------------------:|------------------:|------------------|--------------------|
| buffer_size   |                       replay buffer size                       |               128 | 128              | **1024**           |
| batch_size    |                         mini batch size                        |               1e5 | 1e5              | **1e6**            |
| update_every  |                   timesteps between training                   |                 - | 4                | 10                 |
| gamma         | discount factor                                                | 0.99              | 0.99             | 0.99               |
| tau           | soft update of target parameters of both models                | 1e-3              | 1e-3             | 1e-3               |
| lr_actor      | learning rate of actor                                         | 1e-3              | 5e-4             | 1e-3               |
| lr_critic     | learning rate of critic                                        | 1e-4              |                  | 1e-4               |
| layer_size_1  | output size of the first layer                                 | 128               | 32               | 128                |
| layer_size_2  | output size of the second layer                                | 256               | 32               | 256                |
| activation    | what kind of activation is used after the output of the layers | ReLU              | ReLU             | ReLU               |
| normalization | whether a normalization is used within the layers              | batchnorm1d       | -                | batchnorm1d        |


## Other variations

![Results buffer and batch size][image2]

The default values solve the task of reaching an average score of 30.0 after 874 episodes and 3 hours and 11 minutes (pink line in the graph above). The training was later on repeated to see if it was consisted. After 864 episodes it resuled in an average score of 30.0, although it took a bit longer to train: 3 hours and 41 minutes (green line).

Other variants were tried out:
- removing the batch layer (purple): this drastically slowed down training. Reaching a score of only 2.58 after 1000 episodes
- reordering the network layers (dark blue): first the relu and then the batchnorm, as described [here](https://blog.paperspace.com/busting-the-myths-about-batch-normalization/). The benefit should be that the batchnorm has more control over the output. However, this version only reached a score of 15.03 after 1000 episodes
- reducing the amount of layers to 64 (light blue). This resulted in an average score of just 1.04 after 1000 episodes
- reducing the amount of layers to 128 (turquoise). This somehow resulted in an average score of just 0.04.
- reducing the batch_size to 256 while increasing the update frequency to every 2 timesteps: after 5 hours and 40 minutes an average score of 0.45 was reached
- increasing the update frequency to every 5 timesteps: this reduced the required number of episodes to reach a score of 30.0 to just 510 episodes. Although less experience was required, the training still took 3 hours and 17 minutes to complete. 


# Ideas for Future Work
![Trained agent][image1]
When looking at the trained agent it seems to perform quite optimal. An additional 6 hours and 11 minutes of training increased the average score of the agent to 38.57, but this seems to be about the limit of the current algorithm. The biggest improvement would be to speed up the learning. One thing that was not tried out was adding prioritization when running or storing the memory with all the states. Prioritized experience replay could improve the agent. 