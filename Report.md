[//]: # (Image References)

[image1]: results/trained_agent.gif "Trained Agents"
[image2]: results/best_results.png "Results of best training run"
[image3]: results/untrained_agent.gif "Untrained Agents"
[image4]: results/results.png "Results of multiple training runs"

Within this project we will implement a pair of reinforcement learning agents. The goal of these agents is to play tennis with each other. For an introduction and instructions to get this project to run please read the [README.md](README.md).


![Untrained agent][image1]


# Architecture
The following is a description of the final architecture for the Tennis environment. In the next chapter it is explained how we got here. The solution consists of a multi-agent set-up in which two DPPG agents play against each other. Both share the same architecture: a pytorch neural network with two hidden layer and an output layer. The weights of their network are different. Furthermore, they make use of the same memory to train their networks and improve. This memory is a prioritized experience replay buffer, in which more importance is given to experiences that are currently not estimated correctly. There exist multiple versions of implementing this. This version was inspired by the work of [the-computer-scientist](https://github.com/the-computer-scientist/OpenAIGym/blob/master/PrioritizedExperienceReplayInOpenAIGym.ipynb) ([youtube](https://www.youtube.com/watch?v=MqZmwQoOXw4&ab_channel=TheComputerScientist)). Within this implementation a (high) default importance is given to each new experience. During training this importance is updated by calculting the TD error. This is the difference between expected value of the (local) critic and the received reward and the expected value of the next state (as calculated by the target actor and critic). When the importance is high, the experience is more likely to be picked during the next training phase.

The DDPG is an algorithm that uses an actor and a critic network. The actor is responsibe for choosing the best action. The critic is responsible for estimating the value of a state/action pair.

The solution is able to solve the goal in 977 episodes:

![Best result][image2]

# Architecture Neural Network
The agent makes use of a pytorch deep neural network with two hidden layers for all it's networks (local actor, target actor, local critic, local actor). By default we use [ReLU](https://www.kaggle.com/code/dansbecker/rectified-linear-units-relu-in-deep-learning) activation function, and a [batchnorm layer](https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739) to normalize the outputs and speed up learning.

The following networks is the setup for the actor networks:
- a linear hidden layer that receives the state (inputsize 24) and outputs 128. A batchnorm function normalizes the output, and a ReLU sets negative values to 0
- a linear hidden layer that receives the previous 128 and outputs 256. A batchnorm function normalizes the output, and a ReLU sets negative values to 0
- a linear output layer that receives the previous 256 output values, and outputs 2 values. The activation function is tanh.

The crtitic networks have a similar setup. The only difference is that the actions chosen are also inputs for the second hidden layer, making them have an input of 130, and that the output layer only outputs 1 value.

# Process
1. Initialize the environments
1. Get the states for both agents
1. Use the local version of the actor to choose the right movement (up/down, left/right)
1. Save the states and rewards to the replay memory
1. Check if it is time to train. The agents should have gathered enough experiences in the replay memory. Furthermore it is set to only train each 'X' timestep
1. If it is time for training, perform for each agent the following tasks: 

    A. Train the local critic network

    B. Train the local actor network

    C. Soft update the target networks for both actor and critic by marginally getting these networks closer to the local (parameter tau, by default set to 1e-3)

    D. return the td errors so the importance of the experiences can ber adjusted in the prioritized experience replay buffer.
1. Check whether the environment is done or maximum amount of timesteps has been reached. If not, increase the timestep and go back to step 2. Otherwise break the loop and go to step 8.
1. Save the scores and check whether to goal has been reached. Otherwise, run a new episode by going to step 1.

# Solution
The solution is structured in this way:
- main.py: python scripts that runs the simulation
- rl_environment.py: a wrapper around the Unity environment that handles the looping over episodes and timesteps
- m_agent.py: a class that directs the two agents and is responsible for saving the experiences to the buffer
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

# Used Hyperparameters
The following table gives an overview of the used hyperparameters. The values in bold are different from the previous project

| Parameter     |                           Description                          | Banana project   | Reacher project    |**Tennis Project** |
|:--------------|:---------------------------------------------------------------|-----------------:|-------------------:|------------------:|
| buffer_size   |                       replay buffer size                       | 128              | 1024               | 1024              |
| batch_size    |                         mini batch size                        | 1e5              | 1e6                | **1e5**           |
| update_every  |                   timesteps between training                   | 4                | 10                 |**5**              |
| gamma         | discount factor                                                | 0.99             | 0.99               | 0.99              |
| tau           | soft update of target parameters of both models                | 1e-3             | 1e-3               |**1e-2**           |
| lr_actor      | learning rate of actor                                         | 5e-4             | 1e-3               | 1e-3              |
| lr_critic     | learning rate of critic                                        |                  | 1e-4               | **1e-3**          |
| layer_size_1  | output size of the first layer                                 | 32               | 128                | 128               |
| layer_size_2  | output size of the second layer                                | 32               | 256                | 256               |
| activation    | what kind of activation is used after the output of the layers | ReLU             | ReLU               | ReLU              |
| normalization | whether a normalization is used within the layers              | -                | batchnorm1d        | batchnorm1d       |



# Improvements
At first the agent was a copy from the reacher project. After some small adjustments this resulted in the following gameplay in which the agents seemed to not learn anything at all. 

![Untrained agent][image3]

The following iterations were made during the development of the solution:
- **reintroducing eps**: at first it seemed that the agents got stuck in a local optimum in which they just seemed to avoid the ball. Hence the eps was reintroduced to trigger new gameplay. This value is high at the beginning and low in the later phase of the training. At the beginning of the game the agent would therefor make more often random movements than later in the game. This however, did not seem to help. In stead of making random movements some of the time, the current algorith includes Ornstein–Uhlenbeck Noise: this adds some noise on top of the chosen values.
- **lowering the discount factor from 0.95 to 0.99**: another finding was that in comparison with the reacher project the reward of a good move is only received after quite some time. While the arm in the reacher project has to chase the ball in order to receive a reward, the racket within the tennis project has to wait quite some timesteps after a good move until a reward is received. The moves made by the racket after the ball was hit, and before the score is reeived don't even have an influence anymore. This makes it more important to have a higher discount factor.
- **increasing the speed of the training**: the model seemed to learn too slow. Therefore, the update frequency (after how many timesteps the model should be retrained) was set from once every 10 to every timestep. This however resulted in very long runtimes. Instead of updating the model more frequent, the learning rates of both actor and critic were increased by a tenfold (from 1e4 to 1e3) and update frequency was again set to once every 10 timesteps. This resulted in much lower running times.
- **increasing the tau**: after all these changes the agents still did not seem to learn how to play the game. The tau was increased from 0.001 to 0.01. This means the target models of the actor and critic are more in line with the local versions. 
- **Prioritized Experience Replay**: In order to further speed up the training a version of Prioritized Experience Replay was implemented. 
- **debugging**: after some more debugging it was found that the weights of the actor did not change. It was concluded this was due to a bug in the code. The pytorch model is configured using a json configuration file that contained the layers and it's weights. During the forward pass the layers are iterated using the enumerate function. Pytorch does not 
- **amount of layers**: finally different amounts of hidden layers were used. 128,128, 128,256 and 64,64. The first two versions hardly differ in terms of performance, but 64 layers seem to be a too small number.

# Graph of results of other training runs
In the graph below the rolling averages of other experiments are displayed. Within these experiments the values of batch, gamma, learning rate, update frequency and layer size were altered. Within multiple experiments you can see the score collapsing after reaching a high score. After some time the scores seem to reach their previous high score, although it can take a while. The best performing set-up reached a score of 0.5 after 977 episodes. When it was copied it collapsed, only to return to a higher score after almost 2000 episodes.
![Results][image4]

# Ideas for Future Work
The goal was to reach an average score of 0.5 over 100 consecutive episodes. This goal was reached. Another attempt was made to further increase the score. After 2000 episodes the agents received an only marginally better score: 0.53. To further increase the score other algorithms could be looked into.

Right now the agents are both trained on the same experiences. This is suboptimal, since an agent will never play on the opposite side. It means that 50% of the experiences that the agent is trained on will never occur. Although this do means the agents could be interchanged, it probably negatively impacts the training time.

Within this project a version of Prioritized Experience Replay was introduced. This reduced the training time, but adding and sampling the experiences costed 12% of the processing time. When the amount of experiences grew this processing time increased. This could be resolved by using a sumtree instead of a deque.