[//]: # (Image References)

[image1]: results/trained_agent.gif "Trained Agent"

# Reacher Agent
## Project Details

Within this project we will develop an agent that move an arm around. The goal is to get the arm within the ball for as long as possible. This project is part of the Udacity Deep learning Program and is an adapted version of the Unity  [ML-Agents repository](https://github.com/Unity-Technologies/ml-agents).

In the image below the final trained agent within the environment is shown. The agent receives 33 inputs for each arm ("the state space is 33"), such as position, rotation, velocity, and angular velocities. The action consists of a vector of 4 numbers between -1 and 1. Whenever the arm is within the highlighted ball the agent receives a score of +0.1. It is thus the goal to reach the ball and stay within that region.

There exist two versions of the environment. One with just a single arm (and agent), and one with 20 arms. The last version makes the training faster. We will use the last version within this project.

![Trained Agent][image1]

## Getting Started
This is a python project. In order to run it you will need follow these steps:
1. install Visual Studio Code or another python editor. In case you are running Visual Studio Code install an older version of the 'Python' extension (v2021.9.1246542782) since the Python 3.6 is no longer supported in more recent versions.
1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
1. Follow the instructions in [this repository](https://github.com/openai/gym) to perform a minimal install of OpenAI gym.  
	- Install the **box2d** environment group by following the instructions [here](https://github.com/openai/gym#box2d).
	
1. Clone this repository (if you haven't already!)

1. install the requirements in requirements.txt by typing the following command in the terminal
    ```
    pip install -r requirements.txt
    ```

1. Download the reacher environment within the same folder as you have put this project
    - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)


1. Change the `file_name` parameter within main.py and Navigation.ipynb to match the location of the Unity environment that you downloaded.

    - **Mac**: `"path/to/Reacher.app"`
    - **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
    - **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
    - **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
    - **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
    - **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
    - **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`

    For instance, if you are using Windows 64-bit, then you downloaded `Reacher_Windows_x86_64.zip`.  If you unzipped this in the same folder as this project then the line below should appear as follows:
    ```
    env = UnityEnvironment(file_name="Reacher_Windows_x86_64/Reacher.exe")
    ```

You are now ready to run the agent!

## Instructions
In order to train the agent you will have to run the python file main.py using F5 (debug) or CTRL + F5 (run without debugging). Since python 3.6 is no longer supported in Jupyter notebook a ipynb was not developed.

A Unity Environment application should start up within Visual Studio Code.  The python script connects to this environment and sends commands to the environment. You will see a sped up version of the game in order to make the training faster. The python file will train an agent. Depending on the machine it should take about 3 hours. Afterwards the trained model of the agent is saved within 'results/checkpoint_*.pt' file aswell as to the output folder. Moreover, the results, model and matplotlib-graph will be saved to the output folder.
