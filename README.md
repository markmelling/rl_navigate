[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

In this project deep reinforcement learning has been used to train a single agent to collect bananas in a large, square world.

![Trained Agent][image1]

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to forward, backward, left and right.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The task is episodic, and the goal is for the agent to get an average score of +13 over 100 consecutive episodes.

To solve this problem a number of agent variants have been developed taking inspiration from the following papers:

1. [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) 
2. [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/pdf/1511.06581.pdf)
3. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/pdf/1509.06461.pdf)
4. [Prioritized Experience Replay](https://arxiv.org/pdf/1511.05952.pdf)

The basic agent was implemented using the algorithm described in [Human-level control through deep reinforcement learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) 

## Best and worst results

The agent that reached an average score of +13 over 100 consecutive episodes quickest used a uniformally sampled experience relay, a linear neural network (2 hidden layers) and double Q-learning, it achieved this in 429 episodes. This variant also managed to achieve the highest average for 100 episodes of over 16.

The worst result was using a prioritised experience replay with a dueling network, which took 679 episodes to reach the same mark.



## Getting Started

### Installing the dependencies
To run the code and Jupyter Notebooks you need make sure you have the Unity environment, Python and the required libraries installed.

#### Installing Python
The easiest way to install Python, or the correct version of Python, is to use conda, and the easiest way to install conda is to install miniconda.

If you don't have conda installed then follow the instructions here: [miniconda](https://docs.conda.io/en/latest/miniconda.html)

With conda installed, create and activate a conda environment which uses python 3.6 by entering the following commands in a terminal window

- For Linux or Mac
```shell
conda create --name drlnd python=3.6
source activate drlnd
```

- For Windows
```shell
conda create --name drlnd python=3.6
activate drlnd
```

#### Install the required python libraries
Clone this repository if you've not done it already, then navigate to the python subdirectory and install the dependencies by entering the following commands in a terminal window

```shell
git clone https://github.com/markmelling/rl_navigate
cd rl_navigate
pip install .
```

#### Install the pre-prepared Unity environment

Download the environment from one of the links below.  You need only select the environment that matches your operating system:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

#### Setting up Jupyter Notebooks

Before you can run any of the Jupyter notebooks you need to create an [IPython](https://ipython.readthedocs.io/en/stable/install/kernel_install.html) kernel

```shell
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

## Running the code
The easiest way to run the code to train an agent is to run the `Navigation.ipynb` notebook

First you need to start Jupyter notebooks - in a terminal window, in the root of this repository, type:

```shell
jupyter notebook
```
If everything is installed correctly a browser tab should be opened showing all the files in the root of the repository

Click on `Navigation.ipynb`

Before running any of the code in this notebook change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu

![](https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png)


Just follow the instructions in the Navigation notebook to either train a specific agent variant or all of them.

Have fun!

## Jupyter Notebooks

There are a number of notebooks that you may want to explore

1. Navigation.ipynb - this demonstrates how to train the different variants 
2. Report.ipynb - Provides more details on the different agents and models and the results of these trained agents
3. Navigation_Pixel.ipynb - This demonstrates an agent that uses an image based state.

To make it easy to just view the content of these notebooks I've created PDF versions.

## Source code
All source code, outside of the Jupyter notebooks is stored in the libs folder. 
- dqn.py: provides a function to run multiple espisodes of an agent, automatically saving the model if it achieves the goal
- agents.py: provides a base agent class and two subclasses:
    - AgentExperienceReplay implementing a experience replay algorithm
    - AgentPrioritizedExperienceReplay implementing a prioritised experience replay algorithm
- models.py: provides a number of alternative Neural Network models:
    - QNetwork: A basic linear model
    - DuelingQNetwork: A dueling linear model 
    - DuelingConvQNetwork: a dueling convolution network
- replay_buffers.py
    - ReplayBuffer: simple buffer to store experiences. samples are random
    - PrioritizedReplayBuffer: uses a sumtree to manage prioritised experiences
- image.py: image processing functions that are used to simplify the game images (state).

    




