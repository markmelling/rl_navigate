[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

In this project deep reinforcement learning has been used to train an agent to collect bananas in a large, square world.

![Trained Agent][image1]

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to forward, backward, left and right.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  The goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The task is episodic, and the goal is for the agent to get an average score of +13 over 100 consecutive episodes.

A number of agents have been developed taking inspiration from the following papers:

1. [link](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf "Human-level control through deep reinforcement learning") 
2. [link](https://arxiv.org/pdf/1511.06581.pdf "Dueling Network Architectures for Deep Reinforcement Learning")
3. [link](https://arxiv.org/pdf/1509.06461.pdf "Deep Reinforcement Learning with Double Q-learning")
4. [link](https://arxiv.org/pdf/1511.05952.pdf "Prioritized Experience Replay")

The basic agent was implemented using the algorithm described in [link](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf "Human-level control through deep reinforcement learning") 

## Running the code
If you want to experiment with the code then clone this repository please go to the Getting Started section below

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.


### Notebooks

There are a number of notebooks that you may want to explore

1. Navigation
2. Navigation Test


