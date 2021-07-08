import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import numpy as np
import random
from collections import namedtuple, deque

from lib.models import QNetwork, DuelingQNetwork
from lib.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

# prioritized experience replay
UPDATE_MEM_EVERY = 20          # how often to update the priorities
UPDATE_MEM_PAR_EVERY = 3000     # how often to update the hyperparameters
EXPERIENCES_PER_SAMPLING = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_EVERY)

MODEL_PATH = './saved_models'


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AgentBase():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, local_model, target_model, batch_size=BATCH_SIZE, train_mode=True, double_dqn=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.train_mode = train_mode
        self.double_dqn = double_dqn
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.qnetwork_local = local_model.to(device)
        self.qnetwork_target = target_model.to(device)


        self.t_step = 0
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # torch expects input (state in this case) to be defined in batches
        # so use unsqueeze to add a batch dimension of 1
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # TODO this is for states that are images
        #state = torch.from_numpy(state).float().to(device)

        #print('act state.shape', state.shape)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        if self.train_mode:
            self.qnetwork_local.train()

        # Epsilon-greedy action selection
        # Epsilon should still be > 0 even when not training
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # Double DQN is used to try and avoid over estimating action values
    # see https://arxiv.org/pdf/1509.06461.pdf
    # for Double DQN local NN is used for selection of the action
    # and the target NN is used for evaluating the action
    # when not using Double DQN the target NN is used for both    
    def select_action(self, next_states):
        if self.double_dqn:
            _, next_actions = self.qnetwork_local(next_states).max(dim=1, keepdim=True)
        else:
            _, next_actions = self.qnetwork_target(next_states).max(dim=1, keepdim=True)
        return next_actions
    
    def evaluate_actions(self, next_states, next_actions):
        q_targets_values = self.qnetwork_target(next_states).gather(dim=1, index=next_actions)        
        return q_targets_values
    
    def get_target_and_expected(self, states, actions, rewards, next_states, dones, gamma):
        # get argmax
        # Get max predicted Q values (for next states) from local model
        # print('get_target_and_expected states.shape', states.shape)
        # print('get_target_and_expected next_states.shape', next_states.shape)


        next_actions = self.select_action(next_states)

        #_, next_actions = self.qnetwork_local(next_states).max(dim=1, keepdim=True)
 
        # evaluate action      
        # using target nn to evaluate the actions
        #q_targets_values = self.qnetwork_target(next_states).gather(dim=1, index=next_actions)        
        q_targets_values = self.evaluate_actions(next_states, next_actions)
        
        q_targets = rewards + (gamma * q_targets_values * (1 - dones))
        
        # Get expected Q values from local model
        q_expected = self.qnetwork_local(states).gather(1, actions)
        
        return q_expected, q_targets


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def load_model(self, name, path=MODEL_PATH):
        self.qnetwork_local.load_state_dict(torch.load(f'{path}/{name}.pth'))
        self.qnetwork_local.eval()
        self.qnetwork_target.load_state_dict(torch.load(f'{path}/{name}.pth'))
        self.qnetwork_target.eval()
        
    def save_model(self, name, path=MODEL_PATH):
        torch.save(self.qnetwork_local.state_dict(), f'{path}/{name}.pth')


class AgentExperienceReplay(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, batch_size=BATCH_SIZE, 
                 train_mode=True, create_model=None, double_dqn=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        if create_model:
            local_model = create_model(state_size, action_size, seed)
            target_model = create_model(state_size, action_size, seed)
        else:
            local_model = QNetwork(state_size, action_size, seed)
            target_model = QNetwork(state_size, action_size, seed)
            
        super(AgentExperienceReplay, self).__init__(state_size, 
                                                      action_size, 
                                                      seed,
                                                      local_model,
                                                      target_model,
                                                      batch_size=batch_size,
                                                      train_mode=train_mode,
                                                      double_dqn=double_dqn)


        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, self.batch_size, seed)
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if not self.train_mode:
            return
        #print('step state.shape', state.shape)
        #print('step next_state.shape', next_state.shape)

        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        # print('learn states.shape', states.shape)
        # print('learn next_states.shape', next_states.shape)
        
        q_expected, q_targets = self.get_target_and_expected(states, 
                                                             actions, 
                                                             rewards, 
                                                             next_states, 
                                                             dones, 
                                                             gamma)


        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 
        

            
class AgentPrioritizedExperienceReplay(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, 
                 batch_size=BATCH_SIZE,
                 train_mode=True,
                 create_model=None,
                 double_dqn=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        # Q-Network
        if create_model:
            local_model = create_model(state_size, action_size, seed)
            target_model = create_model(state_size, action_size, seed)
        else:
            local_model = DuelingQNetwork(state_size, action_size, seed)
            target_model = DuelingQNetwork(state_size, action_size, seed)

        super(AgentPrioritizedExperienceReplay, self).__init__(state_size, 
                                                      action_size, 
                                                      seed,
                                                      local_model,
                                                      target_model,
                                                      train_mode=train_mode,
                                                      batch_size=batch_size,
                                                      double_dqn=double_dqn)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE,
                                                  self.batch_size,
                                                  seed)
        self.t_step_mem_par = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if not self.train_mode:
            return
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY
        if self.t_step_mem_par == 0:
            self.memory.update_hyperparameters()
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.experience_count > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indexes = experiences

        q_expected, q_targets = self.get_target_and_expected(states, 
                                                             actions, 
                                                             rewards, 
                                                             next_states, 
                                                             dones, 
                                                             gamma)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)
        with torch.no_grad():
            weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            loss *= weight
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 
        
        # ------------------- update priorities ------------------- #  
        deltas = abs(q_expected.detach() - q_targets.detach()).numpy()
        self.memory.update_priorities(deltas, indexes)  

