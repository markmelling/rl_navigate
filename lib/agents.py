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

    def __init__(self, state_size, action_size, seed, local_model, target_model, train_mode=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.train_mode = train_mode
        self.state_size = state_size
        self.action_size = action_size
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
        #state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        state = torch.from_numpy(state).float().to(device)

        if self.train_mode:
            self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        if self.train_mode:
            self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

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
        self.qnetwork_local.load_state_dict(torch.load(f'{path}/{name}_local.pth'))
        self.qnetwork_local.eval()
        self.qnetwork_target.load_state_dict(torch.load(f'{path}/{name}_target.pth'))
        self.qnetwork_target.eval()
        
    def save_model(self, name, path=MODEL_PATH):
        torch.save(self.qnetwork_local.state_dict(), f'{path}/{name}_local.pth')
        torch.save(self.qnetwork_local.state_dict(), f'{path}/{name}_target.pth')


class AgentExerperienceReplay(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, train_mode=True, create_model=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        #self.state_size = state_size
        #self.action_size = action_size
        #self.seed = random.seed(seed)
        if create_model:
            local_model = create_model(state_size, action_size, seed)
            target_model = create_model(state_size, action_size, seed)
        else:
            local_model = QNetwork(state_size, action_size, seed)
            target_model = QNetwork(state_size, action_size, seed)
            
        super(AgentExerperienceReplay, self).__init__(state_size, 
                                                      action_size, 
                                                      seed,
                                                      local_model,
                                                      target_model,
                                                      train_mode=train_mode)


        # Q-Network
        #self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        #self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        #self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        if not self.train_mode:
            return
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
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

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU) 
        

            
class AgentPrioritizedExperience(AgentBase):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, 
                 compute_weights=False, 
                 prioritized_experience=True, 
                 dueling_nn=False,
                 train_mode=True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        #self.state_size = state_size
        #self.action_size = action_size
        #self.seed = random.seed(seed)
        #super(AgentPrioritizedExperience, self).__init__(state_size, action_size, seed, train_mode=train_mode)
                # Q-Network
        if dueling_nn:
            local = DuelingQNetwork(state_size, action_size, seed)
            target = DuelingQNetwork(state_size, action_size, seed)
        else:
            local = DuelingQNetwork(state_size, action_size, seed)
            target = DuelingQNetwork(state_size, action_size, seed)

        super(AgentPrioritizedExperience, self).__init__(state_size, 
                                                      action_size, 
                                                      seed,
                                                      local,
                                                      target,
                                                      train_mode=train_mode)

        self.compute_weights = compute_weights
        self.prioritized_experience = prioritized_experience
        


        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        if prioritized_experience:
            self.memory = PrioritizedReplayBuffer(BUFFER_SIZE,
                                                  BATCH_SIZE,
                                                  seed)
        else:
            self.memory = ReplayBuffer(action_size, 
                                       BUFFER_SIZE, 
                                       BATCH_SIZE, 
                                       seed, prioritized_experience=prioritized_experience, compute_weights=compute_weights)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        #self.t_step = 0
        # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
        self.t_step_mem_par = 0

    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        self.t_step_mem_par = (self.t_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY
        if self.prioritized_experience and self.t_step_mem_par == 0:
            self.memory.update_hyperparameters()
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.experience_count > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    #def act(self, state, eps=0.):
    #    """Returns actions for given state as per current policy.
    #    
    #    Params
    #    ======
    #        state (array_like): current state
    #        eps (float): epsilon, for epsilon-greedy action selection
    #    """
    #    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    #    self.qnetwork_local.eval()
    #    with torch.no_grad():
    #        action_values = self.qnetwork_local(state)
    #    self.qnetwork_local.train()

        # Epsilon-greedy action selection
    #    if random.random() > eps:
    #        return np.argmax(action_values.cpu().data.numpy())
    #    else:
    #        return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indexes = experiences

        # Get max predicted Q values (for next states) from target model
        #Q_targets_next = self.qnetwork_target(next_states).detach()
        #Q_targets_next = Q_targets_next.max(1)[0].unsqueeze(1)
        
        # select greedy action
        #    _, actions = q_network(states).max(dim=1, keepdim=True)
        
        # using local (online) nn to select actions

        #Q_targets_next = self.qnetwork_local(next_states).detach()
        #Q_targets_next = Q_targets_next.max(1)[0].unsqueeze(1)
        # TODO we could switch the networks around and using target for selecting
        # and local for evaluating
        _, next_actions = self.qnetwork_local(next_states).max(dim=1, keepdim=True)
 
        # evaluate action      
        # using target nn to evaluate the actions
        Q_targets_values = self.qnetwork_target(next_states).gather(dim=1, index=next_actions)        
        Q_targets = rewards + (gamma * Q_targets_values * (1 - dones))
        

        # moved down
        # Get expected Q values from local model
        #Q_expected = self.qnetwork_local(states)
        #Q_expected = Q_expected.gather(1, actions)
        #Q_expected = Q_expected.gather(1, Q_target_next)

        # added for DDQN
        #Q_eval_max_actions = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        # some how rather than Q_targets_next max actions we want local max actions 
        # so this is the maximal actions according to the local (online) nn
        # rather than the maximal actions according to the target nn
        # evaluate selected actions
        #Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # q_targets = rewards + gamma * q_next[indices, Q_eval_max_actions]

       # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states)
        Q_expected = Q_expected.gather(1, actions)
        #Q_expected = Q_expected.gather(1, Q_target_next)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        if self.compute_weights:
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
        if self.prioritized_experience:
            deltas = abs(Q_expected.detach() - Q_targets.detach()).numpy()
            self.memory.update_priorities(deltas, indexes)  



    #def soft_update(self, local_model, target_model, tau):
    #    """Soft update model parameters.
    #    θ_target = τ*θ_local + (1 - τ)*θ_target

    #    Params
    #    ======
    #        local_model (PyTorch model): weights will be copied from
    #        target_model (PyTorch model): weights will be copied to
    #        tau (float): interpolation parameter 
    #    """
    #    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
    #        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


