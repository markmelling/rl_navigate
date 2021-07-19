#
# Sumtree for holding priorities
# the basic idea is that each item has a accumulated value that 
# will correspond to to the likelihood that it should be selected 
#
import numpy as np
import torch
import random
from collections import namedtuple, deque
import queue
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# sumtree implementation is heavily based on this
# https://github.com/adventuresinML/adventures-in-ml-code/blob/master/sum_tree_intro.py
class Node:
    def __init__(self, left, right, is_leaf: bool = False, idx = None):
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        if not self.is_leaf:
            self.value = self.left.value + self.right.value
        self.parent = None
        self.idx = idx  # this value is only set for leaf nodes
        if left is not None:
            left.parent = self
        if right is not None:
            right.parent = self

    @classmethod
    def create_leaf(cls, value, idx):
        leaf = cls(None, None, is_leaf=True, idx=idx)
        leaf.value = value
        return leaf


def create_tree(input: list):
    nodes = [Node.create_leaf(v, i) for i, v in enumerate(input)]
    leaf_nodes = nodes
    while len(nodes) > 1:
        inodes = iter(nodes)
        nodes = [Node(*pair) for pair in zip(inodes, inodes)]

    return nodes[0], leaf_nodes


def retrieve_node(value: float, node: Node):
    if node.is_leaf:
        return node

    if node.left.value >= value:
        return retrieve_node(value, node.left)
    else:
        return retrieve_node(value - node.left.value, node.right)


def update_node(node: Node, new_value: float):
    change = new_value - node.value

    node.value = new_value
    propagate_changes(change, node.parent)


def propagate_changes(change: float, node: Node):
    node.value += change

    if node.parent is not None:
        propagate_changes(change, node.parent)


        
class PrioritizedReplayBuffer(object):
    # TODO remove this when ready
    def __init__(self, 
                 size: int, 
                 batch_size: int, 
                 seed: int, 
                 alpha = 0.6,
                 beta = 0.4):
        # alpha = 0.6 and beta = 0.4 recommended in paper for proportional variant

        self.size = size
        self.batch_size = batch_size
        np.random.seed(seed)
        self.alpha = alpha
        self.alpha_decay_rate = 0.99
        self.beta = beta
        self.beta_growth_rate = 1.001
        # min_priority prevents edge-case of transitions not being revisited
        # once their error is zero
        self.min_priority = 0.01 
        self.max_priority = 0

        self.curr_write_idx = 0
        self.experience_count = 0
        # sampled_count was included so as to conduct some tests 
        # on the behaviour of the buffer 
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "sampled_count"])

        
        # this was used to store experiences when they are first recieved
        # when used experienced will be sampled before experiences already with a priority 
        # once sampled they will get a priority and consequently will be sampled 
        # according to their priority
        #self.not_sampled = queue.Queue()

        # allocate the buffer at start - better than failing half way through!
        self.buffer = [self.experience for i in range(self.size)]
        self.raw_priorities = [0 for i in range(self.size)]
        # create sumtree used for storing experience priorities
        # this provides an efficient way of sampling experiences proportional with their priority (td error)
        self.base_node, self.leaf_nodes = create_tree([0 for i in range(self.size)])

        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, 0)
       
        # Note - no longer used 
        # add index to 'not_sampled' queue so these can be provided first 
        # when sampling so as to ensure they are seen at least once 
        #  self.not_sampled.put(self.curr_write_idx) 
        # add experience to buffer
        self.buffer[self.curr_write_idx] = e
        # Note - no longer used 
        # setting the priority of a new experience to 0 as it will be sampled
        # first anyway, as it is in not_sampled queue
        priority = self.max_priority
        self.update_priority(self.curr_write_idx, priority)

        self.curr_write_idx += 1
        # if the index is >= buffer size then loop round to begining
        if self.curr_write_idx >= self.size:
            self.curr_write_idx = 0
        # max out available samples at the memory buffer size
        self.experience_count += 1 
   
    def update_priorities(self, deltas, idxs):
        for i in range(len(idxs)):
            self.update_priority(idxs[i], deltas[i][0])
            
    def update_priority(self, idx: int, priority: float):
        self.raw_priorities[idx] = priority
        update_node(self.leaf_nodes[idx], self.adjust_priority(priority))
        
    # see paper adding epsilon to priority and then taking to power of alpha
    def adjust_priority(self, priority: float):
        if self.max_priority < priority:
            self.max_priority = priority
            # print('updating max priority', self.max_priority)
        return np.power(priority + self.min_priority, self.alpha)
    
    def update_hyperparameters(self):
        """
        Update hyper parameters
        """
        # it says (page 4) that segment boundaries only change when N or alpha change
        # re-computing the priorities when alpha decays
  
        #prev_alpha = self.alpha
        self.alpha *= self.alpha_decay_rate
        self.max_priority = 0
        for i in range(min(self.experience_count, self.size)):
            self.update_priority(i, self.raw_priorities[i])

        # print('update_hyperparameters max_priority', self.max_priority)
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1
    
    def sample(self):
        # assumes there are sufficient samples available
        # sample with proportional prioritization
        sampled_idxs = []
        samples = []
        is_weights = []
        # Proportional prioritization:
        # To sample a minibatch of size k, the range [0, ptotal] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range.
        # Finally the transitions that correspond to each of these sampled values are retrieved from the tree
        segment_size = self.base_node.value/self.batch_size

        for i in range(self.batch_size):
            # Note: this is no longer used, hence the assert 
            if not self.not_sampled.empty():
                assert False
                sample_idx = self.not_sampled.get()
                sample_node = self.leaf_nodes[sample_idx]
            else:
                start = i * segment_size
                end = (i + 1) * segment_size
                sample_val = np.random.uniform(start, end)
                #sample_val = np.random.uniform(0, self.base_node.value)
                sample_node = retrieve_node(sample_val, self.base_node)
            
            sampled_idxs.append(sample_node.idx)
            e = self.buffer[sample_node.idx]
            # update sampled count
            e = self.experience(e.state, 
                                e.action, 
                                e.reward, 
                                e.next_state, 
                                e.done,
                                e.sampled_count + 1)
            self.buffer[sample_node.idx] = e
            samples.append(e)
            prob = sample_node.value / self.base_node.value
            available_samples = min(self.batch_size, self.experience_count)
            is_weights.append((available_samples + 1) * prob)
        is_weights = np.array(is_weights)
        is_weights = np.power(is_weights, -self.beta)
        # scale weights so the max weight == 1
        is_weights = is_weights / np.max(is_weights)
      
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, is_weights, sampled_idxs)

    def save(self, name):
        print('save not implemented')
            
    def load(self, name):
        print('load not implemented')


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
    
    def sample_for_images(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        # print('sample before first experience state shape', experiences[0].state.shape)

        # In the case of an image they are 84,84,3 vstack is equivalent to concatenating the first dimension
        # so for two images you would have 168, 84, 3 
        # If we stored images as their original single item batch i.e. 1, 84, 84, 3 then
        # vstack would do the right thing as it would be like concatenating the first dimension
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        # print('sample afterfirst experience state shape', states[0].shape)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def save(self, name):
        d = [e._asdict() for e in self.memory]
        with open(f'{name}.pickle', 'wb') as fp:
            pickle.dump(d, fp)
            
    def load(self, name):
        try:
            with open(f'{name}.pickle', 'rb') as fp:
                d = pickle.load(fp)
            for ex in d:
                self.add(ex['state'], ex['action'], ex['reward'], ex['next_state'], ex['done'])

        except Exception:
            print('not found')

