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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def demonstrate_sampling(root_node: Node):
    tree_total = root_node.value
    iterations = 1000000
    selected_vals = []
    for i in range(iterations):
        rand_val = np.random.uniform(0, tree_total)
        selected_val = retrieve(rand_val, root_node).value
        selected_vals.append(selected_val)
    
    return selected_vals

class PrioritizedReplayBuffer(object):
    # TODO remove this when ready
    def __init__(self, size: int, batch_size: int, seed: int):

        self.size = size
        self.batch_size = batch_size
        np.random.seed(seed)
        self.curr_write_idx = 0
        self.experience_count = 0
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

        # this is used to store experiences when they are first recieved
        # they will be sampled before experiences already with a priority 
        # once sampled they will get a priority and consequently will be sampled 
        # according to their priority
        self.not_sampled = queue.Queue()

        # allocate the buffer at start - better than failing half way through!
        self.buffer = [self.experience for i in range(self.size)]
        self.raw_priorities = [0 for i in range(self.size)]
        # create sumtree used for storing experience priorities
        # this provides an efficient way of sampling experiences proportional with their priority (td error)
        self.base_node, self.leaf_nodes = create_tree([0 for i in range(self.size)])
        # alpha = 0.6 and beta = 0.4 recommended in paper for proportional variant
        self.alpha = 0.6
        self.alpha_decay_rate = 0.99
        self.beta = 0.4
        self.beta_growth_rate = 1.001

        self.min_priority = 0.01
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        # add index to 'not_sampled' queue so these can be provided first 
        # when sampling so as to ensure they are seen at least once 
        self.not_sampled.put(self.curr_write_idx) 
        # add experience to buffer
        self.buffer[self.curr_write_idx] = e
        # setting the priority of a new experience to 0 as it will be sampled
        # first anyway, as it is in not_sampled queue
        priority = 0
        self.update_priority(self.curr_write_idx, priority)

        self.curr_write_idx += 1
        # if the index is >= buffer size then loop round to begining
        if self.curr_write_idx >= self.size:
            self.curr_write_idx = 0
        # max out available samples at the memory buffer size
        self.experience_count += 1 
        #if self.available_samples + 1 < self.size:
        #    self.available_samples += 1
        #else:
        #    self.available_samples = self.size - 1

        ## add experience to normal buffer, with 0 priority and
        ## also append to no priority buffer
        # update priority of experience in sumtree, for new experiences
        # I'm setting this to 0 as they will be selected first when sampled
        # 

    #def append(self, experience: tuple, priority: float):
    #    self.buffer[self.curr_write_idx] = experience
    #    self.update_priority(self.curr_write_idx, priority)
    #    self.curr_write_idx += 1
    #    # reset the current writer position index if creater than the allowed size
    #    if self.curr_write_idx >= self.size:
    #        self.curr_write_idx = 0
    #    # max out available samples at the memory buffer size
    #    if self.available_samples + 1 < self.size:
    #        self.available_samples += 1
    #    else:
    #        self.available_samples = self.size - 1
   
    def update_priorities(self, deltas, idxs):
        for i in range(len(idxs)):
            self.update_priority(idxs[i], deltas[i][0])
            
    def update_priority(self, idx: int, priority: float):
        self.raw_priorities[idx] = priority
        update_node(self.leaf_nodes[idx], self.adjust_priority(priority))
        
    # see paper adding epsilon to priority and then taking to power of alpha
    def adjust_priority(self, priority: float):
        return np.power(priority + self.min_priority, self.alpha)
    
    def update_hyperparameters(self):
        """
        Update hyper parameters
        """
        # it says (page 4) that segment boundaries only change when N or alpha change
        # does this mean we should be re-computing the priorities when alpha decays?
  
        prev_alpha = self.alpha
        self.alpha *= self.alpha_decay_rate
        for i in range(min(self.experience_count, self.size)):
            self.update_priority(i, self.raw_priorities[i])
    
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1
    
    def sample(self):
        # assumes there are sufficient samples available
        sampled_idxs = []
        samples = []
        is_weights = []
        # Proportional prioritization:
        # To sample a minibatch of size k, the range [0, ptotal] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range.
        # Finally the transitions that correspond to each of these sampled values are retrieved from the tree
        for i in range(self.batch_size):
            # get sample from 
            if not self.not_sampled.empty():
                sample_idx = self.not_sampled.get()
                sample_node = self.leaf_nodes[sample_idx]
            else:
                segment_size = self.base_node.value/self.batch_size
                start = i * segment_size
                end = (i + 1) * segment_size
                sample_val = np.random.uniform(start, end)
                #sample_val = np.random.uniform(0, self.base_node.value)
                sample_node = retrieve_node(sample_val, self.base_node)
            
            sampled_idxs.append(sample_node.idx)
            samples.append(self.buffer[sample_node.idx])
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

  #  
  #  
  #  states = np.zeros((num_samples, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
  #                           dtype=np.float32)
  #      next_states = np.zeros((num_samples, POST_PROCESS_IMAGE_SIZE[0], POST_PROCESS_IMAGE_SIZE[1], NUM_FRAMES),
  #                          dtype=np.float32)
  #      actions, rewards, terminal = [], [], [] 
  #      for i, idx in enumerate(sampled_idxs):
  #          for j in range(NUM_FRAMES):
  #              states[i, :, :, j] = self.buffer[idx + j - NUM_FRAMES + 1][self.frame_idx][:, :, 0]
  #              next_states[i, :, :, j] = self.buffer[idx + j - NUM_FRAMES + 2][self.frame_idx][:, :, 0]
  #          actions.append(self.buffer[idx][self.action_idx])
  #          rewards.append(self.buffer[idx][self.reward_idx])
  #          terminal.append(self.buffer[idx][self.terminal_idx])
  #      return states, np.array(actions), np.array(rewards), next_states, np.array(terminal), sampled_idxs, is_weights

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

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class ReplayBufferProbablyNotNeeded:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, prioritized_experience=False, compute_weights=False):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        # used for storing priorities
        self.seed = random.seed(seed)
        self.prioritized_experience = prioritized_experience
        self.alpha = 0.5
        self.alpha_decay_rate = 0.99
        self.beta = 0.5
        self.beta_growth_rate = 1.001
        self.experience_count = 0
        self.current_batch = 0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1
        self.compute_weights = compute_weights

        
        if prioritized_experience:
            # initialise memory and priorities based on index key
            # we're using an index so that we can update the priorities as they are updated during learning
            self.priority = namedtuple("Priority", field_names=["priority", "priority_alpha", "probability", "weight","index"])
            index = []
            priorities = []
            for i in range(buffer_size):
                index.append(i)
                p = self.priority(0,0,0,0,i)
                priorities.append(p)
            self.memory = {key: self.experience for key in index}
            self.memory_priorities = {key: priority for key,priority in zip(index, priorities)}
        else:
            self.memory = deque(maxlen=buffer_size)  
            #index = []
            #priorities = []
            #for i in range(buffer_size):
            #    index.append(i)
            #self.memory = {key: self.experience for key in index}

            # self.memory = deque(maxlen=buffer_size)  

    def update_priorities(self, deltas, indexes):
        for delta, index in zip(deltas, indexes):
            #print('update_priorities index', index)
            N = min(self.experience_count, self.buffer_size)

            updated_priority = delta[0]
            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            
            if self.compute_weights:
                # compute importance-sampling weight
                assert self.weights_max != 0, 'update_priorities:weights_max can\'t be 0'
                updated_weight = ((N * updated_priority)**(-self.beta))/self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1
            p = self.memory_priorities[index]
            #print('p', p)

            old_priority = self.memory_priorities[index].priority
            old_priority_alpha = self.memory_priorities[index].priority_alpha

            updated_priority_alpha = updated_priority**self.alpha
            self.priorities_sum_alpha +=  updated_priority_alpha - old_priority_alpha
            assert self.priorities_sum_alpha != 0, 'update_priorities:priorities_sum_alpha can\'t be 0'
            updated_probability = updated_priority**self.alpha / self.priorities_sum_alpha
            priority = self.priority(updated_priority, updated_priority_alpha, updated_probability, updated_weight, index) 
            self.memory_priorities[index] = priority
   
    def update_hyperparameters(self):
        """
        Update hyper parameters and then recalculate probabilities and weights
        """
        print('\nupdate_hpyerparameters')
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1
        N = min(self.experience_count, self.buffer_size)
        self.priorities_sum_alpha = 0
        sum_prob_before = 0
        for element in self.memory_priorities.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority**self.alpha
        sum_prob_after = 0
        print('about to update priorities #', len(self.memory_priorities))
        # this should be restricted to experience_count if < buffer size
        for element in self.memory_priorities.values():
            if element.priority != 0:
                priority_alpha = element.priority**self.alpha
                assert self.priorities_sum_alpha != 0, 'priorities_sum_alpha can\'t be 0'
                probability =  (priority_alpha / self.priorities_sum_alpha)
                assert probability != 0, 'Probability should not be zero'
                sum_prob_after += probability
                weight = 1
                if self.compute_weights:
                    #  compute importance-sampling weights
                    #print('N', N, 'element.prob', element.probability)
                    #print('beta', self.beta, 'weights_max', self.weights_max)
                    try: 
                        #print('compute_weights ', self.weights_max, 'probability', probability)
                        # self.weights_max != 0, 'weights_max can\'t be 0'
                        tmp = (N * probability)**(-self.beta)
                        #print('compute_weights ', self.weights_max, 'probability', probability, 'tmp', tmp, 'tmp type', type(tmp))
                        weight = tmp/self.weights_max
  
                        #weight = ((N *  probability)**(-self.beta))/self.weights_max
                    except ZeroDivisionError as e:
                        print('exception', e)

                priority = self.priority(element.priority, priority_alpha, probability, weight, element.index)
                self.memory_priorities[element.index] = priority
        print("sum_prob before", sum_prob_before)
        print("sum_prob after : ", sum_prob_after)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.experience_count += 1
        if not self.prioritized_experience:
            self.memory.append(e)
            return


        index = self.experience_count % self.buffer_size
        self.memory[index] = e
        
        # num of experiences is > buffer_size so check if experience being replaced
        # is the max_priority, if it is then need to update max with new max
        if self.experience_count > self.buffer_size:
            temp = self.memory_priorities[index]
            # reduce running sum by the removed transition
            # TODO this is not actually correct as the alpha now may not be
            # the same as the alpha when the transition was added
            #self.priorities_sum_alpha -= temp.priority**self.alpha
            self.priorities_sum_alpha -= temp.priority_alpha

            if temp.priority == self.priorities_max:
                # this won't work as can't be set
                self.memory_priorities[index].priority = 0
                self.priorities_max = max(self.memory_priorities.items(), key=operator.itemgetter(1)).priority
            if self.compute_weights:
                if temp.weight == self.weights_max:
                    # this won't work as can't be set
                    self.memory_priorities[index].weight = 0
                    self.weights_max = max(self.memory_priorities.items(), key=operator.itemgetter(2)).weight

        # As per the paper 'PRIORITIZED EXPERIENCE REPLAY'
        # New transitions arrive without a known TD-error, so we put them at maximal priority 
        # in order to guarantee that all experience is seen at least once
        priority = self.priorities_max
        weight = self.weights_max
        
        # As defined in equation (1) in paper 
        # In order to increase diversity and ensuring a non-zero probability of being 
        # sampled even for the lowest-priority transition the probability is calculated as
        # follows
        priority_alpha = priority ** self.alpha
        self.priorities_sum_alpha += priority_alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha

        p = self.priority(priority, priority_alpha, probability, weight, index)
        self.memory_priorities[index] = p

    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        indexes = []
        weights = []
        if not self.prioritized_experience:
            experiences = random.sample(self.memory, k=self.batch_size)
        else:
            # this will probably be very inefficient
            values = list(self.memory_priorities.values())
            random_priorities = random.choices(self.memory_priorities, 
                                               [priority.probability for priority in values], 
                                               k=self.batch_size)
            experiences = [self.memory.get(p.index) for p in random_priorities]
            indexes = [p.index for p in random_priorities]
            weights = [p.weight for p in random_priorities]
        #print('# experiences ', len(experiences))
        #print('first experience', str(experiences[0]))
        #print('first experience.state', experiences[0].state)
          

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, weights, indexes)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)