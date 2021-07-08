from collections import deque
import numpy as np

def get_state(env_info):
    return env_info.vector_observations[0] 

def dqn(env, 
        brain_name, 
        agent, 
        train_mode=True, 
        n_episodes=2000, 
        max_t=1000, 
        eps_start=1.0, 
        eps_end=0.01, 
        eps_decay=0.995,
        checkpoint=None,
        get_state=get_state,
        batched_state=False,
        agent_name=''):
    """Deep Q-Learning.
    
    Params
    ======
        env (UnityEnvironment)
        brain_name (str)
        agent (instance of class derived from AgentBase)
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        batched_state: in the case of image based state then these are received in batches of 1
        (1, 84, 84, 3) so need to be handled slightly differently
        
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_mean_score = 0
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train_mode)[brain_name]
        state = get_state(env_info)            # get the current state
        score = 0
        for t in range(max_t):
            if batched_state:
                no_batch_state = np.squeeze(state)
                #print('batched_state size', no_batch_state.shape)
            else:
                no_batch_state = state 
            action = agent.act(no_batch_state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = get_state(env_info)  # get the next state
            done = env_info.local_done[0]                  # see if episode has finished
            reward = env_info.rewards[0]                   # get the reward
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        mean_score = np.mean(scores_window)
        if train_mode and checkpoint and mean_score >= checkpoint and mean_score >= max_mean_score:
            max_mean_score = mean_score
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, mean_score))
            agent.save_model(f'checkpoint-{agent_name}-{np.round(max_mean_score, 2)}')
            #break
    return scores
