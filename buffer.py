import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # We create large empty tables in RAM to store history
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)

    def add(self, obs, action, reward, next_obs, done):
        """Adds one frame (moment) to memory."""
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done
        
        # Circular buffer: When it reaches the end (eg 100,000), it resets to zero and overwrites the oldest
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, horizon):
        """
        The most important function for TD-MPC!
        Extracts 'batch_size' (eg 512) shorts of length 'horizon' (eg 5).
        """
        # We are preparing empty 3D tensors that we will fill with movies
        batch_obs = np.zeros((batch_size, horizon + 1, self.obs.shape[1]), dtype=np.float32)
        batch_action = np.zeros((batch_size, horizon, self.action.shape[1]), dtype=np.float32)
        batch_reward = np.zeros((batch_size, horizon, 1), dtype=np.float32)
        
        for i in range(batch_size):
            while True:
                # 1. We choose a random index (the beginning of the movie)
                # Be careful not to select the very end of the buffer so that we miss few frames
                idx = np.random.randint(0, self.size - horizon - 1)
                
                # 2. CHECK: Did the robot "die" (done=True) in the middle of this movie?
                # We can't study on a movie where the episode is cut in half!
                if self.done[idx : idx + horizon].any():
                    continue 
                    
                # 3. If the film is good, cut it from the big table and put it in our package
                batch_obs[i] = self.obs[idx : idx + horizon + 1] # H+1 states
                batch_action[i] = self.action[idx : idx + horizon] # H action
                batch_reward[i] = self.reward[idx : idx + horizon] # H reward
                break # We successfully found 1 movie, let's go to the next one in the for loop
                
        return batch_obs, batch_action, batch_reward