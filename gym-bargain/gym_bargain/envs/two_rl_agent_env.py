import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class TwoRLAgentEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        
        # Angle at which to fail the episode
        self.x_neg_threshold = -5
        self.y_neg_threshold = -5
        self.x_pos_threshold = 5
        self.y_pos_threshold = 5

        self.delta_t = 0.01

        self.seed()
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        action:list = [[a,b,c,d], [p,q,r,s]]
        '''
        xt, yt = self.state

        action1 = action[0]
        action2 = action[1]
        # a,b,c,d = action # feed action wtih 0-1 range for easeness in sigmoid and translate back to -1 to 1
        a = action1[0]*2-1
        b = action1[1]*2-1
        c = action1[2]*2-1
        d = action1[3]*2-1 

        p = action2[0]*2-1
        q = action2[1]*2-1
        r = action2[2]*2-1
        s = action2[3]*2-1 

        xt = xt+ self.delta_t*(a*(xt+np.abs(xt))/2+b*(xt-np.abs(xt))/2+c*(yt+np.abs(yt))/2+d*(yt-np.abs(yt))/2)
        yt = yt+ self.delta_t*(p*(xt+np.abs(xt))/2+q*(xt-np.abs(xt))/2+r*(yt+np.abs(yt))/2+s*(yt-np.abs(yt))/2)

        self.state = (xt, yt)

        done = bool(
            xt < self.x_neg_threshold
            or xt > self.x_pos_threshold
            or yt < self.y_neg_threshold
            or yt > self.y_pos_threshold
        )

        if not done:
            if xt>=0 and yt>=0: 
                reward1 = xt*yt/2
                reward2 = xt*yt/2
            elif xt<=0 and yt>=0: 
                reward1 = -1*xt*yt
                reward2 = xt*yt/2
            elif xt<=0 and yt<=0: 
                reward1 = -1*xt*yt
                reward2 = -1*xt*yt
            else: 
                reward1 = xt*yt/2
                reward2 = -1*xt*yt

        elif self.steps_beyond_done is None:
            # game just over!
            self.steps_beyond_done = 0
            if xt>=0 and yt>=0: 
                reward1 = xt*yt/2
                reward2 = xt*yt/2
            elif xt<=0 and yt>=0: 
                reward1 = -1*xt*yt
                reward2 = xt*yt/2
            elif xt<=0 and yt<=0: 
                reward1 = -1*xt*yt
                reward2 = -1*xt*yt
            else: 
                reward1 = xt*yt/2
                reward2 = -1*xt*yt
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward1 = 0.0
            reward2 = 0.0

        reward_n = [reward1, reward2]
        return np.array(self.state), reward_n, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-4, high=4, size=(2,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass