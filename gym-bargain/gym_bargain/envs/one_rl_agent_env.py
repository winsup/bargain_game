import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class OneRLAgentEnv(gym.Env):

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

        self.p = 1
        self.q = 1
        self.r = 0
        self.s = 0

        self.seed()
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        action:list = [a,b,c,d]
        '''
        xt, yt = self.state

        p = self.p
        q = self.q
        r = self.r
        s = self.s
        # a,b,c,d = action # feed action wtih 0-1 range for easeness in sigmoid and translate back to -1 to 1
        a = action[0]*2-1
        b = action[1]*2-1
        c = action[2]*2-1
        d = action[3]*2-1 

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
            if xt>=0 and yt>=0: reward = xt*yt/2
            elif xt<=0 and yt>=0: reward = -1*xt*yt
            elif xt<=0 and yt<=0: reward = -1*xt*yt
            else: reward = xt*yt/2

        elif self.steps_beyond_done is None:
            # game just over!
            self.steps_beyond_done = 0
            if xt>=0 and yt>=0: reward = xt*yt/2
            elif xt<=0 and yt>=0: reward = -1*xt*yt
            elif xt<=0 and yt<=0: reward = -1*xt*yt
            else: reward = xt*yt/2
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-4, high=4, size=(2,))
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass