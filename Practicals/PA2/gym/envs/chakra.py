from gym import Env
from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
import numpy as np


class chakra(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,))

        self._seed()
        self.viewer = None
        self.state = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        #Fill your code here
        if np.linalg.norm(action) > 0.025:
            action[0] = action[0]/np.linalg.norm(action) * 0.025
            action[1] = action[1]/np.linalg.norm(action) * 0.025
        
        self.state = np.array([self.state[0]+action[0], self.state[1]+action[1]])

        if 1 < self.state[0] or self.state[0]<-1  or 1 < self.state[1] or self.state[1]<-1:
            self.reset()

        reward = -np.linalg.norm(self.state)
        if self.state[0] == 0 and self.state[1] ==0:
            done = True
        else:
            done = False

        obs = np.array(self.state)
        return obs, reward, done, None

    def _reset(self):
        while True:
            self.state = self.np_random.uniform(low=-1, high=1, size=(2,))
            # Sample states that are far away
            if np.linalg.norm(self.state) > 0.9:
                break
        return np.array(self.state)

    # method for rendering

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            origin = rendering.make_circle(
                min(screen_height, screen_width) * 0.03)
            trans = rendering.Transform(translation=(0, 0))
            agent.add_attr(trans)
            self.trans = trans
            agent.set_color(1, 0, 0)
            origin.set_color(0, 0, 0)
            origin.add_attr(rendering.Transform(
                translation=(screen_width // 2, screen_height // 2)))
            self.viewer.add_geom(agent)
            self.viewer.add_geom(origin)

        # self.trans.set_translation(0, 0)
        self.trans.set_translation(
            (self.state[0] + 1) / 2 * screen_width,
            (self.state[1] + 1) / 2 * screen_height,
        )

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

