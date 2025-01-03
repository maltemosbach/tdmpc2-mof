from collections import defaultdict, deque
import gym
from multi_object_fetch.env import MultiObjectFetchEnv
from PIL import Image
import numpy as np
import torch
from typing import Tuple


class Pixels(gym.Wrapper):
    def __init__(self, env: gym.Env, image_size: Tuple[int, int] = (64, 64), num_frames: int = 3,) -> None:
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3 * num_frames,) + image_size, dtype=np.uint8)
        self._frames = deque([], maxlen=num_frames)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        _, reward, done, info = self.env.step(action)
        return self._get_obs(), reward, done, info

    def reset(self) -> np.ndarray:
        self.env.reset()
        return self._get_obs(is_reset=True)

    def _get_obs(self, is_reset=False) -> np.ndarray:
        if isinstance(self.env.unwrapped, MultiObjectFetchEnv):
            image = self.env.render(mode='rgb_array', size=self.image_size)
        else:
            image = Image.fromarray(self.env.render(mode='rgb_array'))
            image = np.array(image.resize(self.image_size))
        image = image.transpose(2, 0, 1)

        num_frames = self._frames.maxlen if is_reset else 1
        for _ in range(num_frames):
            self._frames.append(image)

        return torch.from_numpy(np.concatenate(self._frames))
