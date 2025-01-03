import gym
from envs.my_wrappers.action_repeat import ActionRepeat
from envs.my_wrappers.pixels import Pixels
from envs.my_wrappers.time_limit import TimeLimit
from typing import Tuple


def make_env(name: str, image_size: Tuple[int, int], max_episode_steps: int, action_repeat: int, seed: int = 0):
    env = gym.make(name)
    env = ActionRepeat(env, action_repeat)
    env = TimeLimit(env, max_episode_steps)
    env = Pixels(env, image_size)
    return env
