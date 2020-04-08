from gym import Env
from gym.spaces import Dict, Discrete, Box
import numpy as np

import ray
import ray.rllib.agents.ppo as ppo
from ray import tune
from ray.tune import run as run_tune
from ray.tune.registry import register_env

class DummyEnv(Env):

    def __init__(self):
        super().__init__()
        self.time_counter = 0

    @property
    def observation_space(self):
        temp_box = Box(low=-np.infty, high=np.infty, shape=(10,))
        temp_discrete = Discrete(4)
        Dict1 = Dict({'obs1': temp_box, 'obs2': temp_discrete})
        Dict2 = Dict({'obs3': Dict1})
        return Dict2

    @property
    def action_space(self):
        return Box(low=-1, high=1, shape=(2,))

    def step(self, action):
        self.time_counter += 1
        done = False
        if self.time_counter > 100:
            done = True
        return {'obs3': {'obs1': np.zeros(10), 'obs2': 3}}, 0, done, {}

    def reset(self):
        return {'obs3': {'obs1': np.zeros(10), 'obs2': 3}}

def env_creator(env_config):
    return DummyEnv()

if __name__=='__main__':
    ray.init()
    alg_run = 'PPO'
    config = ppo.DEFAULT_CONFIG.copy()
    register_env('DummyEnv', env_creator)
    config['env'] = 'DummyEnv'
    exp_dict = {
        'name': 'Test',
        'run_or_experiment': alg_run,
        'checkpoint_freq': 1000,
        'stop': {
            'training_iteration': 10000
        },
        'config': config,
    }

    run_tune(**exp_dict, queue_trials=False)
