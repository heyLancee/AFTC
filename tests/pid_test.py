import os
import sys
import numpy as np
import torch

module_path = os.path.abspath('../')
if module_path not in sys.path:
    sys.path.append(module_path)

from satellite import SunPointSatellite
from config import EnvConfig

def eval_pd(kp, kd, env, seed, path=None, is_plot=False):
    state, done = env.reset(), False

    while not done:
        omega = env.omega.flatten()
        q = env.q.flatten()
        qv = q[1:]
        sb = env.sb.flatten()
        sd = env.sd.flatten()
        se = env.se.flatten()

        action = -kp * se.flatten() - kd * omega.flatten()
        # action = -kp * qv - kd * omega

        next_state, reward, done, _ = env.step(action.reshape(-1, 1))
        state = next_state

    if is_plot:
        env.plot()


if __name__ == '__main__':
    config = EnvConfig()
    env = SunPointSatellite(config)
    eval_pd(kp=0.2, kd=1, env=env, seed=0, is_plot=True)
    



