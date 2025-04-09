import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
root_path = os.path.dirname(parent_dir)
sys.path.append(root_path)

from src.satellite import SunPointSatellite
from configs.config import EnvConfig

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
    
