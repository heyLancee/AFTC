import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
root_path = os.path.dirname(parent_dir)
sys.path.append(root_path)

import gym
import pandas as pd
import torch
from typing import Optional

from communication.udp_commu import UdpClient
from configs.config import EnvConfig
from src.satellite import *
import src.td3 as td3
from src.pid import *

def eval_policy(agent:Optional[td3.TD3], env:Satellite, seed:int, pid:PController,
                path:Optional[str]=None, client:Optional[UdpClient]=None, is_plot:bool=False):
    eval_env = env
    eval_env.seed(seed)

    rewards = []
    states = []
    state, done = eval_env.reset(), False
    state = np.concatenate((state, np.zeros(td3.STATE_APPEND_NUM)))

    # print("等待故障注入")
    # time.sleep(10)

    while not done:
        if agent is not None:
            agent_action = agent.select_action(np.array(state))
        else:
            agent_action = np.zeros(4)
            
        pid.update(*agent_action)
        qev = state[1:4]
        omegae = state[4:7]
        torque = pid.compute(qev, omegae)

        next_state, reward, done, _ = eval_env.step(torque.reshape(-1, 1))
        state = next_state

        states.append(state)
        rewards.append(reward)

        if client:
            client.send_data(eval_env)

    states = np.array(states)
    rewards = np.array(rewards)
    if hasattr(eval_env, 'theta_buffer'):
        angles = np.array(eval_env.theta_buffer)
    else:
        angles = np.zeros_like(rewards)
    actions = np.array(eval_env.u_buffer)

    min_length = min(len(states), len(angles), len(actions), len(rewards))
    states = states[:min_length]
    angles = angles[:min_length]
    actions = actions[:min_length]
    rewards = rewards[:min_length]

    if path:
        df = pd.DataFrame(states, columns=[f'state_{i}' for i in range(len(states[0]))])
        df_uc = pd.DataFrame(actions, columns=[f'u_{i}' for i in range(len(actions[0]))])
        df = pd.concat([df, df_uc], axis=1)
        df["angle"] = angles
        df['reward'] = rewards
        df.to_csv(path, index=False)

    if is_plot:
        eval_env.plot()

    return np.sum(rewards)


if __name__ == "__main__":
    policy = "TD3"
    # seed = np.random.randint(1, 100)
    seed = 1
    env_name = "Satellite"
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    policy_model_path = ""

    config = EnvConfig()
    if env_name == "Satellite":
        env = Satellite(config)
    elif env_name == "FaultSatellite":
        env = FaultSatellite(config)
    elif env_name == "SunPointSatellite":
        env = SunPointSatellite(config)
    elif env_name == "SunPointFaultSatellite":
        env = SunPointFaultSatellite(config)
    else:
        env = gym.make(env_name)

    # client = UdpClient(config.udp.host, config.udp.port, local_port=config.udp.local_port, header=config.udp.header, tail=config.udp.tail)
    # if not client.connect_to_server():
    #     print("Failed to connect to server")
    #     sys.exit(1)
    # client.start_receiving(env)

    state_dim = env.observation_space.shape[0]
    state_dim += td3.STATE_APPEND_NUM
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
        "hidden_size": [400, 300],
    }

    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        policy = td3.TD3(**kwargs)
    else:
        raise NotImplementedError

    model_path = os.path.join(root_path, "models")
    if policy_model_path != "":
        policy.load(f"{model_path}/{policy_model_path}")

    # pid
    pid = PDController(dt=env.ts)

    # Evaluate untrained policy
    reward = eval_policy(policy, env, seed, pid, path=None, client=None, is_plot=True)
    print("reward: ", reward)
    # client.close()
