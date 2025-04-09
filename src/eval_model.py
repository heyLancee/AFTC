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
from src.dyn_net import AttitudeDynamicsNN, OUTPUT_NUM

def eval_policy(agent:Optional[td3.TD3], dynamic_net:AttitudeDynamicsNN, env:Satellite, seed:int,
                path:Optional[str]=None, client:Optional[UdpClient]=None, is_plot:bool=False):
    eval_env = env
    eval_env.seed(seed)

    rewards = []
    states = []
    state, done = eval_env.reset(), False
    state = np.concatenate((state, np.zeros(OUTPUT_NUM)))

    # print("等待故障注入")
    # time.sleep(10)

    while not done:
        if agent is not None:
            agent_action = agent.select_action(np.array(state))
        else:
            agent_action = np.zeros(4)
        action = np.diag(agent_action) @ eval_env.u_max
        
        # dynamic net
        net_input = np.concatenate((eval_env.omega.flatten(), (eval_env.C@action).flatten()))
        pred = dynamic_net(torch.tensor(net_input, dtype=torch.float32).unsqueeze(0)).cpu().detach().numpy()

        next_state, reward, done, _ = eval_env.step(action.reshape(-1, 1))
        
        pred_error = eval_env.omega.flatten() - pred.flatten()

        next_state = np.concatenate((next_state.flatten(), pred_error.flatten()))
        state = next_state

        states.append(state)
        rewards.append(reward)

        if client:
            client.send_data(eval_env)

    states = np.array(states)
    rewards = np.array(rewards)
    angles = np.array(eval_env.theta_buffer)
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
    env_name = "SunPointFaultSatellite"
    dynamic_net_path = "dynamic_net/attitude_dynamics_model.pth"
    hidden_size = [64, 128]
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    policy_model_path = "u_max_008/TD3_SunPointFaultSatellite_0"
    save_path = "results/u_max_008/eval_res.csv"

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

    client = UdpClient(config.udp.host, config.udp.port, local_port=config.udp.local_port, header=config.udp.header, tail=config.udp.tail)
    if not client.connect_to_server():
        print("Failed to connect to server")
        sys.exit(1)
    client.start_receiving(env)

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

    dynamicNet = AttitudeDynamicsNN(hidden_size)
    if dynamic_net_path != "":
        print(f"Load dynamic net from {dynamic_net_path}")
        dynamicNet.load_model(f"{model_path}/{dynamic_net_path}")

    # Evaluate untrained policy
    reward = eval_policy(policy, dynamicNet, env, seed, path=None, client=client, is_plot=True)
    print("reward: ", reward)
    # client.close()
