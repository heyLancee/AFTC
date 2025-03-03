import gym
import pandas as pd
import torch
import time
from dyn_net import AttitudeDynamicsNN
import dyn_net
import td3
from satellite import *
from pyflywheel import FlyWheel
from PyRealTime import RealTimeSimulation
import sys
from udp_client import UdpClient
from base import TelemetryStruct
from config import EnvConfig
from copy import deepcopy


def eval_policy(client, agent, dynamic_net, env_name, seed, path=None, is_plot=False):
    config = EnvConfig()

    if env_name == "Satellite":
        eval_env = Satellite(config)
    elif env_name == "FaultSatellite":
        eval_env = FaultSatellite(config)
    elif env_name == "SunPointSatellite":
        eval_env = SunPointSatellite(config)
    elif env_name == "SunPointFaultSatellite":
        eval_env = SunPointFaultSatellite(config)
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(seed)

    rewards = []
    states = []
    state, done = eval_env.reset(), False
    state = np.concatenate((state, np.zeros(dyn_net.OUTPUT_NUM)))

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

        states.append(deepcopy(state))
        states[-1][:3] = deepcopy(eval_env.omega.flatten())
        rewards.append(reward)

        # send telemetry data
        telemetry_data = TelemetryStruct()
        telemetry_data.timeStep = eval_env.t
        telemetry_data.wx = eval_env.omega[0] * 180 / np.pi
        telemetry_data.wy = eval_env.omega[1] * 180 / np.pi
        telemetry_data.wz = eval_env.omega[2] * 180 / np.pi
        telemetry_data.q0 = eval_env.q[0]
        telemetry_data.q1 = eval_env.q[1]
        telemetry_data.q2 = eval_env.q[2]
        telemetry_data.q3 = eval_env.q[3]
        telemetry_data.zAngle = eval_env.theta_buffer[-1]
        client.send_data(telemetry_data)

    states = np.array(states)
    rewards = np.array(rewards)
    angles = np.array(eval_env.theta_buffer)
    actions = np.array(eval_env.u_buffer)

    if path is not None:
        df = pd.DataFrame(states, columns=[f'state_{i}' for i in range(len(states[0]))])
        df_uc = pd.DataFrame(actions, columns=[f'u_{i}' for i in range(len(actions[0]))])
        df = pd.concat([df, df_uc], axis=1)
        df["angle"] = angles
        df['reward'] = rewards
        df.to_csv(path, index=False)

    if is_plot:
        eval_env.plot()


def eval_policy_with_flywheel(agent, dynamic_net, env_name, fault_mode, seed, path=None, is_plot=False):
    config = EnvConfig()

    if env_name == "Satellite":
        eval_env = Satellite(config)
    elif env_name == "FaultSatellite":
        eval_env = FaultSatellite(config)
    elif env_name == "SunPointSatellite":
        eval_env = SunPointSatellite(config)
    elif env_name == "SunPointFaultSatellite":
        eval_env = SunPointFaultSatellite(config)
    else:
        eval_env = gym.make(env_name)
    eval_env.seed(seed)

    rewards = []
    states = []
    state, done = eval_env.reset(), False
    if fault_mode != -1:
        eval_env.fault_mode = fault_mode
    state = np.concatenate((state, np.zeros(dyn_net.OUTPUT_NUM)))
    torque_0 = 0

    def fly_callback(telemetry, last_telemetry):
        # 修改力矩值
        nonlocal torque_0, flywheel
        # 差分计算torque
        speed = telemetry['flywheel_speed_feedback']
        last_speed = last_telemetry['flywheel_speed_feedback']
        timestamp = telemetry['timestamp']
        last_timestamp = last_telemetry['timestamp']

        if timestamp - last_timestamp > 0:
            torque_0 = (speed - last_speed) / (timestamp - last_timestamp) * flywheel.inertia

        print("--------------------------------")
        print(f"callback function, speed: {speed}")
        print(f"last speed: {last_speed}")
        print(f"timestamp: {timestamp}")
        print(f"last timestamp: {last_timestamp}")
        print(f"torque_0: {torque_0}")
        print("--------------------------------")

    # flywheel
    COM = config.flywheel.COM
    BAUD = config.flywheel.BAUD
    flywheel = FlyWheel(port=COM, baudrate=BAUD, auto_polling=True, polling_frequency=config.flywheel.polling_frequency, communication_frequency=config.flywheel.communication_frequency, 
                        callback=fly_callback, queue_size=10)
    flywheel.connect()
    real_time_sim = RealTimeSimulation(eval_env.ts)

    time.sleep(1)
    flywheel.start()
    flywheel.set_speed(200)  # 200转初速度
    time.sleep(1)

    def simulation_step(current_time):
        nonlocal state, done, flywheel
        if agent is not None:
            agent_action = agent.select_action(np.array(state))
        else:
            agent_action = np.zeros(4)
        action = np.diag(agent_action) @ eval_env.u_max
        
        # 这里只模拟飞轮0
        flywheel.set_torque(action[0])
        action[0] = torque_0

        # dynamic net
        net_input = np.concatenate((eval_env.omega.flatten(), (eval_env.C@action).flatten()))
        pred = dynamic_net(torch.tensor(net_input, dtype=torch.float32).unsqueeze(0)).cpu().detach().numpy()

        next_state, reward, done, _ = eval_env.step(action.reshape(-1, 1))
        
        pred_error = eval_env.omega.flatten() - pred.flatten()

        next_state = np.concatenate((next_state.flatten(), pred_error.flatten()))
        state = next_state

        states.append(state)
        rewards.append(reward)

        return done

    real_time_sim.start(simulation_step)

    # wait until RealTimeSimulation is stop
    while real_time_sim.is_running:
        time.sleep(0.1)

    real_time_sim.stop()
    time.sleep(1)

    # 在循环结束后转换为NumPy数组
    states = np.array(states)
    rewards = np.array(rewards)
    actions = np.array(eval_env.u_buffer)

    if path is not None:
        df = pd.DataFrame(states, columns=[f'state_{i}' for i in range(len(states[0]))])
        df_uc = pd.DataFrame(actions, columns=[f'u_{i}' for i in range(len(actions[0]))])
        df = pd.concat([df, df_uc], axis=1)
        df['reward'] = rewards
        df.to_csv(path, index=False)

    if is_plot:
        eval_env.plot()


if __name__ == "__main__":
    policy = "TD3"
    # seed = np.random.randint(1, 100)
    seed = 1
    env_name = "SunPointFaultSatellite"
    dynamic_net_path = "models/dynamic_net/attitude_dynamics_model.pth"
    hidden_size = [64, 128]
    discount = 0.99
    tau = 0.005
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    policy_model_path = "u_max_005\TD3_SunPointFaultSatellite_1"
    save_path = "results/u_max_005/eval_res.csv"

    config = EnvConfig()

    client = UdpClient(config.udp.host, config.udp.port, local_port=config.udp.local_port, header=config.udp.header, tail=config.udp.tail)
    if not client.connect_to_server():
        print("Failed to connect to server")
        sys.exit(1)

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

    # Set seeds
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

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
        "hidden_size": [256, 256],
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

    if policy_model_path != "":
        policy.load(f"./models/{policy_model_path}")

    dynamicNet = dyn_net.AttitudeDynamicsNN(hidden_size)
    if dynamic_net_path != "":
        print(f"Load dynamic net from {dynamic_net_path}")
        dynamicNet.load_model(dynamic_net_path)

    # Evaluate untrained policy
    eval_policy(client, policy, dynamicNet, env_name, seed, save_path, is_plot=True)

    client.close()
