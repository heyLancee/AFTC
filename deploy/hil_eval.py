import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
root_path = os.path.dirname(parent_dir)
sys.path.append(root_path)

import gym
import pandas as pd
import torch
import time
from typing import Optional

from communication.udp_commu import UdpClient
from communication.serial_commu import SerialComm, CallbackEvent
from configs.config import EnvConfig
from src.satellite import *
from pyflywheel import FlyWheel
from pyrealtime import RealTimeSimulation
import src.td3 as td3
from src.dyn_net import AttitudeDynamicsNN, OUTPUT_NUM
from src.base import TelemetryStruct


def hil_eval(agent:Optional[td3.TD3], dynamic_net:AttitudeDynamicsNN, env:SunPointFaultSatellite, seed:int,
             path:Optional[str]=None, is_plot:bool=False, use_flywheel:bool=False, use_serial:bool=False, use_udp:bool=False):
    config = EnvConfig()
        
    eval_env = env
    eval_env.seed(seed)

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

        # print("--------------------------------")
        # print(f"callback function, speed: {speed}")
        # print(f"last speed: {last_speed}")
        # print(f"timestamp: {timestamp}")
        # print(f"last timestamp: {last_timestamp}")
        # print(f"torque_0: {torque_0}")
        # print("--------------------------------")

    # flywheel
    if use_flywheel:
        COM = config.flywheel.COM
        BAUD = config.flywheel.BAUD
        flywheel = FlyWheel(port=COM, baudrate=BAUD, auto_polling=True, polling_frequency=config.flywheel.polling_frequency, communication_frequency=config.flywheel.communication_frequency, 
                            callback=fly_callback, queue_size=10)
        flywheel.connect()
    else:
        flywheel = None
    
    # real-time simulation
    real_time_sim = RealTimeSimulation(eval_env.ts*3)

    # serial
    if use_serial:
        def serial_callback(event:CallbackEvent, data: TelemetryStruct):
            nonlocal action
            if event == CallbackEvent.RECV_TELE_DATA:
                action[0] = data.tx
                action[1] = data.ty
                action[2] = data.tz
                # print(f"tx: {data.tx}, ty: {data.ty}, tz: {data.tz}")
                
        serial_comm = SerialComm(config.serial.COM, config.serial.BAUD, config.serial.timeout, callback=serial_callback)
    else:
        serial_comm = None

    # udp
    if use_udp:
        client = UdpClient(config.udp.host, config.udp.port, local_port=config.udp.local_port, header=config.udp.header, tail=config.udp.tail)
        if not client.connect_to_server():
            print("Failed to connect to server")
            sys.exit(1)
        client.start_receiving(env)
    else:
        client = None

    # 变量初始化
    rewards = []
    states = []
    state, done = eval_env.reset(), False
    state = np.concatenate((state, np.zeros(OUTPUT_NUM)))
    torque_0 = 0

    action = np.zeros((3, 1))

    def simulation_step(current_time):
        nonlocal state, done, flywheel, action, torque_0
        if serial_comm is not None:
            data = TelemetryStruct()
            # 模拟是误差向量
            data.q0 = state[0]
            data.q1 = state[1]
            data.q2 = state[2]
            data.q3 = state[3]
            data.wx = state[4]
            data.wy = state[5]
            data.wz = state[6]
            # 模拟是res
            data.rx = state[7]
            data.ry = state[8]
            data.rz = state[9]
            serial_comm.request(data)
        else:
            agent_output = agent.select_action(np.array(state))
            action = np.diag(agent_output) @ eval_env.u_max

        # 这里只模拟飞轮0
        if flywheel is not None:
            flywheel.set_torque(action[0])
            action[0] = torque_0

        # dynamic net
        net_input = np.concatenate((eval_env.omega.flatten(), (eval_env.C@action).flatten()))
        pred = dynamic_net(torch.tensor(net_input, dtype=torch.float32).unsqueeze(0)).cpu().detach().numpy()

        next_state, reward, done, _ = eval_env.step(action.reshape(-1, 1))
        
        pred_error = eval_env.omega.flatten() - pred.flatten()

        next_state = np.concatenate((next_state.flatten(), pred_error.flatten()))
        state = next_state

        if client is not None:
            client.send_data(eval_env)

        states.append(state)
        rewards.append(reward)

        return done

    # 外设初始化
    if flywheel is not None:
        flywheel.start()
        flywheel.set_speed(200)  # 初始速度拉到200rpm
        time.sleep(1)
    if serial_comm is not None:
        serial_comm.connect()
        serial_comm.start()

    real_time_sim.start(simulation_step)
    time.sleep(1)

    # wait until RealTimeSimulation is stop
    while not real_time_sim.sim_finished:
        time.sleep(1)

    # 外设关闭
    real_time_sim.stop()
    if serial_comm is not None:
        serial_comm.stop()
        serial_comm.disconnect()
    if flywheel is not None:
        flywheel.stop()
    if client is not None:
        client.close()
    
    time.sleep(1)

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
    reward = hil_eval(policy, dynamicNet, env, seed, path=None, is_plot=True, use_flywheel=False, use_serial=True, use_udp=False)
    print("reward: ", reward)

