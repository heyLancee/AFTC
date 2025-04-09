import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
root_path = os.path.dirname(parent_dir)
sys.path.append(root_path)

import random
import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import Union, Tuple, Optional, List

from src.satellite_func import *
from src.util_classes import Noise, GyroscopeNoise, QuaternionNoise, Flywheel
from configs.config import EnvConfig, OrbitConfig
from src.base import FaultParams, ComponentFaultType


class Orbit:
    def __init__(self, ts, t_max, config: OrbitConfig):
        # 轨道常数
        self.mu = config.mu  # 地球引力常数
        self.a = config.a  # 轨道半长轴(km)
        self.omega = config.omega * np.pi/180  # 近地点幅角
        self.Omega = config.Omega * np.pi/180  # 升交点赤经
        self.incline = config.incline * np.pi/180  # 轨道倾角
        self.f = config.f * np.pi/180  # 真近点角
        self.e = config.e  # 轨道离心率
        
        # 地磁场常数
        self.mum = config.mum  # 地球磁场中的偶极子强度(Wb·km)
        self.thetam = config.thetam * np.pi/180  # 偶极子的共生角度
        self.we = config.we * np.pi/180/24/3600  # 地球的平均自转角速度
        self.alpha0 = config.alpha0  # t=0时偶极子的赤经
        
        # 时间参数
        self.ts = ts
        self.Nk = int(t_max/ts)
        
        # 初始化状态向量
        self.R = np.zeros((3, self.Nk))  # 位置向量
        self.V = np.zeros((3, self.Nk))  # 速度向量
        self.Bi = np.zeros((3, self.Nk))  # 地磁场向量
        
        # 设置初始状态
        init_pos = np.array(config.position) * 1e-3
        init_vel = np.array(config.velocity) * 1e-3
        self.R[:,0] = init_pos
        self.V[:,0] = init_vel
        
        # 计算初始地磁场
        self._update_magnetic_field(0)
        
        # 计算整个轨道
        self._propagate_orbit()
    
    def _update_magnetic_field(self, k):
        """更新地磁场向量"""
        hat_R = self.R[:,k] / np.linalg.norm(self.R[:,k])
        hat_p = np.array([
            np.sin(self.thetam) * np.cos(self.we * k * self.ts + self.alpha0),
            np.sin(self.thetam) * np.sin(self.we * k * self.ts + self.alpha0),
            np.cos(self.thetam)
        ])
        
        R_norm = np.linalg.norm(self.R[:,k])
        self.Bi[:,k] = (self.mum/R_norm**3) * (
            3 * np.dot(hat_p, hat_R) * hat_R - hat_p
        )
    
    def _propagate_orbit(self):
        """传播整个轨道"""
        for k in range(1, self.Nk):
            # 二体运动方程
            R_norm = np.linalg.norm(self.R[:,k-1])
            dotV = -self.mu * self.R[:,k-1] / R_norm**3
            
            # 更新状态
            self.V[:,k] = self.V[:,k-1] + dotV * self.ts
            self.R[:,k] = self.R[:,k-1] + self.V[:,k] * self.ts
            
            # 更新地磁场
            self._update_magnetic_field(k)
    
    def get_magnetic_field(self, t):
        """获取特定时刻的地磁场向量"""
        k = int(t/self.ts)
        if k >= self.Nk:
            k = self.Nk - 1
        return self.Bi[:,k]
    
    def get_position(self, t):
        """获取特定时刻的位置向量"""
        k = int(t/self.ts)
        if k >= self.Nk:
            k = self.Nk - 1
        return self.R[:,k]
    

class Satellite:
    def __init__(self, config: EnvConfig):
        if hasattr(self, 'is_init') and self.is_init:
            self.logger.info("卫星环境已经初始化，无需再次初始化")
            return

        self.is_init = True
        self.ts = config.simulation.ts
        self.t = 0
        self.t_max = config.simulation.t_max
        self._max_episode_steps = int(self.t_max / self.ts)

        self.j = config.satellite.inertia
        self.j_inv = np.linalg.inv(self.j)
        self.delta_j = config.satellite.delta_inertia
        self.C = config.satellite.actuator.installation_matrix

        self.u_max = config.satellite.actuator.u_max

        self.omega_buffer = []
        self.q_buffer = []
        self.torque_buffer = []
        self.u_buffer = []
        self.qe_buffer = []
        self.omega_e_buffer = []
        self.state = None
        self.q = np.zeros((4, 1))
        self.q_m = np.zeros((4, 1))  # 四元数测量值（噪声影响）
        self.omega = np.zeros((3, 1))
        self.omega_m = np.zeros((3, 1))  # 角速度测量值（噪声、故障影响）
        self.qd = np.zeros((4, 1))
        
        self.orbit = Orbit(ts=self.ts, t_max=self.t_max, config=config.orbit)
        self.td = np.zeros((3, 1))
       
        obs = np.array(config.satellite_observation_space.upper_bound, dtype=np.float32)
        action = np.array(config.satellite_action_space.upper_bound, dtype=np.float32)
        self.action_space = spaces.Box(-action, action, dtype=np.float32)
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)

        self.q_noise = QuaternionNoise(mean=config.sensor_noise.quaternion.noise_mean, sigma_q=config.sensor_noise.quaternion.noise_std)
        self.gyro_noise = GyroscopeNoise(ARW=config.sensor_noise.gyroscope.ARW, RRW=config.sensor_noise.gyroscope.RRW, head_cnt=3)
        self.s_noise = Noise(mean=config.sensor_noise.sun_sensor.noise_mean, std=config.sensor_noise.sun_sensor.noise_std)

        # 飞轮组
        self.flywheel_group = []
        for _ in range(self.C.shape[1]):
            self.flywheel_group.append(Flywheel(time_constant=config.flywheel.time_constant, time_step=self.ts))

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def step(self, torque):
        torque = torque.reshape(-1, 1)
        # clip
        torque = np.clip(torque.flatten(), -self.u_max, self.u_max)
        for i in range(self.C.shape[1]):
            torque[i] = self.flywheel_group[i].update(torque[i])

        Bi = self.orbit.get_magnetic_field(self.t)*1e-6
        q_correct = np.array([self.q[1], self.q[2], self.q[3], self.q[0]])
        R = Rotation.from_quat(q_correct.flatten()).as_matrix().T
        Bb = R @ Bi
        
        # 计算重力梯度力矩
        w0 = 0.001  # 轨道角速度
        E_r = R @ np.array([0, 0, -1]).reshape(-1,1)  # 地心指向矢量在体坐标系下的表示
        E_r = E_r / np.linalg.norm(E_r)
        Tg = 3 * w0**2 * np.cross(E_r.flatten(), (self.j @ E_r).flatten())
        
        # 计算磁力矩
        M_res = np.array([6, -6, 6]).reshape(-1,1)  # 剩余磁矩,单位:A·m^2
        Tm = np.cross(M_res.flatten(), Bb.flatten())
        
        # 合并环境力矩
        self.td = Tg.reshape(-1,1) + Tm.reshape(-1,1)

        u = (self.C @ torque).reshape(-1, 1)
        u = u + self.td.reshape(-1, 1)

        self.q, self.omega = R_K(self.q, self.omega, self.ts, self.j_inv, self.j+self.delta_j, u)

        self.q_m = self.q_noise.add_quaternion_noise(self.q)
        self.omega_m = self.gyro_noise.add_gyro_noise(self.omega, dt=self.ts)

        omega_d = get_omega_d(self.t)
        qe = get_q_e(self.qd, self.q_m)
        qev = qe[1:]
        omega_e = get_omega_e(self.omega_m, omega_d)
        self.state = np.concatenate([qe, omega_e], axis=0).flatten()

        self.omega_buffer.append(self.omega_m.flatten())
        self.q_buffer.append(self.q_m.flatten())
        self.torque_buffer.append(torque.flatten())
        self.u_buffer.append(u.flatten())
        self.qe_buffer.append(qe.flatten())
        self.omega_e_buffer.append(omega_e.flatten())
        reward = Satellite.reward(self, torque, qev, omega_e)

        self.t += self.ts
        done = False
        if self.t >= self.t_max:
            done = True
        return self.state, reward, done, {}

    def reward(self, f, qev, omega_e):
        reward_1 = 0
        reward_2 = -4 * np.linalg.norm(f)
        reward_3 = -20 * np.linalg.norm(qev)
        reward_4 = -10 * np.linalg.norm(omega_e)
        reward = reward_1 + reward_2 + reward_3 + reward_4
        return reward

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return seed
    
    def plot(self, size: Tuple[int, int]=(6, 4)):
        qe_buffer = np.array(self.qe_buffer)
        qe_buffer = qe_buffer[:-1, :]
        omega_e_buffer = np.array(self.omega_e_buffer) * 180 / np.pi
        omega_e_buffer = omega_e_buffer[:-1, :]
        torque_buffer = np.array(self.torque_buffer)
        u_buffer = np.array(self.u_buffer)
        q_buffer = np.array(self.q_buffer)
        q_buffer = q_buffer[:-1, :]
        omega_buffer = np.array(self.omega_buffer) * 180 / np.pi
        omega_buffer = omega_buffer[:-1, :]
        times = np.linspace(0, self.t_max, len(qe_buffer))

        # qe_buffer
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.plot(times, qe_buffer[:, 0], label='qe0')
        ax.plot(times, qe_buffer[:, 1], label='qe1')
        ax.plot(times, qe_buffer[:, 2], label='qe2')
        ax.plot(times, qe_buffer[:, 3], label='qe3')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Quaternion') 

        # omega_e_buffer
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.plot(times, omega_e_buffer[:, 0], label='omega_e0')
        ax.plot(times, omega_e_buffer[:, 1], label='omega_e1')
        ax.plot(times, omega_e_buffer[:, 2], label='omega_e2')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Omega_e')

        # torque_buffer
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.plot(times, torque_buffer[:, 0], label='u0')
        ax.plot(times, torque_buffer[:, 1], label='u1')
        ax.plot(times, torque_buffer[:, 2], label='u2')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Torque')

        # u_buffer
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.plot(times, u_buffer[:, 0], label='u0')
        ax.plot(times, u_buffer[:, 1], label='u1')
        ax.plot(times, u_buffer[:, 2], label='u2')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Torque')

        # q_buffer
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.plot(times, q_buffer[:, 0], label='q0')
        ax.plot(times, q_buffer[:, 1], label='q1')
        ax.plot(times, q_buffer[:, 2], label='q2')
        ax.plot(times, q_buffer[:, 3], label='q3')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Quaternion')
        
        # omega_buffer
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.plot(times, omega_buffer[:, 0], label='omega0')
        ax.plot(times, omega_buffer[:, 1], label='omega1')
        ax.plot(times, omega_buffer[:, 2], label='omega2')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Omega')
     
        plt.show()

    def reset(self):
        self.t = 0

        for flywheel in self.flywheel_group:
            flywheel.reset()

        self.gyro_noise.reset()

        self.qd = np.random.random((4, 1))
        self.qd = self.qd / np.linalg.norm(self.qd)
        self.q = np.random.random((4, 1))
        self.q = self.q / np.linalg.norm(self.q)
        self.q_m = self.q_noise.add_quaternion_noise(self.q)
        qe = get_q_e(self.qd, self.q_m)
        self.omega = (2 * np.random.random((3, 1)) - 1) * 0.1
        self.omega_m = self.gyro_noise.add_gyro_noise(self.omega, dt=self.ts)
        omega_d = get_omega_d(self.t)
        omega_e = get_omega_e(self.omega_m, omega_d)
        self.state = np.concatenate([qe, omega_e], axis=0).flatten()

        self.td = np.zeros((3, 1))
        self.t = 0
        self.q_buffer = [self.q_m.flatten()]
        self.omega_buffer = [self.omega_m.flatten()]
        self.torque_buffer = []
        self.u_buffer = []
        self.qe_buffer = [qe.flatten()]
        self.omega_e_buffer = [omega_e.flatten()]

        self.logger.info("quat init: %s", self.q)
        self.logger.info("omega init: %s", self.omega)
        self.logger.info("q desired init: %s", self.qd)

        return self.state
    

class FaultSatellite(Satellite):
    def __init__(self, config: EnvConfig):
        super().__init__(config)

        self.uf_buffer = []  # 单机故障力矩
        
        # 执行器故障相关
        self.fault_mode:FaultParams.FaultType = config.simulation.fault_mode
        self.fault_params: List[float] = config.simulation.fault_params
        self.fault_start_time: float = config.simulation.fault_start_time
        self.fault_end_time: float = config.simulation.fault_end_time
        self.flywheel_fault_idx: int = config.simulation.flywheel_fault_idx
        self.gyro_fault_idx: int = config.simulation.gyro_fault_idx

    # fault_data是向外部提供的故障配置参数
    def update_u_f(self, torque, fault_data:Optional[FaultParams]=None):
        if fault_data is None:
            fault_data = FaultParams(self.fault_mode, self.fault_params)
            fault_data.fault_start_time = self.fault_start_time
            fault_data.fault_end_time = self.fault_end_time
            fault_data.flywheel_fault_idx = self.flywheel_fault_idx
        else:
            self.fault_mode = fault_data.fault_type
            self.fault_params = fault_data.params
            self.fault_start_time = fault_data.fault_start_time
            self.fault_end_time = fault_data.fault_end_time
            self.flywheel_fault_idx = fault_data.flywheel_fault_idx

        # 如果故障类型不是飞轮相关，则返回
        u, u_f = self.fault_inject(self.t, torque, ComponentFaultType.FLYWHEEL, fault_data)
        u = np.clip(u.flatten(), -self.u_max, self.u_max)

        u = u.flatten()
        u_f = u_f.flatten()
        u_f = np.where(u_f != 0, u - torque, 0).reshape(-1, 1)
        self.uf_buffer.append(u_f.flatten())
        return u
    
    def update_omega_f(self, omega, fault_data:Optional[FaultParams]=None):
        if fault_data is None:
            fault_data = FaultParams(self.fault_mode, self.fault_params)
            fault_data.fault_start_time = self.fault_start_time
            fault_data.fault_end_time = self.fault_end_time
            fault_data.gyro_fault_idx = self.gyro_fault_idx
        else:
            self.fault_mode = fault_data.fault_type
            self.fault_params = fault_data.params
            self.fault_start_time = fault_data.fault_start_time
            self.fault_end_time = fault_data.fault_end_time
            self.gyro_fault_idx = fault_data.gyro_fault_idx
            
        omega, _ = self.fault_inject(self.t, omega, ComponentFaultType.GYROSCOPES, fault_data)
        return omega
    
    def step_fault_satellite(self):
        self.omega_m = self.update_omega_f(self.omega)
        omega_d = get_omega_d(self.t)
        qe = get_q_e(self.qd, self.q_m)
        omega_e = get_omega_e(self.omega_m, omega_d)
        self.omega_e_buffer[-1] = omega_e.flatten()
        self.state = np.concatenate([qe, omega_e], axis=0).flatten()
        return self.state

    def step(self, torque):
        torque = torque.reshape(-1, 1)
        u = self.update_u_f(torque)
        _, reward, done, info = Satellite.step(self, u)
        state = self.step_fault_satellite()

        return state, reward, done, info

    def fault_inject(self, t, data:np.ndarray, fault_conponent:ComponentFaultType, 
                     fault_data:FaultParams) -> Union[None, Tuple[np.ndarray, np.ndarray]]:
        if fault_data.fault_type == FaultParams.FaultType.NO_FAULT:
            return (data, np.zeros((3, 1)))
        if t < fault_data.fault_start_time:
            return (data, np.zeros((3, 1)))
        if (
            fault_data.fault_type == FaultParams.FaultType.FLYWHEEL_PARTIAL_LOSS or 
            fault_data.fault_type == FaultParams.FaultType.FLYWHEEL_BIAS or
            fault_data.fault_type == FaultParams.FaultType.FLYWHEEL_COMPREHENSIVE
        ):
            if fault_conponent != ComponentFaultType.FLYWHEEL:
                return (data, np.zeros((3, 1)))
            if t > fault_data.fault_end_time:
                return (data, np.zeros((3, 1)))
            e1 = e2 = e3 = 0
            b1 = b2 = b3 = 0
            if fault_data.flywheel_fault_idx == 1:
                e1 = fault_data.e
                b1 = fault_data.b
            elif fault_data.flywheel_fault_idx == 2:
                e2 = fault_data.e
                b2 = fault_data.b
            elif fault_data.flywheel_fault_idx == 3:
                e3 = fault_data.e
                b3 = fault_data.b
            E = np.diag([e1, e2, e3])
            B = np.array([b1, b2, b3])

            u_f = - E @ data + B.reshape(-1, 1)
            u = data + u_f
            return (u, u_f)
        elif (
            fault_data.fault_type == FaultParams.FaultType.GYRO_SLOW_FAULT or
            fault_data.fault_type == FaultParams.FaultType.GYRO_INTERMITTENT_FAULT or
            fault_data.fault_type == FaultParams.FaultType.GYRO_MULTI_FAULT
        ):
            if fault_conponent!= ComponentFaultType.GYROSCOPES:
                return (data, np.zeros((3, 1)))
            if fault_data.fault_type == FaultParams.FaultType.GYRO_INTERMITTENT_FAULT:
                if t > fault_data.fault_end_time:
                    return (data, np.zeros((3, 1)))
                
                f1 = fault_data.f1
                if fault_data.gyro_fault_idx == 1:
                    omega_f = np.array([[f1], [0], [0]])
                elif fault_data.gyro_fault_idx == 2:
                    omega_f = np.array([[0], [f1], [0]])
                elif fault_data.gyro_fault_idx == 3:
                    omega_f = np.array([[0], [0], [f1]])
                else:
                    omega_f = np.zeros((3, 1))
                omega = data + omega_f
                return (omega, omega_f)
            elif fault_data.fault_type == FaultParams.FaultType.GYRO_SLOW_FAULT:
                if t > fault_data.fault_end_time:
                    f2 = fault_data.k_s*(fault_data.fault_end_time-fault_data.fault_start_time)
                else:
                    f2 = fault_data.lambda_s*fault_data.k_s*(t-fault_data.fault_start_time)
                if fault_data.gyro_fault_idx == 1:
                    omega_f = np.array([[f2], [0], [0]])
                elif fault_data.gyro_fault_idx == 2:
                    omega_f = np.array([[0], [f2], [0]])
                elif fault_data.gyro_fault_idx == 3:
                    omega_f = np.array([[0], [0], [f2]])
                else:
                    omega_f = np.zeros((3, 1))
                omega = data + omega_f
                return (omega, omega_f)
            elif fault_data.fault_type == FaultParams.FaultType.GYRO_MULTI_FAULT:
                if t > fault_data.fault_end_time:
                    return (data, np.zeros((3, 1)))
                if fault_data.gyro_fault_idx == 1:
                    omega_f = np.diag([fault_data.lambda_m, 1, 1]) @ data
                elif fault_data.gyro_fault_idx == 2:
                    omega_f = np.diag([1, fault_data.lambda_m, 1]) @ data
                elif fault_data.gyro_fault_idx == 3:
                    omega_f = np.diag([1, 1, fault_data.lambda_m]) @ data
                else:
                    omega_f = np.diag([1, 1, 1]) @ data
                omega = omega_f
                omega_f = omega_f - data
                return (omega, omega_f)
            
        return (data, np.zeros((3, 1)))

    def plot_fault_satellite(self, size: Tuple[int, int]=(6, 4)):
        times = np.linspace(0, self.t_max, len(self.uf_buffer))
        uf_buffer = np.array(self.uf_buffer)

        # 绘制se
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.plot(times, uf_buffer[:, 0], label='uf0')
        ax.plot(times, uf_buffer[:, 1], label='uf1')
        ax.plot(times, uf_buffer[:, 2], label='uf2')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('fault torque')

    def plot(self, size: Tuple[int, int]=(6, 4)):
        self.plot_fault_satellite(size)
        return Satellite.plot(self, size)

    def reset_fault_satellite(self):
        self.uf_buffer = []
        # self.fault_mode = np.random.randint(0, 3)
        self.logger.info("fault mode: %s", self.fault_mode)

    def reset(self):
        self.reset_fault_satellite()
        return Satellite.reset(self)


class SunPointSatellite(Satellite):
    def __init__(self, config: EnvConfig):
        super().__init__(config)

        obs = np.array(config.sun_pointing_observation_space.upper_bound, dtype=np.float32)
        action = np.array(config.sun_pointing_action_space.upper_bound, dtype=np.float32)
        self.action_space = spaces.Box(-action, action, dtype=np.float32)
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)

        self.sd = config.sun_pointing.desired_vector.reshape(-1, 1)
        self.sd = self.sd / np.linalg.norm(self.sd)
        self.si = config.orbit.sun_vector.reshape(-1, 1)
        self.si = self.si / np.linalg.norm(self.si)
        self.sb = np.zeros((3, 1))
        self.se = np.zeros((3, 1))
        self.theta = 0

        self.rd_sun_point, self.qd_sunpoint = self.compute_qd_in_sun_point(self.si, self.sd)

        self.theta_buffer = []
        self.se_buffer = []
    
    def update_se(self):
        q_correct = np.array([self.q[1], self.q[2], self.q[3], self.q[0]])
        R = Rotation.from_quat(q_correct.flatten()).as_matrix().T
        self.sb = R @ self.si
        # sb也加个噪声
        # self.sb = self.s_noise.add_gaussian_noise(self.sb, need_normalize=True)
        self.se = np.cross(self.sb.flatten(), self.sd.flatten())
        self.theta = np.arccos(np.dot(self.sb.flatten(), self.sd.flatten()) / (np.linalg.norm(self.sb) * np.linalg.norm(self.sd)))
        self.theta_buffer.append(self.theta*180/np.pi)
        self.se_buffer.append(self.se)

    def compute_qd_in_sun_point(self, si, sd):
        # sd[:2] = sd[:2] + np.random.normal(0, 0.01, (2, 1))
        # 归一化 si 和 sd
        si_norm = si / np.linalg.norm(si)
        sd_norm = sd / np.linalg.norm(sd)

        # 计算旋转轴 (si 和 sd 的叉积)
        axis = np.cross(si_norm.flatten(), sd_norm.flatten())
        axis = axis / np.linalg.norm(axis)  # 归一化旋转轴

        # 计算旋转角度 (si 和 sd 的点积)
        cos_theta = np.dot(si_norm.flatten(), sd_norm.flatten())
        theta = np.arccos(cos_theta)  # 计算旋转角度

        # 使用 Rodrigues' 公式计算旋转矩阵 R
        I = np.eye(3)  # 单位矩阵
        K = cross_matrix(axis)  # 旋转轴的反对称矩阵

        # Rodrigues' 公式：R = I + sin(theta) * K + (1 - cos(theta)) * K^2
        Rd = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

        # R 转 单位四元数
        qd_correct = Rotation.from_matrix(Rd.T).as_quat()
        qd = np.array([qd_correct[3], qd_correct[0], qd_correct[1], qd_correct[2]]).reshape(-1, 1)
        qd = qd / np.linalg.norm(qd)

        return Rd, qd

    def step_sun_point_satellite(self):
        self.update_se()
        qse = get_q_e(self.qd_sunpoint, self.q)
        omega_d = get_omega_d(self.t)
        omega_e = get_omega_e(self.omega_m, omega_d)
        self.state = np.concatenate([qse.flatten(), omega_e.flatten()], axis=0).flatten()
        return self.state

    def step(self, torque):
        torque = torque.reshape(-1, 1)
        _, _, done, info = Satellite.step(self, torque)
        state = self.step_sun_point_satellite()
        qev = self.state[1:4]
        omegae = self.state[4:7]
        reward = self.reward(torque, qev, omegae)
        return state, reward, done, info

    def reward(self, torque, qev, omegae):
        reward = Satellite.reward(self, torque, qev, omegae)
        return reward

    def reset_sun_point_satellite(self):
        self.theta_buffer = []
        self.se_buffer = []
        self.update_se()
        qse = get_q_e(self.qd_sunpoint, self.q_m)
        omega_d = get_omega_d(self.t)
        omega_e = get_omega_e(self.omega_m, omega_d)
        self.state = np.concatenate([qse.flatten(), omega_e.flatten()], axis=0).flatten()
        return self.state

    def reset(self):
        Satellite.reset(self)
        state = self.reset_sun_point_satellite()
        return state
    
    def plot_sun_point_satellite(self, size: Tuple[int, int]=(6, 4)):
        times = np.linspace(0, self.t_max, len(self.theta_buffer))
        theta_buffer = np.array(self.theta_buffer)

        # 绘制theta
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111)
        ax.plot(times, theta_buffer, label='theta')
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Theta')

    def plot(self, size: Tuple[int, int]=(6, 4)):
        self.plot_sun_point_satellite(size)
        return Satellite.plot(self, size)
        

class SunPointFaultSatellite(FaultSatellite, SunPointSatellite):
    def __init__(self, config: EnvConfig):
        super().__init__(config)
        SunPointSatellite.__init__(self, config)

        obs = np.array(config.sun_pointing_observation_space.upper_bound, dtype=np.float32)
        action = np.array(config.sun_pointing_action_space.upper_bound, dtype=np.float32)
        self.action_space = spaces.Box(-action, action, dtype=np.float32)
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)

    def step(self, torque):
        torque = torque.reshape(-1, 1)
        u = FaultSatellite.update_u_f(self, torque)
        _, reward, done, info = Satellite.step(self, u)
        state = FaultSatellite.step_fault_satellite(self)
        state = SunPointSatellite.step_sun_point_satellite(self)
        qev = self.state[1:4]
        omegae = self.state[4:7]
        reward = self.reward(torque, qev, omegae)
        return state, reward, done, info

    def reward(self, torque, qev, omegae):
        reward = Satellite.reward(self, torque, qev, omegae)
        return reward
    
    def reset(self):
        Satellite.reset(self)
        FaultSatellite.reset_fault_satellite(self)
        self.state = SunPointSatellite.reset_sun_point_satellite(self)
        return self.state
    
    def plot(self, size: Tuple[int, int]=(6, 4)):
        self.plot_fault_satellite(size)
        self.plot_sun_point_satellite(size)
        return Satellite.plot(self, size)
