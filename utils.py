import numpy as np
import torch
import matplotlib.pyplot as plt


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)

class Noise:
    def __init__(self, mean=0.0, std=0.0001, seed=None):
        self.mean = mean
        self.std = std
        if seed is not None:
            np.random.seed(seed)
            
    def add_gaussian_noise(self, value, need_normalize=False):
        noise = np.random.normal(self.mean, self.std, size=value.shape)
        value = value + noise
        if need_normalize:
            return value / np.linalg.norm(value)
        return value
    
class GyroscopeNoise:
	def __init__(self, ARW=0.08, RRW=0.5, head_cnt=3, seed=None):
		self.ARW = ARW  # 角度随机游走，单位：°/√h
		self.RRW = RRW  # 速率随机游走，单位：°/h^(3/2)
		if seed is not None:
			np.random.seed(seed)
		
		self.head_cnt = head_cnt
		self.bg = np.zeros((self.head_cnt, 1))  # 单位：°/h
	
	def add_gyro_noise(self, value, dt):
		if self.head_cnt != value.shape[0]:
			raise ValueError("The number of heads does not match the shape of the input value.")
		
		# 将时间步长转换为小时
		dt_h = dt / 3600.0

		# 生成速率随机游走噪声
		delta_b = self.RRW * np.random.normal(0, 1, size=value.shape) * np.sqrt(dt_h)
		self.bg = self.bg + delta_b  # 单位：°/h

		# 生成角度随机游走噪声
		eta_omega = self.ARW * np.random.normal(0, 1, size=value.shape) * np.sqrt(dt_h)
		eta_v = eta_omega / dt_h  # 单位：°/h

		# 单位转换为rad/s
		bg_rad_s = self.bg * np.pi / 180.0 / 3600.0  # rad/s
		eta_v_rad_s = eta_v * np.pi / 180.0 / 3600.0  # rad/s
        
		# 计算带噪声的测量值
		omega_meas = value + bg_rad_s + eta_v_rad_s  # rad/s
		return omega_meas
	
class QuaternionNoise:
    def __init__(self, mean=0.0, sigma_q=0.0001, seed=None):
        self.mean = mean
        self.sigma_q = sigma_q
        if seed is not None:
            np.random.seed(seed)
    
    def add_quaternion_noise(self, q_true):
        # 确保输入是列向量
        q_true = np.array(q_true).reshape(4, 1)
        
        # 生成噪声向量 v_q (3x1)，并确保平方和小于1
        v_q = np.random.normal(self.mean, self.sigma_q, (3, 1))
        square_sum = np.sum(v_q**2)
        if square_sum >= 1:
            # 如果平方和大于1，对向量进行缩放
            v_q = v_q / np.sqrt(square_sum + 1e-6)  # 添加小量防止除零
            
        # 构造测量噪声向量 q_v (4x1)
        q_v = np.zeros((4, 1))
        q_v[0] = np.sqrt(1 - np.sum(v_q**2))
        q_v[1:] = v_q
        
        # 计算带噪声的测量值 (q_m = q ⊗ q_v)
        q_m = self._quaternion_multiply(q_true, q_v)
        
        return q_m
    
    def _quaternion_multiply(self, q1, q2):
        # 提取标量部分和向量部分
        w1, v1 = q1[0], q1[1:]
        w2, v2 = q2[0], q2[1:]
        
        # 计算结果四元数的标量和向量部分
        w = w1 * w2 - np.dot(v1.T, v2)
        v = w1 * v2 + w2 * v1 + np.cross(v1.T, v2.T).T
        
        return np.vstack((w, v))


class Flywheel:
    def __init__(self, time_constant, initial_torque=0.0, time_step=0.01):
        """
        初始化飞轮类。

        参数:
        - time_constant: 飞轮的时间常数 (τ_w)
        - initial_torque: 飞轮的初始输出力矩 (默认 0.0)
        - time_step: 时间步长 (Δt) (默认 0.01)
        """
        self.tau_w = time_constant  # 时间常数
        self.u_w = initial_torque   # 当前输出力矩
        self.dt = time_step         # 时间步长

    def update(self, input_torque):
        """
        更新飞轮输出力矩。

        参数:
        - input_torque: 输入力矩 (u(t))

        返回:
        - 更新后的输出力矩 (u_w(t))
        """
        # 使用后向欧拉法更新输出力矩
        input_torque = -input_torque
        self.u_w = (self.tau_w * self.u_w - self.dt * input_torque) / (self.tau_w + self.dt)
        return self.u_w

    def reset(self, initial_torque=0.0):
        """
        重置飞轮状态。

        参数:
        - initial_torque: 重置后的初始输出力矩 (默认 0.0)
        """
        self.u_w = initial_torque

# 示例使用
if __name__ == "__main__":
    # 初始化飞轮
    flywheel = Flywheel(time_constant=0.1, time_step=0.01)
    
    # 生成时间序列
    t_end = 10.0  # 10秒模拟时间
    time = np.arange(0, t_end, flywheel.dt)
    
    # 生成高频输入信号（多个不同频率正弦波的叠加）
    input_torques = (
        0.5 * np.sin(2 * np.pi * 2.0 * time) +    # 2 Hz
        0.3 * np.sin(2 * np.pi * 5.0 * time) +    # 5 Hz
        0.2 * np.sin(2 * np.pi * 10.0 * time)     # 10 Hz
    )
    
    output_torques = []
    
    # 动态更新飞轮输出力矩
    for u in input_torques:
        u_w = flywheel.update(u)
        output_torques.append(u_w)
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    
    # 绘制完整时间范围的响应
    plt.subplot(1, 2, 1)
    plt.plot(time, input_torques, label='Input Torque (u(t))', linestyle='--')
    plt.plot(time, output_torques, label='Output Torque (u_w(t))')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Complete Flywheel Dynamic Response')
    plt.legend()
    plt.grid(True)
    
    # 放大显示一小段时间范围
    plt.subplot(1, 2, 2)
    t_zoom = 1.0  # 放大显示1秒的数据
    idx_zoom = int(t_zoom / flywheel.dt)
    plt.plot(time[:idx_zoom], input_torques[:idx_zoom], 
            label='Input Torque (u(t))', linestyle='--')
    plt.plot(time[:idx_zoom], output_torques[:idx_zoom], 
            label='Output Torque (u_w(t))')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.title('Zoomed Flywheel Response (First 1s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    