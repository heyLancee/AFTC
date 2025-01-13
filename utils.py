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
	