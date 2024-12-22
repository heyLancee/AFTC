import numpy as np
from numpy import rad2deg

from satellite import *
import pandas as pd


def generate_data(env: FaultSatellite, num_samples=1000):
    """
    生成训练数据
    :param env: 环境对象, 包含q和omega的状态
    :param num_samples: 生成样本的数量
    :return: 输入数据X和输出数据y
    """
    inputs = []
    outputs = []
    env.reset()

    for i in range(num_samples):
        # 随机生成一个[-1, 1]之间的四维效率向量
        efficiency = np.random.rand(4) * 2 - 1  # 在[-1, 1]区间内随机生成4个数
        
        # 转变成输出力矩，liag * u_max
        torque = np.diag(efficiency) @ env.u_max  # 计算力矩
        
        # 生成输入数据：omega_k、力矩、omega_{k+1}
        input_data = np.concatenate((env.omega.flatten(), torque.flatten(), np.zeros(3)))  # 3维角速度 + 4维力矩 + 3维角速度
        
        # 更新到下一时刻的状态
        _, _, done, _ = env.step(torque.reshape(-1, 1))  # 更新状态
        input_data[7:10] = env.omega.flatten()

        # 生成输出数据：当前时刻的故障值
        output_data = np.zeros(4)
        if hasattr(env, 'u_f'):
            output_data = env.u_f.flatten()

        # 将输入输出数据添加到列表中
        inputs.append(input_data)
        outputs.append(output_data)

        if done:
            print(f"episode {i+1}/{num_samples} done")
            env.reset()

    # 将数据转换为numpy数组
    inputs = np.stack(inputs)  # (num_samples, 6)
    outputs = np.stack(outputs)  # (num_samples, 3)

    return inputs, outputs


if __name__ == '__main__':
    env = FaultSatellite(t_max=1000)  # 这里创建你的环境实例
    X_train, y_train = generate_data(env, num_samples=500000)

    # 查看生成的数据维度
    print("X_train shape:", X_train.shape)  # (num_samples, 10)
    print("y_train shape:", y_train.shape)  # (num_samples, 7)

    # save to csv
    df_x = pd.DataFrame(X_train, columns=['omega0', 'omega1', 'omega2', 'u0', 'u1', 'u2', 'u3', 'omega0next', 'omega1next', 'omega2next'])
    df_y = pd.DataFrame(y_train, columns=['f0', 'f1', 'f2', 'f3'])
    df = pd.concat([df_x, df_y], axis=1)
    df.to_csv('sample_data/data.csv', index=False)
