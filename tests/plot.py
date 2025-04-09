import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
root_path = os.path.dirname(parent_dir)
sys.path.append(root_path)

from scipy import io
import numpy as np
import time
from communication.udp_commu import UdpClient
from base import TelemetryStruct


def send_data_sequence(omega_data, angle_data, host="127.0.0.1", port=12345, T=1000, Ts=0.5):
    """
    通过UDP发送数据序列
    
    Args:
        omega_data (np.ndarray): omega数据
        angle_data (np.ndarray): angle数据
        host (str): UDP服务器地址
        port (int): UDP服务器端口
        T (float): 总运行时间
        Ts (float): 采样时间间隔
    """
    # 初始化UDP客户端
    client = UdpClient(host, port, T, Ts)
    if not client.connect_to_server():
        print("Failed to connect to server")
        return
    
    try:
        cnt = 0
        curr_time = 0
        print(f"Starting to send data sequence")

        while curr_time < T:
            # 创建遥测数据结构
            telemetry = TelemetryStruct()
            # TODO: 根据实际的数据结构设置telemetry的字段
            telemetry.timeStep = curr_time
            telemetry.wx = omega_data[cnt, 0]
            telemetry.wy = omega_data[cnt, 1]
            telemetry.wz = omega_data[cnt, 2]
            telemetry.zAngle = angle_data[cnt, 0]

            # 发送数据
            client.send_data(telemetry)

            cnt += 1
            curr_time += Ts

            time.sleep(0.01)
        
    finally:
        client.close()

def process_and_send_mat_data(omega_path, angle_path, host="127.0.0.1", port=1200, T=10, Ts=0.1):
    """
    处理mat文件并发送数据的主函数
    
    Args:
        omega_path (str): omega数据路径
        angle_path (str): angle数据路径
        host (str): UDP服务器地址
        port (int): UDP服务器端口
        T (float): 总运行时间
        Ts (float): 采样时间间隔
    """
    try:
        # 读取数据
        omega_data = io.loadmat(omega_path)
        angle_data = io.loadmat(angle_path)

        omega_data = omega_data.get('data', [])
        omega_data = np.array(omega_data).reshape(-1, 3)
        angle_data = angle_data.get('theta_pro', [])
        angle_data = np.array(angle_data).reshape(-1, 1)
        
        # 发送数据
        send_data_sequence(omega_data, angle_data, host, port, T, Ts)
        
    except Exception as e:
        print(f"Error processing and sending data: {e}")
        raise

if __name__ == "__main__":
    # 示例用法
    # if len(sys.argv) != 2:
    #     print("Usage: python plot.py <mat_file_path>")
    #     sys.exit(1)
        
    # mat_file_path = sys.argv[1]
    omega_path = r"D:\PersonalFiles\大论文\pre\FTCPGNN\数据\new\f3_proposed.mat"
    angle_path = r"D:\PersonalFiles\大论文\pre\FTCPGNN\数据\new\f3_proposed_theta.mat"
    
    # 可以根据需要修改这些默认参数
    HOST = "192.168.233.129"
    PORT = 1200    
    T = 1000  # 总运行时间（秒）
    Ts = 0.5  # 采样时间间隔（秒）
    
    process_and_send_mat_data(omega_path, angle_path, HOST, PORT, T, Ts)
