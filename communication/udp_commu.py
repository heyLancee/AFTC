import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
root_path = os.path.dirname(parent_dir)
sys.path.append(root_path)

import socket
import numpy as np
import time
from src.base import TelemetryStruct, CommuDataType, PackageManager, FaultParams
from src.satellite import FaultSatellite


class UdpClient:
    def __init__(self, host, port, header="SSSSSSSS", tail="EEEEEEEE", local_port=None):
        self.host = host
        self.port = port
        self.header = header
        self.tail = tail
        self.local_port = local_port  # 添加本地端口参数

        self.package_manager = PackageManager()
        self.package_manager.set_package_params(header, tail)

        self.sock = None
        self.is_receiving = False  # 控制接收循环的标志

    def connect_to_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            
            if self.local_port:
                # 如果指定了本地端口，则绑定到该端口
                self.sock.bind(('', self.local_port))
                print(f"UDP socket bound to local port {self.local_port}")
            
            print("UDP socket created")
            return True
        except Exception as e:
            print(f"Error creating/binding socket: {e}")
            return False

    def send_data(self, env):
        telemetry_data = TelemetryStruct()
        telemetry_data.timeStep = env.t
        telemetry_data.wx = env.omega[0] * 180 / np.pi
        telemetry_data.wy = env.omega[1] * 180 / np.pi
        telemetry_data.wz = env.omega[2] * 180 / np.pi
        telemetry_data.q0 = env.q[0]
        telemetry_data.q1 = env.q[1]
        telemetry_data.q2 = env.q[2]
        telemetry_data.q3 = env.q[3]
        telemetry_data.zAngle = env.theta * 180 / np.pi
        packet = self.package_manager.package(telemetry_data, CommuDataType.TELEMETRY)
        self.sock.sendto(packet, (self.host, self.port))

    def start_receiving(self, env):
        """启动数据接收线程"""
        import threading
        self.is_receiving = True
        self.receive_thread = threading.Thread(target=self._receive_data, args=(env,))
        self.receive_thread.daemon = True  # 设置为守护线程
        self.receive_thread.start()

    def stop_receiving(self):
        """停止数据接收"""
        self.is_receiving = False
        if hasattr(self, 'receive_thread'):
            self.receive_thread.join()

    def _receive_data(self, env:FaultSatellite):
        """接收数据的内部方法"""
        if not self.sock:
            print("Socket not initialized")
            return

        self.sock.settimeout(1.0)
        buffer_size = 1024

        while self.is_receiving:
            try:
                data, addr = self.sock.recvfrom(buffer_size)
                print(f"Received data from {addr}: {data}")
                data = self.package_manager.unpackage(data)

                if isinstance(data, FaultParams):
                    self._handle_fault_para(data, env)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error receiving data: {e}")
                continue

    def _handle_fault_para(self, data: FaultParams, env:FaultSatellite):
        """处理故障参数数据"""
        try:
            print(f"Received fault parameters: {vars(data)}")
            print(f"Fault params: {data.params}")
            
            env.fault_mode = data.fault_type
            env.fault_params = data.params
            env.fault_start_time = data.fault_start_time
            env.fault_end_time = data.fault_end_time
            env.flywheel_fault_idx = data.flywheel_fault_idx
            
            # 在这里添加你的故障参数处理逻辑
        except Exception as e:
            print(f"Error handling fault parameters: {e}")

    def _handle_save_data(self, data):
        """处理保存数据请求"""
        try:
            print(f"Received save data command: {data}")
            # 在这里添加你的数据保存处理逻辑
        except Exception as e:
            print(f"Error handling save data: {e}")

    def close(self):
        """关闭客户端连接"""
        self.stop_receiving()
        if self.sock:
            self.sock.close()


if __name__ == "__main__":
    # if len(sys.argv) < 5:
    #     print("Usage: python3 udp.py <Host> <Port> [LocalPort]")
    #     sys.exit(1)

    # try:
    #     host = sys.argv[1]
    #     port = int(sys.argv[2])
    #     local_port = int(sys.argv[3]) if len(sys.argv) > 3 else None
    #     print(f"host: {host}, port: {port}, local_port: {local_port}")
    # except ValueError as e:
    #     print(f"Invalid argument: {e}")
    #     sys.exit(1)

    host = "192.168.233.129"
    port = 1200
    local_port = 5010

    from satellite import SunPointFaultSatellite
    from configs.config import EnvConfig
    config = EnvConfig()
    env = SunPointFaultSatellite(config)
    env.reset()

    client = UdpClient(host, port, local_port=local_port)
    if client.connect_to_server():
        client.start_receiving()  # 启动接收线程
        try:
            client.send_data(env)
            # 可以在这里添加主程序逻辑
            time.sleep(60)  # 运行T秒
        finally:
            client.close()
    else:
        print("Failed to connect to server")



