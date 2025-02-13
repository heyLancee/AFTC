import socket
import struct
import time
import sys
from base import TelemetryStruct, CommuDataType, FaultParaStruct


class PackageManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PackageManager, cls).__new__(cls)
            cls._instance.header = ""
            cls._instance.tail = ""
        return cls._instance
    
    def set_package_params(self, header: str, tail: str):
        self.header = header
        self.tail = tail
    
    def package(self, data: str, identifier: int) -> str:
        return f"{self.header}{chr(identifier)}{data}{self.tail}"
    
    def unpackage(self, package: str) -> tuple[str, int]:
        data_type = None
        if not self.validate_package(package, data_type):
            return "", None
        
        data_start = len(self.header) + 1  # Header length + identifier length
        data_end = len(package) - len(self.tail)
        
        return package[data_start:data_end], data_type
    
    def validate_package(self, package: str, data_type: int) -> bool:
        min_length = len(self.header) + 1 + len(self.tail)
        if len(package) < min_length:
            return False
        
        if not (package.startswith(self.header) and package.endswith(self.tail)):
            return False
        
        data_type = ord(package[len(self.header)])
        return True


class UdpClient:
    def __init__(self, host, port, T=10, Ts=0.1, header="SSSSSSSS", tail="EEEEEEEE", local_port=None):
        self.host = host
        self.port = port
        self.T = T
        self.Ts = Ts
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

    def send_data(self, telemetry_data: TelemetryStruct):
        packet = self.package_manager.package(telemetry_data, CommuDataType.TELEMETRY.value)
        self.sock.sendto(packet, (self.host, self.port))

    def start_receiving(self):
        """启动数据接收线程"""
        import threading
        self.is_receiving = True
        self.receive_thread = threading.Thread(target=self._receive_data)
        self.receive_thread.daemon = True  # 设置为守护线程
        self.receive_thread.start()

    def stop_receiving(self):
        """停止数据接收"""
        self.is_receiving = False
        if hasattr(self, 'receive_thread'):
            self.receive_thread.join()

    def _receive_data(self):
        """接收数据的内部方法"""
        if not self.sock:
            print("Socket not initialized")
            return

        self.sock.settimeout(1.0)  # 设置超时时间为1秒
        buffer_size = 1024

        while self.is_receiving:
            try:
                data, addr = self.sock.recvfrom(buffer_size)
                data_content, data_type = self.package_manager.unpackage(data)
                
                if data_type is None:
                    continue

                if data_type == CommuDataType.FAULT_PARA.value:
                    self._handle_fault_para(data_content)
                elif data_type == CommuDataType.SAVE_DATA.value:
                    self._handle_save_data(data_content)

            except socket.timeout:
                continue
            except Exception as e:
                print(f"Error receiving data: {e}")
                continue

    def _handle_fault_para(self, data):
        """处理故障参数数据"""
        try:
            fault_para = FaultParaStruct.from_byte_array(data)
            print(f"Received fault parameters: {vars(fault_para)}")
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
    if len(sys.argv) < 5:
        print("Usage: python3 udp_client.py <Host> <Port> <T> <Ts> [LocalPort]")
        sys.exit(1)

    try:
        host = sys.argv[1]
        port = int(sys.argv[2])
        T = float(sys.argv[3])
        Ts = float(sys.argv[4])
        local_port = int(sys.argv[5]) if len(sys.argv) > 5 else None
        print(f"host: {host}, port: {port}, T: {T}, Ts: {Ts}, local_port: {local_port}")
    except ValueError as e:
        print(f"Invalid argument: {e}")
        sys.exit(1)

    client = UdpClient(host, port, T, Ts, local_port=local_port)
    if client.connect_to_server():
        client.start_receiving()  # 启动接收线程
        try:
            client.send_data()
            # 可以在这里添加主程序逻辑
            time.sleep(T)  # 运行T秒
        finally:
            client.close()
    else:
        print("Failed to connect to server")



