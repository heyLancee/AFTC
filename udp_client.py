import socket
import struct
import time
import sys
from base import TelemetryStruct, CommuDataType

class UdpClient:
    def __init__(self, host, port, T=10, Ts=0.1, header="SSSSSSSS", tail="EEEEEEEE"):
        self.host = host
        self.port = port
        self.T = T
        self.Ts = Ts
        self.header = header
        self.tail = tail

    def connect_to_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print("UDP socket created")
            return True
        except Exception as e:
            print(f"Error creating socket: {e}")
            return False

    def send_data(self, telemetry_data: TelemetryStruct):
        # 示例帧头和帧尾
        frame_head = self.header.encode('utf-8')
        frame_tail = self.tail.encode('utf-8')

        # 将数据转换为字节数组
        byte_array = telemetry_data.to_byte_array()

        # 组装完整的包（帧头、数据类型、数据、帧尾）
        packet = frame_head
        packet += struct.pack('B', CommuDataType.TELEMETRY.value)  # 0表示telemetryType，作为数据标识符
        packet += byte_array
        packet += frame_tail

        # 发送数据，使用sendto而不是send
        self.sock.sendto(packet, (self.host, self.port))

    def close(self):
        self.sock.close()


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 udp_client.py <Host> <Port> <T> <Ts>")
        sys.exit(1)

    try:
        host = sys.argv[1]
        port = int(sys.argv[2])
        T = float(sys.argv[3])
        Ts = float(sys.argv[4])
        print(f"host: {host}, port: {port}, T: {T}, Ts: {Ts}")
    except ValueError as e:
        print(f"Invalid argument: {e}")
        sys.exit(1)

    client = UdpClient(host, port, T, Ts)
    if client.connect_to_server():
        client.send_data()
    else:
        print("Failed to connect to server")
    client.close()
