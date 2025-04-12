"""
下位机串口通信
"""
import os
import sys

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
root_path = os.path.dirname(parent_dir)
sys.path.append(root_path)

from collections import deque
import serial
import time
import threading
from typing import Callable, Optional
from queue import Queue, Empty, Full
import logging
import struct
from concurrent.futures import ThreadPoolExecutor

from src.base import TelemetryStruct, PackageManager, CommuDataType


class CallbackEvent:
    """
    回调函数触发类型
    """
    RECV_TELE_DATA = 1


class SerialComm:
    """
    串口通信类，负责与下位机硬件通过RS232进行通信
    """
    def __init__(self, port: str, baudrate: int, timeout: Optional[int] = 1, queue_size: int = 1000, communication_frequency: int = 200,
                 callback: Callable[[CallbackEvent, TelemetryStruct], None] = None, rtx_buffer_size: int = 4096, 
                 max_thread_poll_workers=2, header: str = "SSSSSSSS", tail: str = "EEEEEEEE"):
        """
        初始化串口通信

        Args:
            port: RS232端口名称
            baudrate: 波特率
            timeout: 串口超时时间
            queue_size: 通信队列大小
            communication_frequency: 通信线程频率
            callback: 回调函数,接收遥测数据字典,可用于实时控制转速
            rtx_buffer_size: 串口接收缓存大小
            max_thread_poll_workers: 线程池最大线程数
            header: 包头
            tail: 包尾
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.logger = logging.getLogger(__name__)

        self.cmd_queue = Queue(maxsize=queue_size)
        self.resp_queue = Queue(maxsize=queue_size)

        self._communication_frequency = communication_frequency

        self._comm_thread = None
        self._polling_thread = None
        self._resp_thread = None
        self._proc_resp_thread = None

        self._polling = False
        self._running = False
        self._is_connected = False

        self.callback = callback

        self.serial = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout
        )
        self.serial.set_buffer_size(rx_size=rtx_buffer_size, tx_size=rtx_buffer_size)
        self.serial.reset_input_buffer()

        self.thread_poll = ThreadPoolExecutor(max_workers=max_thread_poll_workers, thread_name_prefix="serial_comm_thread_poll")

        self.package_manager = PackageManager()
        self.package_manager.set_package_params(header, tail)

    def __del__(self):
        """
        析构函数，确保资源正确释放
        """
        self.disconnect()

    def connect(self) -> bool:
        """
        建立串口连接

        Returns:
            bool: 是否成功连接

        Raises:
            ConnectionError: 当连接失败时
        """
        if self._is_connected:
            self.logger.warning("串口已连接")
            return True

        try:
            if not self.serial.is_open:
                self.serial.open()
            self._is_connected = True
            return True
        except Exception as e:
            self.logger.error(f"串口连接失败: {str(e)}")
            self._is_connected = False
            return False

    def start(self):
        """
        启动串口通信和轮询
        """
        if self._running:
            self.logger.warning("串口已启动")
            return False

        self.logger.info("启动")
        self._running = True

        self.serial.reset_input_buffer()
        self.serial.reset_output_buffer()

        # 启动通信线程
        self._comm_thread = threading.Thread(target=self._communication_loop, daemon=True)
        self._comm_thread.start()
        self.logger.info(f"已启动通信线程, 频率: {self._communication_frequency}Hz")

        # 启动响应接收线程
        self._resp_thread = threading.Thread(target=self._response_loop, daemon=True)
        self._resp_thread.start()
        self.logger.info(f"已启动响应接收线程, 频率: {self._communication_frequency}Hz")

        # 启动响应处理线程
        self._proc_resp_thread = threading.Thread(target=self._process_response, daemon=True)
        self._proc_resp_thread.start()
        self.logger.info(f"已启动响应处理线程")

        return True

    def stop(self):
        """
        停止串口通信和轮询
        """
        self._running = False
        self._polling = False

        # 哨兵
        self.cmd_queue.put(None)
        self.resp_queue.put(None)
        self.serial.cancel_read()

        # 等待所有线程结束
        if self._comm_thread is not None:
            self._comm_thread.join()
        if self._resp_thread is not None:
            self._resp_thread.join()
        if self._polling_thread is not None:
            self._polling_thread.join()
        if self._proc_resp_thread is not None:
            self._proc_resp_thread.join()

    def request(self, data: TelemetryStruct) -> bool:
        """
        请求控制指令

        Args:
            data: 控制指令结构体
            
        Returns:
            bool: 是否成功将命令放入队列

        Raises:
            Full: 当命令队列已满时
        """

        command = self.package_manager.package(data, CommuDataType.TELEMETRY)
        try:
            self.cmd_queue.put_nowait(command)
            return True
        except Full:
            self.logger.error("命令队列已满")
            return False

    def disconnect(self):
        """
        断开与飞轮的连接
        """
        self._running = False  # 这会终止所有线程
        self._polling = False  # 为了明确性,也设置轮询标志

        # 等待通信线程完成
        if self._comm_thread is not None and self._comm_thread.is_alive():
            self._comm_thread.join()
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join()

        if self.serial is not None and self.serial.is_open:
            self.serial.close()
        self._is_connected = False

    def _communication_loop(self) -> None:
        """
        通信循环，只负责发送命令
        """
        period = 1.0 / self._communication_frequency
        next_time = time.perf_counter() + period

        while self._running:
            try:
                if not self._is_connected:
                    time.sleep(1)
                    continue

                try:
                    command = self.cmd_queue.get()
                except Empty:
                    continue

                if not command:
                    continue

                write_len = self.serial.write(command)

                if write_len != len(command):
                    self.logger.error(f"发送命令失败: {command.hex()}")
                    continue

                next_time = self._wait_for_next_cycle(next_time, period)

            except Exception as e:
                self.logger.exception(f"通信循环错误: {e}")

        print("communication loop exit")

    def _response_loop(self) -> None:
        """
        响应处理循环，专门处理串口响应数据
        """
        period = 1.0 / self._communication_frequency  # 使用与通信相同的频率
        next_time = time.perf_counter() + period

        while self._running:
            try:
                if not self._is_connected:
                    next_time = self._wait_for_next_cycle(next_time, period)
                    continue

                response = self.serial.read(size=self.serial.in_waiting)
                if not response:
                    next_time = self._wait_for_next_cycle(next_time, period)
                    continue

                try:
                    self.resp_queue.put_nowait(response)
                except Full:
                    self.logger.error("命令队列已满")
                    continue

                next_time = self._wait_for_next_cycle(next_time, period)

            except Exception as e:
                self.logger.exception(f"响应处理循环错误: {e}")
                next_time = self._wait_for_next_cycle(next_time, period)

        print("response loop exit")

    def _process_response(self) -> None:
        """
        处理响应数据，维护缓冲区处理字节串
        """
        period = 1.0 / self._communication_frequency  # 使用与通信相同的频率
        next_time = time.perf_counter() + period

        buffer = bytearray()
        per_packet_len = len(self.package_manager.header) + len(self.package_manager.tail)
        per_packet_len += struct.calcsize('I')
        per_packet_len += struct.calcsize('<25d')

        while self._running:
            try:
                if self.resp_queue.empty():
                    next_time = self._wait_for_next_cycle(next_time, period)
                    continue

                chunk = self.resp_queue.get()
                if not chunk:
                    continue

                buffer.extend(chunk)

                # 查找header的位置
                while True:
                    header_index = buffer.find(self.package_manager.header)
                    if header_index == -1:
                        buffer.clear()
                        break
                    
                    # 清除header之前的数据
                    buffer = buffer[header_index:]

                    # 检查从header开始是否有足够的数据
                    if len(buffer) >= per_packet_len:
                        packet = buffer[:per_packet_len]
                        try:
                            data = self.package_manager.unpackage(packet)
                            if not isinstance(data, TelemetryStruct):
                                self.logger.error(f"解包数据失败: {data}")
                                buffer = buffer[per_packet_len:]
                                continue
                            if self.callback:
                                self.thread_poll.submit(self.callback, CallbackEvent.RECV_TELE_DATA, data)
                        except Exception as e:
                            self.logger.error(f"解包数据时发生错误: {str(e)}")

                        # 移除已处理的数据
                        buffer = buffer[per_packet_len:]
                    else:
                        break

            except Exception as e:
                self.logger.error(f"处理响应数据时发生错误: {str(e)}")

        print("process response loop exit")

    def _wait_for_next_cycle(self, next_time: float, period: float) -> float:
        """
        等待下一个周期
        """
        current_time = time.perf_counter()
        sleep_time = next_time - current_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        next_time = next_time + period

        return next_time


if __name__ == '__main__':
    # 用虚拟串口测试下
    def callback(event: CallbackEvent, data: TelemetryStruct):
        print(f"callback: {event}")
        if event == CallbackEvent.RECV_TELE_DATA:
            print(vars(data))

    serial_comm = SerialComm(port="COM5", baudrate=115200, communication_frequency=200, callback=callback, timeout=None)
    serial_comm.connect()
    serial_comm.start()

    for _ in range(3):
        time.sleep(1)
        telemetry_data = TelemetryStruct()
        telemetry_data.timeStep = 1.0
        telemetry_data.wx = 1.0
        telemetry_data.wy = 1.0
        telemetry_data.wz = 1.0

        serial_comm.request(telemetry_data)

    serial_comm.stop()
    serial_comm.disconnect()
