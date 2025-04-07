import struct
from enum import IntEnum
from typing import Union, List, Dict

# 定义枚举类型
class CommuDataType(IntEnum):
    TELEMETRY = 1
    FAULT_RESULT = 2
    FAULT_PARA = 3

# 数据结构定义
class TelemetryStruct:
    def __init__(self):
        self.timeStep : float = 0.0
        self.wx : float = 0.0  # deg/s
        self.wy : float = 0.0
        self.wz : float = 0.0
        self.q0 : float = 0.0
        self.q1 : float = 0.0
        self.q2 : float = 0.0
        self.q3 : float = 0.0
        self.rx : float = 0.0
        self.ry : float = 0.0
        self.rz : float = 0.0
        self.vx : float = 0.0
        self.vy : float = 0.0
        self.vz : float = 0.0
        self.wboX : float = 0.0
        self.wboY : float = 0.0
        self.wboZ : float = 0.0
        self.qbo0 : float = 0.0
        self.qbo1 : float = 0.0
        self.qbo2 : float = 0.0
        self.qbo3 : float = 0.0
        self.tx : float = 0.0   # Nm
        self.ty : float = 0.0
        self.tz : float = 0.0
        self.zAngle : float = 0.0  # deg

    # 从字节数据中恢复数据
    @staticmethod
    def from_byte_array(data):
        # 计算所有成员变量数量
        if len(data) != 25*8:
            return None
        
        values = struct.unpack('<25d', data[:25*8])
        obj = TelemetryStruct()
        obj.timeStep = values[0]
        obj.wx, obj.wy, obj.wz = values[1:4]
        obj.q0, obj.q1, obj.q2, obj.q3 = values[4:8]
        obj.rx, obj.ry, obj.rz = values[8:11]
        obj.vx, obj.vy, obj.vz = values[11:14]
        obj.wboX, obj.wboY, obj.wboZ = values[14:17]
        obj.qbo0, obj.qbo1, obj.qbo2, obj.qbo3 = values[17:21]
        obj.tx, obj.ty, obj.tz = values[21:24]
        obj.zAngle = values[24]
        return obj

    # 转换为字节数组
    def to_byte_array(self) -> bytes:
        return struct.pack('<25d', 
                           self.timeStep,
                           self.wx, self.wy, self.wz,
                           self.q0, self.q1, self.q2, self.q3,
                           self.rx, self.ry, self.rz,
                           self.vx, self.vy, self.vz,
                           self.wboX, self.wboY, self.wboZ,
                           self.qbo0, self.qbo1, self.qbo2, self.qbo3,
                           self.tx, self.ty, self.tz,
                           self.zAngle)

class FaultParams:
    
    class FaultType(IntEnum):
        """故障类型枚举（嵌套在FaultParams类中）"""
        NO_FAULT = 0                    # 无故障
        GYRO_INTERMITTENT_FAULT = 1     # 陀螺间歇故障
        GYRO_SLOW_FAULT = 2             # 陀螺缓变故障
        GYRO_MULTI_FAULT = 3            # 陀螺乘性故障
        FLYWHEEL_PARTIAL_LOSS = 4       # 飞轮部分失效
        FLYWHEEL_BIAS = 5               # 飞轮偏置故障
        FLYWHEEL_COMPREHENSIVE = 6      # 飞轮综合故障

    # 类常量：每种故障类型对应的参数数量
    _PARAM_COUNTS: Dict['FaultParams.FaultType', int] = {
        FaultType.NO_FAULT: 1,
        FaultType.GYRO_INTERMITTENT_FAULT: 1,
        FaultType.GYRO_SLOW_FAULT: 2,
        FaultType.GYRO_MULTI_FAULT: 1,
        FaultType.FLYWHEEL_PARTIAL_LOSS: 1,
        FaultType.FLYWHEEL_BIAS: 1,
        FaultType.FLYWHEEL_COMPREHENSIVE: 2
    }

    def __init__(self, fault_type: FaultType, params: List[float]):
        self.fault_type:FaultParams.FaultType = fault_type
        self.params:List[float] = params

        self.fault_start_time: float = 0.0  # 故障开始时间
        self.fault_end_time: float = 0.0    # 故障结束时间
        
        # 初始化所有可能的参数为0.0
        self.f1: float = 0.0          # 间歇故障系数
        self.lambda_s: float = 0.0     # 慢变系数
        self.k_s: float = 0.0          # 慢变指数
        self.lambda_m: float = 0.0     # 乘性系数
        self.e: float = 0.0            # 部分失效系数
        self.b: float = 0.0            # 偏置系数

        self.gyro_fault_idx: int = 0   # 陀螺故障索引
        self.flywheel_fault_idx: int = 0  # 飞轮故障索引
        
        self._validate_and_assign_params()

    def _validate_and_assign_params(self):
        """验证参数并赋值给对应属性"""
        expected_count = self._get_expected_param_count(self.fault_type)
        num_params = len(self.params)
        if num_params != expected_count:
            raise ValueError(f"Expected {expected_count} parameters for fault type {self.fault_type.name}, but got {num_params}")
        
        if self.fault_type == self.FaultType.NO_FAULT:
            pass
        elif self.fault_type == self.FaultType.GYRO_INTERMITTENT_FAULT:
            self.f1 = self.params[0]
        elif self.fault_type == self.FaultType.GYRO_SLOW_FAULT:
            self.lambda_s, self.k_s = self.params[0], self.params[1]
        elif self.fault_type == self.FaultType.GYRO_MULTI_FAULT:
            self.lambda_m = self.params[0]
        elif self.fault_type == self.FaultType.FLYWHEEL_PARTIAL_LOSS:
            self.e = self.params[0]
        elif self.fault_type == self.FaultType.FLYWHEEL_BIAS:
            self.b = self.params[0]
        elif self.fault_type == self.FaultType.FLYWHEEL_COMPREHENSIVE:
            self.e, self.b = self.params[0], self.params[1]

    def to_bytes(self) -> bytes:
        try:
            return struct.pack(f'<{len(self.params)}d', *self.params)
        except struct.error as e:
            raise ValueError(f"Parameter serialization failed: {e}") from e

    @classmethod
    def from_bytes(cls, data: bytes, fault_type: FaultType) -> 'FaultParams':
        """从字节反序列化"""
        num_params = cls._get_expected_param_count(fault_type)
        if len(data) != num_params * 8:  # 每个double占8字节
            raise ValueError(f"Invalid data length for {fault_type.name}")
            
        params = list(struct.unpack(f'<{num_params}d', data))
        return cls(fault_type, params)

    @classmethod
    def _get_expected_param_count(cls, fault_type: FaultType) -> int:
        """根据故障类型获取预期的参数数量"""
        return cls._PARAM_COUNTS.get(fault_type, 0)

    def to_byte_array(self) -> bytes:
        """序列化为字节数组"""
        fault_param_bytes = self.to_bytes()
        if (
            self.fault_type == self.FaultType.GYRO_INTERMITTENT_FAULT or
            self.fault_type == self.FaultType.GYRO_SLOW_FAULT or
            self.fault_type == self.FaultType.GYRO_MULTI_FAULT
        ):
            fault_component_id:int = self.gyro_fault_idx
        elif (
            self.fault_type == self.FaultType.FLYWHEEL_PARTIAL_LOSS or
            self.fault_type == self.FaultType.FLYWHEEL_BIAS or
            self.fault_type == self.FaultType.FLYWHEEL_COMPREHENSIVE
        ):
            fault_component_id:int = self.flywheel_fault_idx
        return struct.pack('<2f2I',
                         self.fault_start_time,
                         self.fault_end_time,
                         self.fault_type.value,
                         fault_component_id) + fault_param_bytes

    @classmethod
    def from_byte_array(cls, data: bytes) -> 'FaultParams':
        """从字节数组反序列化"""
        if len(data) < 16:  # 2个float + 2个int = 16字节
            raise ValueError("Data too short")
            
        start_time, end_time, fault_type_val, fault_component_id = struct.unpack('<2f2I', data[:16])
        fault_type = FaultParams.FaultType(fault_type_val)

        # 实例化
        ret = FaultParams.from_bytes(data[16:], fault_type)
        ret.fault_start_time = start_time
        ret.fault_end_time = end_time
        if (
            fault_type == FaultParams.FaultType.GYRO_INTERMITTENT_FAULT or
            fault_type == FaultParams.FaultType.GYRO_SLOW_FAULT or
            fault_type == FaultParams.FaultType.GYRO_MULTI_FAULT
        ):
            ret.gyro_fault_idx = fault_component_id
        elif (
            fault_type == FaultParams.FaultType.FLYWHEEL_PARTIAL_LOSS or
            fault_type == FaultParams.FaultType.FLYWHEEL_BIAS or
            fault_type == FaultParams.FaultType.FLYWHEEL_COMPREHENSIVE
        ):
            ret.flywheel_fault_idx = fault_component_id
        return ret
    

class PackageManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.header = b""
            cls._instance.tail = b""
        return cls._instance
    
    def set_package_params(self, header: str, tail: str) -> None:
        """设置数据包头尾标识"""
        self.header = header.encode('utf-8')
        self.tail = tail.encode('utf-8')
    
    def package(self, data: Union[TelemetryStruct, FaultParams], commu_type: CommuDataType) -> bytes:
        """封装数据包：header + type + data + tail"""
        data_bytes = data.to_byte_array()
        return (
            self.header +
            struct.pack('<I', commu_type.value) +
            data_bytes +
            self.tail
        )
    
    def unpackage(self, package: bytes) -> Union[TelemetryStruct, FaultParams, None]:
        """解包数据包"""
        if not self._validate_package(package):
            return None
        
        try:
            # 计算各部分位置
            header_len = len(self.header)
            tail_len = len(self.tail)
            type_start = header_len
            type_end = type_start + 4  # I占4字节
            data_start = type_end
            data_end = len(package) - tail_len

            if data_end <= data_start:
                return None
            
            # 解包类型
            type_value = struct.unpack_from('<I', package, type_start)[0]
            commu_type = CommuDataType(type_value)
            
            # 提取数据部分
            data = package[data_start:data_end]
            
            # 根据类型返回不同对象
            if commu_type == CommuDataType.TELEMETRY:
                return TelemetryStruct.from_byte_array(data)
            elif commu_type == CommuDataType.FAULT_PARA:
                return FaultParams.from_byte_array(data)
            else:
                return None
                
        except (struct.error, ValueError):
            return None
    
    def _validate_package(self, package: bytes) -> bool:
        """验证数据包格式"""
        return (
            len(package) >= len(self.header) + 4 + len(self.tail)  # 最小长度: header + type(I) + tail
            and package.startswith(self.header)
            and package.endswith(self.tail)
        )
    

if __name__ == "__main__":
    import numpy as np

    # 示例用法
    package_manager = PackageManager()
    package_manager.set_package_params("SSSSSSSS", "EEEEEEEE")

    # 测试TelemetryStruct
    telemetry_data = TelemetryStruct()
    telemetry_data.timeStep = 1.0
    telemetry_data.wx = np.random.rand()
    telemetry_data.wy = np.random.rand()
    telemetry_data.wz = np.random.rand()
    telemetry_data.q0 = np.random.rand()
    telemetry_data.q1 = np.random.rand()
    telemetry_data.q2 = np.random.rand()
    telemetry_data.q3 = np.random.rand()
    telemetry_data.zAngle = np.random.rand()
    telemetry_package = package_manager.package(telemetry_data, CommuDataType.TELEMETRY)
    print(f"Telemetry Package: {telemetry_package}")
    unpacked_telemetry = package_manager.unpackage(telemetry_package)
    
    if unpacked_telemetry:
        # 打印十六进制
        print(f"Unpacked Telemetry (Hex): {unpacked_telemetry.to_byte_array().hex()}")

    # 测试FaultParams
    fault_params = FaultParams(FaultParams.FaultType.GYRO_INTERMITTENT_FAULT, [0.5])
    fault_package = package_manager.package(fault_params, CommuDataType.FAULT_PARA)
    unpacked_fault = package_manager.unpackage(fault_package)

    if unpacked_fault:
        print(f"Unpacked Fault: {vars(unpacked_fault)}")
        # 打印故障参数
        print(f"Fault Type: {unpacked_fault.faultType.name}")
        print(f"Fault Params: {unpacked_fault.faultParams.params}")
    
