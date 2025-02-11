import struct
import json
from enum import Enum

# 定义枚举类型
class CommuDataType(Enum):
    TELEMETRY = 0
    FAULT_RESULT = 1
    RUN_PLATFORM = 2
    STOP_PLATFORM = 3
    FAULT_PARA = 4
    SAVE_DATA = 5

# 数据结构定义
class TelemetryStruct:
    def __init__(self):
        self.timeStep : float = 0.0
        self.wx : float = 0.0
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
        self.tx : float = 0.0
        self.ty : float = 0.0
        self.tz : float = 0.0
        self.zAngle : float = 0.0

    # 从字节数据中恢复数据
    @staticmethod
    def from_byte_array(data):
        if len(data) < 96:
            return None
        
        values = struct.unpack('<24f', data[:96])  # 24个float类型数据
        obj = TelemetryStruct()
        obj.timeStep = values[0]
        obj.wx, obj.wy, obj.wz = values[1:4]
        obj.q0, obj.q1, obj.q2, obj.q3 = values[4:8]
        obj.rx, obj.ry, obj.rz = values[8:11]
        obj.vx, obj.vy, obj.vz = values[11:14]
        obj.wboX, obj.wboY, obj.wboZ = values[14:17]
        obj.qbo0, obj.qbo1, obj.qbo2, obj.qbo3 = values[17:21]
        obj.tx, obj.ty, obj.tz = values[21:24]
        obj.zAngle = values[23]
        return obj

    # 转换为字节数组
    def to_byte_array(self):
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

class FaultResultStruct:
    def __init__(self):
        self.timeStep = 0.0
        self.faultValue = 0.0
        self.faultType = 0

    @staticmethod
    def from_byte_array(data):
        if len(data) < 12:
            return None
        values = struct.unpack('<3f', data[:12])
        obj = FaultResultStruct()
        obj.timeStep, obj.faultValue, obj.faultType = values
        return obj

    def to_byte_array(self):
        return struct.pack('<3f', self.timeStep, self.faultValue, self.faultType)

class FaultParaStruct:
    def __init__(self):
        self.faultTimeLow = 0.0
        self.faultAttLow = 0.0
        self.faultTimeUp = 0.0
        self.faultAttUp = 0.0
        self.faultType = 0
        self.gyroGroup = 0
        self.gyroID = 0
        self.runMode = 0

    def to_byte_array(self):
        return struct.pack('<8f', 
                           self.faultTimeLow, self.faultAttLow,
                           self.faultTimeUp, self.faultAttUp,
                           self.faultType,
                           self.gyroGroup, self.gyroID, self.runMode)

class SaveDataStruct:
    def __init__(self):
        self.gyroIsChecked = False
        self.starIsChecked = False
        self.sunIsChecked = False
        self.rwIsChecked = False
        self.path = ""

    def to_byte_array(self):
        return struct.pack('<4?', 
                           self.gyroIsChecked, self.starIsChecked,
                           self.sunIsChecked, self.rwIsChecked) + \
               self.path.encode('utf-8')

# 包管理类
class PackageManager:
    def __init__(self):
        self.header = ""
        self.tail = ""

    def set_package_params(self, header, tail):
        self.header = header
        self.tail = tail

    def package(self, data, identifier: CommuDataType):
        package = self.header
        package += struct.pack('<B', identifier.value)  # 以字节形式添加标识符
        package += data.encode('utf-8')
        package += self.tail
        return package

    def unpackage(self, package):
        if not self.validate_package(package):
            return None, None
        
        data_start = len(self.header) + 1
        data_end = len(package) - len(self.tail)
        
        data = package[data_start:data_end].decode('utf-8')
        data_type = CommuDataType(package[len(self.header)])
        return data, data_type

    def validate_package(self, package):
        min_length = len(self.header) + 1 + len(self.tail)
        if len(package) < min_length:
            return False
        
        if package[:len(self.header)] != self.header or package[-len(self.tail):] != self.tail:
            return False
        
        return True

