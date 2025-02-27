import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class SimulationConfig:
    t_max: float
    ts: float
    fault_mode: int

@dataclass
class ActuatorConfig:
    u_max: np.ndarray
    installation_matrix: np.ndarray

@dataclass
class SatelliteConfig:
    inertia: np.ndarray
    actuator: ActuatorConfig

@dataclass
class GyroscopeNoiseConfig:
    ARW: float
    RRW: float

@dataclass
class NoiseConfig:
    noise_mean: float
    noise_std: float

@dataclass
class SensorNoiseConfig:
    gyroscope: GyroscopeNoiseConfig
    quaternion: NoiseConfig
    sun_sensor: NoiseConfig

@dataclass
class SunPointingConfig:
    desired_vector: np.ndarray

@dataclass
class ObservationSpaceConfig:
    upper_bound: np.ndarray
    lower_bound: np.ndarray

@dataclass
class ActionSpaceConfig:
    upper_bound: np.ndarray
    lower_bound: np.ndarray

@dataclass
class FlywheelConfig:
    COM: str
    BAUD: int
    polling_frequency: int
    communication_frequency: int

@dataclass
class UdpConfig:
    host: str
    port: int
    local_port: int
    header: str
    tail: str

class EnvConfig:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EnvConfig, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._load_config()
        self._initialized = True

    def _load_config(self):
        """加载并解析配置文件"""
        try:
            config_path = Path(__file__).parent / 'configs' / 'params.json'
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 解析仿真参数
            self.simulation = SimulationConfig(
                t_max=config['simulation']['t_max'],
                ts=config['simulation']['ts'],
                fault_mode=config['simulation']['fault_mode']
            )

            # 解析卫星参数
            self.satellite = SatelliteConfig(
                inertia=np.array(config['satellite']['inertia']['J']),
                actuator=ActuatorConfig(
                    u_max=np.array(config['satellite']['actuator']['u_max']),
                    installation_matrix=np.array(config['satellite']['actuator']['installation_matrix'])
                )
            )

            # 解析传感器噪声参数
            self.sensor_noise = SensorNoiseConfig(
                gyroscope=GyroscopeNoiseConfig(
                    ARW=config['sensor_noise']['gyroscope']['ARW'],
                    RRW=config['sensor_noise']['gyroscope']['RRW']
                ),
                quaternion=NoiseConfig(
                    noise_mean=config['sensor_noise']['quaternion']['noise_mean'],
                    noise_std=config['sensor_noise']['quaternion']['noise_std']
                ),
                sun_sensor=NoiseConfig(
                    noise_mean=config['sensor_noise']['sun_sensor']['noise_mean'],
                    noise_std=config['sensor_noise']['sun_sensor']['noise_std']
                )
            )

            # 解析定日参数
            desired_vector = np.array(config['sun_pointing']['desired_vector']).reshape(-1, 1)
            self.sun_pointing = SunPointingConfig(
                desired_vector=desired_vector
            )

            # 解析观测
            self.satellite_observation_space = ObservationSpaceConfig(
                upper_bound=np.array(config['satellite_observation_space']['upper_bound']),
                lower_bound=np.array(config['satellite_observation_space']['lower_bound'])
            )

            # 解析动作
            self.satellite_action_space = ActionSpaceConfig(
                upper_bound=np.array(config['satellite_action_space']['upper_bound']),
                lower_bound=np.array(config['satellite_action_space']['lower_bound'])
            )

            # 解析定日观测
            self.sun_pointing_observation_space = ObservationSpaceConfig(
                upper_bound=np.array(config['sun_pointing_observation_space']['upper_bound']),
                lower_bound=np.array(config['sun_pointing_observation_space']['lower_bound'])
            )

            # 解析定日动作
            self.sun_pointing_action_space = ActionSpaceConfig(
                upper_bound=np.array(config['sun_pointing_action_space']['upper_bound']),
                lower_bound=np.array(config['sun_pointing_action_space']['lower_bound'])
            )

            # 解析飞轮参数
            self.flywheel = FlywheelConfig(
                COM=config['flywheel']['COM'],
                BAUD=config['flywheel']['BAUD'],
                polling_frequency=config['flywheel']['polling_frequency'],
                communication_frequency=config['flywheel']['communication_frequency']
            )

            # 解析UDP参数
            self.udp = UdpConfig(
                host=config['udp']['host'],
                port=config['udp']['port'],
                local_port=config['udp']['local_port'],
                header=config['udp']['header'],
                tail=config['udp']['tail']
            )
            
        except Exception as e:
            print(f"Error loading config file: {e}")
            raise

    def reload(self):
        """重新加载配置文件"""
        self._load_config()


# 使用示例
if __name__ == "__main__":
    config = EnvConfig()
    
    # 访问配置参数
    print(f"Simulation time: {config.simulation.t_max}")
    print(f"Inertia matrix:\n{config.satellite.inertia}")
    print(f"Installation matrix:\n{config.satellite.actuator.installation_matrix}")
    print(f"Gyroscope ARW: {config.sensor_noise.gyroscope.ARW}")
    print(f"Desired vector: {config.sun_pointing.desired_vector}") 
    print(f"Observation space: {config.satellite_observation_space.upper_bound}")
    print(f"Action space: {config.satellite_action_space.upper_bound}")
    print(f"Sun pointing observation space: {config.sun_pointing_observation_space.upper_bound}")
    print(f"Sun pointing action space: {config.sun_pointing_action_space.upper_bound}")
    print(f"Flywheel COM: {config.flywheel.COM}")
    print(f"Flywheel BAUD: {config.flywheel.BAUD}")
    print(f"Flywheel polling frequency: {config.flywheel.polling_frequency}")
    print(f"Flywheel communication frequency: {config.flywheel.communication_frequency}")
    print(f"UDP host: {config.udp.host}")
    print(f"UDP port: {config.udp.port}")
    print(f"UDP local port: {config.udp.local_port}")
    print(f"UDP header: {config.udp.header}")
    print(f"UDP tail: {config.udp.tail}")

