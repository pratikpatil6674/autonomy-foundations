from dataclasses import dataclass
from enum import StrEnum

@dataclass
class VehicleParams:
    wheelbase: float
    delta_min: float
    delta_max: float
    v_min: float
    v_max: float
    a_min: float
    a_max: float
    steering_constant: float

@dataclass
class SimParams:
    dt: float
    # steps: int
    integrator: str

@dataclass
class NoiseParams:
    x_sigma: float  # process noise
    y_sigma: float  # process noise
    psi_sigma: float  # process noise
    v_sigma: float  # process noise
    delta_sigma: float  # process noise
    delta_cmd_sigma: float  # control noise
    a_sigma: float  # control noise
    steering_bias: float  # steering bias

class NoiseMode(StrEnum):
    OFF = "OFF"
    IID = "IID"
    BIAS = "BIAS"
