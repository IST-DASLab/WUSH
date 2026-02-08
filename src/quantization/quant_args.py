from enum import Enum


class QuantizationFormat(str, Enum):
    """
    Enum storing quantization format options
    """
    INT = "int"
    FP = "fp"
    NVFP = "nvfp"
    MXFP = "mxfp"

class QuantizationGranularity(str, Enum):
    """
    Enum storing quantization granularity options
    """
    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"

class QuantizationObserver(str, Enum):
    """
    Enum storing quantization observer options
    """
    MINMAX = "minmax"
    MSE = "mse"

class QuantizationOrder(str, Enum):
    """
    Enum storing quantization order options
    """
    DEFAULT = "default"
    ACTIVATION = "activation"

class ScalePrecision(str, Enum):
    """
    Enum scale precision options
    """
    FP16 = "fp16"
    E4M3 = "e4m3"
    E8M0 = "e8m0"
