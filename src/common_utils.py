import dataclasses
import gc
import random

import numpy as np
import torch


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def clear_device_cache(garbage_collection=False):
    if garbage_collection:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.xpu.is_available():
        torch.xpu.empty_cache()


def to(data, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list, set)):
        return type(data)(to(x, *args, **kwargs) for x in data)
    elif isinstance(data, dict):
        return type(data)((k, to(v, *args, **kwargs)) for k, v in data.items())
    elif dataclasses.is_dataclass(data):
        return type(data)(**{k: to(v, *args, **kwargs) for k, v in vars(data).items()})
    else:
        return data  # do nothing if provided value is not tensor or collection of tensors
