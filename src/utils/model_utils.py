import numpy as np
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoConfig

from .common_utils import to

### Calibration utils and modules

LINEAR_LAYERS = (nn.Linear, _ConvNd)


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):

    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        # register the wrapped module as a proper submodule
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []

    def forward(self, *input_args, **input_kwargs):
        """
        Assumes that the wrapped module has a single
        input that can reside in inputs or input_kwargs.
        """
        if self.cpu_offload:
            input_args = to(input_args, device="cpu")
            input_kwargs = to(input_kwargs, device="cpu")

        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)

        # keep your existing behavior
        raise ForwardInterrupt

    def __getattr__(self, name):
        """
        Delegate unknown attributes to the wrapped module so that
        things like `decoder_layer.attention_type` keep working.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
def get_number_of_rows_and_cols(layer):
    return layer.weight.shape[0], np.prod(layer.weight.shape[1:])
