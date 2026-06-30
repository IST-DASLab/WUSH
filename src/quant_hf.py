"""
Register this repo's QuantLinear as a transformers quantizer.

A checkpoint saved with `model.save_pretrained` reloads straight back into `QuantLinear` via
`AutoModelForCausalLM.from_pretrained` -- no manual state-dict surgery. The on-disk format is
metadata-free: `config.json`'s `quantization_config` carries only `{"quant_method": "wush"}`, the dispatch
key that routes `from_pretrained` to `WushHfQuantizer`. Which linears are quantized, the group size, the
scale dtype / format, and the transform shape are all recovered from the stored tensors: the swap converts
exactly the linears whose packed weight `<name>.e2m1` exists in the checkpoint, into a bare `QuantLinear`
skeleton whose buffers the by-name weight loader fills with the checkpoint tensors. The loader is lenient on
buffer SHAPE (it exempts a quantizer's buffers from the shape check) but fills each buffer by CASTING the
checkpoint tensor into the buffer's existing dtype -- so the skeleton's `scale_quant` must be built in the
checkpoint's fp8 dtype (read from the header), else nvfp4 (e4m3) scales would be silently rounded to the
default e8m0 and the forward mis-routed. Every other buffer's dtype is format-invariant (uint8 codes, fp32
scalars) or harmless under the cast (the bf16 transform widens losslessly and is re-cast at use).

Importing this module registers the config + quantizer, so import it before `from_pretrained`.
"""

import torch
from safetensors import safe_open
from transformers.quantizers.auto import register_quantization_config, register_quantizer
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from quant_module import QuantLinear


QUANT_METHOD: str = 'wush'


@register_quantization_config(QUANT_METHOD)
class WushQuantConfig(QuantizationConfigMixin):
    # The checkpoint is self-describing, so the only thing config.json must carry is the dispatch key that
    # routes from_pretrained to WushHfQuantizer; everything else is read from the stored tensors. to_dict()
    # (deepcopy of __dict__) therefore emits exactly {'quant_method': 'wush'}, and from_dict replays it.
    def __init__(self, **kwargs) -> None:
        self.quant_method = QUANT_METHOD


def replace_with_quant_linear(module: torch.nn.Module, quant_scale_dtypes: dict[str, torch.dtype], prefix: str = '') -> None:
    # Recursively swap each nn.Linear whose dotted name is in quant_scale_dtypes for a bare meta QuantLinear
    # skeleton: the loader overwrites each buffer by name from the checkpoint (recovering shape / group size /
    # transform), so the buffer NAMES must match -- and the scale buffer's fp8 DTYPE must match too, since the
    # loader casts into the existing buffer rather than replacing it (hence we pass the checkpoint's scale
    # dtype). Building on meta avoids any real allocation.
    for name, child in list(module.named_children()):
        child_name: str = f'{prefix}{name}'
        if child_name in quant_scale_dtypes and isinstance(child, torch.nn.Linear):
            # A 0-size placeholder (not None) for `transform`: transformers' expected-key set is the model's
            # state_dict, which omits None buffers -- so a None `transform` would make the checkpoint's
            # `<prefix>.transform` an unexpected (dropped) key. As a real buffer it is loaded by name.
            # scale_dtype builds scale_quant in the checkpoint's fp8 format (e8m0 mxfp4 / e4m3 nvfp4) so the
            # loader's cast-into-buffer is a no-op that preserves the stored scale values + format routing.
            with torch.device('meta'):
                quant_linear: QuantLinear = QuantLinear(transform=torch.empty(0), scale_dtype=quant_scale_dtypes[child_name])
            for buffer_name, buffer in list(quant_linear._buffers.items()):
                if buffer is not None and buffer.device.type != 'meta':  # keep the skeleton fully on meta
                    quant_linear._buffers[buffer_name] = buffer.to('meta')
            quant_linear.requires_grad_(False)
            setattr(module, name, quant_linear)
        else:
            replace_with_quant_linear(module=child, quant_scale_dtypes=quant_scale_dtypes, prefix=f'{child_name}.')


@register_quantizer(QUANT_METHOD)
class WushHfQuantizer(HfQuantizer):
    requires_calibration = False

    def _process_model_before_weight_loading(self, model, checkpoint_files=None, **kwargs):
        # A linear is quantized iff the checkpoint stores its packed fp4 weight '<prefix>.e2m1'. For each such
        # prefix also read the fp8 dtype of its '<prefix>.scale_quant' from the header, so the skeleton's scale
        # buffer can be built in the matching fp8 format -- the scale's fp8 dtype IS the format tag (e8m0 => mxfp4,
        # e4m3 => nvfp4), and matching it makes the loader's cast-into-buffer a no-op (see module docstring). Only
        # the safetensors headers are read (no tensor data).
        safetensors_fp8_to_torch: dict[str, torch.dtype] = {
            'F8_E8M0': torch.float8_e8m0fnu,
            'F8_E4M3': torch.float8_e4m3fn,
        }
        suffix: str = '.e2m1'
        quant_scale_dtypes: dict[str, torch.dtype] = {}
        for path in checkpoint_files or ():
            with safe_open(path, framework='pt') as f:
                for key in f.keys():
                    if key.endswith(suffix):
                        prefix: str = key[:-len(suffix)]
                        st_dtype: str = f.get_slice(f'{prefix}.scale_quant').get_dtype()
                        if st_dtype not in safetensors_fp8_to_torch:
                            raise ValueError(f"{prefix}.scale_quant has unsupported fp8 dtype {st_dtype!r}; expected one of {sorted(safetensors_fp8_to_torch)}")
                        quant_scale_dtypes[prefix] = safetensors_fp8_to_torch[st_dtype]
        replace_with_quant_linear(module=model, quant_scale_dtypes=quant_scale_dtypes)
        return model

    def _process_model_after_weight_loading(self, model, **kwargs):
        model.requires_grad_(False)
        return model

    def is_serializable(self, *args, **kwargs) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        return False
