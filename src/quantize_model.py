import itertools
from typing import Sequence

import torch

from block_transform import apply_block_transform
from common_utils import clear_device_cache, to
from quant_module import QuantLinear
from quantize_layer import quantize_layer
from triton_kernels.accumulate_hessian import accumulate_hessian


def quantize_layers(
        *weights,
        hessian: torch.Tensor,
        quantize_format_name: str,
        force_rtn: bool = False,
        transform_type: str,
        dampen_ratio: float = 1e-2,
) -> tuple[list[torch.nn.Module], list[dict]]:
    """
    Quantize the (fused) weight matrices using RTN/GPTQ with a block transform
    """
    weight: torch.Tensor = torch.cat(weights, dim=0)
    device: torch.device = weight.device
    weight_quant_result: dict = quantize_layer(
        x=weight,
        hessian=hessian,
        quantize_format_name=quantize_format_name,
        method_name='RTN' if force_rtn else 'GPTQ',
        transform_type=transform_type,
        quantize_side_name='WA',
        dampen_ratio=dampen_ratio,
        round_dtype=None,
    )
    sizes: list[int] = [w.size(0) for w in weights]
    e2m1_splits: tuple = weight_quant_result['e2m1'].split(sizes, dim=0)
    scale_splits: tuple = weight_quant_result['scale_quant'].split(sizes, dim=0)
    quant_layers: list[torch.nn.Module] = [
        QuantLinear(
            bias=False,  # weight-only quantization: no bias buffer (the quantized forward never uses one anyway)
            transform=weight_quant_result['transform'],
            e2m1=e2m1,
            scale_quant=scale_quant,
            weight_global_scale=weight_quant_result['global_scale'],
            activation_scale_scale=weight_quant_result['scale_scale'],  # activation reuses the weight's ss
        ).to(device=device)
        for e2m1, scale_quant in zip(e2m1_splits, scale_splits)
    ]
    quant_results: list[dict] = [
        {
            'transform': weight_quant_result['transform'].cpu(),
            'e2m1': e2m1.cpu(),
            'scale_quant': scale_quant.cpu(),
            'global_scale': weight_quant_result['global_scale'],  # already a python float (rtn_xxfp4 captures it via .item()), not a device tensor -> nothing to move, no .cpu()
        }
        for e2m1, scale_quant in zip(e2m1_splits, scale_splits)
    ]
    return quant_layers, quant_results


def maybe_first_element(x):
    # a transformer block returns either the hidden states tensor or a tuple led by it
    return x[0] if isinstance(x, Sequence) else x


class HessianWrapper:
    def __init__(self):
        self.hessian: torch.Tensor | None = None
        self.num_samples: int = 0

    def update(self, inp: torch.Tensor) -> None:
        inp = inp.flatten(end_dim=-2).contiguous()
        if self.hessian is None:
            self.hessian = torch.zeros(inp.size(-1), inp.size(-1), dtype=torch.float32, device=inp.device)
        accumulate_hessian(mat_hessian=self.hessian, mat_input=inp)  # X^T X
        self.num_samples += inp.size(0)


@torch.no_grad()
def calibrate_activation_global_scale(
        block: torch.nn.Module,
        layer_names: list[str],
        input_args: list[tuple],
        input_kwargs: list[dict],
        device: torch.device,
        activation_quantile: float,
) -> None:
    """
    Empirically set each nvfp4 QuantLinear's per-tensor activation_global_scale. An element of the activation
    clips iff |T x| > finfo(e4m3).max * scale_scale * gs, so we run the calibration data through the (already
    quantized) block and, per layer, accumulate a log-magnitude histogram of the TRANSFORMED input |T x| (exactly
    what the QuantLinear quantizes), then size gs so the activation_quantile of that magnitude lands at the clip
    threshold (the top 1 - q clips). A full-pass histogram -- not the Hessian -- is required because activations
    are heavy-tailed: the scale must track the real peaks, which the variance cannot predict. mxfp4 layers are
    skipped (they keep the inherited weight gs). The input hooks fire before each QuantLinear's own forward, so the
    measured magnitudes do not depend on the current (pre-calibration) gs.
    """
    e4m3: torch.dtype = torch.float8_e4m3fn
    nvfp4_layers: dict[str, QuantLinear] = {
        name: layer for name in layer_names if isinstance(layer := block.get_submodule(name), QuantLinear) and layer.scale_quant.dtype == e4m3
    }
    if not nvfp4_layers:
        return

    log2_lo, log2_hi, num_bins = -24., 24., 2048  # log2|T x| histogram range (covers ~6e-8 .. 1.7e7) and resolution
    counts: dict[str, torch.Tensor] = {name: torch.zeros(num_bins, dtype=torch.float64, device=device) for name in nvfp4_layers}
    totals: dict[str, int] = {name: 0 for name in nvfp4_layers}

    def make_hook(name: str):
        def hook(module: QuantLinear, args: tuple) -> None:
            x: torch.Tensor = args[0]
            # mirror QuantLinear._quantize_torch: transform the input the same way the forward will (y = T x)
            transform: torch.Tensor | None = None if module.transform is None else module.transform.to(dtype=x.dtype)
            if transform is not None and transform.dim() == 2:  # shared 2D -> per-block stack
                transform = transform.unsqueeze(0).expand(module.in_features // module.group_size, -1, -1)
            y: torch.Tensor = x if transform is None else apply_block_transform(x=x, transform=transform, is_inverse_transpose=False, high_dtype=torch.float32, round_dtype=x.dtype)
            magnitude: torch.Tensor = y.abs().flatten().clamp_min(2. ** log2_lo).log2().clamp(log2_lo, log2_hi)
            counts[name] += torch.histc(magnitude.to(dtype=torch.float32), bins=num_bins, min=log2_lo, max=log2_hi).to(dtype=torch.float64)
            totals[name] += magnitude.numel()
        return hook

    handles = [layer.register_forward_pre_hook(make_hook(name)) for name, layer in nvfp4_layers.items()]
    for inp_args, inp_kwargs in zip(input_args, input_kwargs):
        block(*to(inp_args, device=device), **to(inp_kwargs, device=device))
    for handle in handles:
        handle.remove()

    for name, layer in nvfp4_layers.items():
        target: torch.Tensor = torch.as_tensor(activation_quantile * totals[name], dtype=torch.float64, device=device)
        bin_index: int = min(int(torch.searchsorted(counts[name].cumsum(dim=0), target).item()), num_bins - 1)
        quantile_magnitude: float = 2. ** (log2_lo + (bin_index + 1) * (log2_hi - log2_lo) / num_bins)  # upper edge of the crossing bin
        clip_divisor: float = torch.finfo(e4m3).max * layer.activation_scale_scale.item()  # finfo(e4m3).max * ss == 448 * 6 == 2688
        layer.activation_global_scale.fill_(quantile_magnitude / clip_divisor)


@torch.no_grad()
def quantize_model(
        model,
        calibration_data: list[torch.Tensor],
        quantize_format_name: str,
        transform_type: str,
        dampen_ratio: float,
        activation_quantile: float | None,
        force_rtn: bool,
        device: torch.device,
        cpu_offload_modules: bool,
        cpu_offload_activations: bool,
) -> dict[str, dict]:
    print('Quantization started.')
    quantized_state_dict: dict[str, dict] = {}
    act_offload_device: torch.device = torch.device('cpu') if cpu_offload_activations else device
    blocks: torch.nn.ModuleList = model.model.layers

    # catch the inputs to the first transformer block with a pre-hook that interrupts the forward, so the partial forward never reaches any block past the first one
    class ForwardInterrupt(Exception):
        pass

    def catcher_hook(_, block_args, block_kwargs):
        raise ForwardInterrupt(block_args, block_kwargs)

    catcher_handle: torch.utils.hooks.RemovableHandle = blocks[0].register_forward_pre_hook(catcher_hook, prepend=True, with_kwargs=True)

    # the partial forward up to blocks[0] needs the non-block submodules (embeddings, rotary, ...) on-device; stream them in while the rest of the model stays cpu-offloaded
    auxiliary_layers: dict[str, torch.nn.Module] = {k: v for k, v in model.model.named_children() if k != 'layers'}
    if cpu_offload_modules:
        for auxiliary_layer in auxiliary_layers.values():
            auxiliary_layer.to(device=device)

    input_args: list[tuple] = []
    input_kwargs: list[dict] = []
    for sample in calibration_data:
        try:
            model(sample.to(device=device))
        except ForwardInterrupt as caught:
            input_args.append(to(caught.args[0], device=act_offload_device))
            input_kwargs.append(to(caught.args[1], device=act_offload_device))

    catcher_handle.remove()
    if cpu_offload_modules:
        for auxiliary_layer in auxiliary_layers.values():
            auxiliary_layer.cpu()

    layer_id2name: dict[int, str] = {}
    hessian_wrappers: dict[str, HessianWrapper] = {}

    # one shared hook for all layers; dispatch to the right wrapper by module identity
    def hessian_hook(module, inp, __):
        hessian_wrappers[layer_id2name[id(module)]].update(inp[0])

    # Iterate over transformer blocks
    for block_idx, block in enumerate(blocks):
        print(f'Processing block {block_idx}...')
        if cpu_offload_modules:
            block.to(device=device)

        # define the layer groups to quantize jointly: fused projections for vllm, otherwise one linear at a time
        use_fused_layers: bool = False
        layer_sequences: list[list[str]] = [
            ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj'],
            ['self_attn.o_proj'],
            ['mlp.gate_proj', 'mlp.up_proj'],
            ['mlp.down_proj'],
        ]  # for vllm (fused)
        if not use_fused_layers:
            layer_sequences: list[list[str]] = [[v] for v in itertools.chain.from_iterable(layer_sequences)]  # for non-vllm
            print('Layer sequences:', layer_sequences)

        # register the hessian hooks, then run the calibration data through the block to accumulate the hessians and reference outputs
        hessian_handles: dict[str, torch.utils.hooks.RemovableHandle] = {}
        for layer_name, *_ in layer_sequences:
            layer: torch.nn.Module = block.get_submodule(layer_name)
            layer_id2name[id(layer)] = layer_name
            hessian_wrappers[layer_name] = HessianWrapper()
            hessian_handles[layer_name] = layer.register_forward_hook(hessian_hook)
        output_ref: list[torch.Tensor] = []
        for inp_args, inp_kwargs in zip(input_args, input_kwargs):
            output_ref.append(maybe_first_element(block(*to(inp_args, device=device), **to(inp_kwargs, device=device))).to(device=act_offload_device))
        for h in hessian_handles.values():
            h.remove()

        # try quantization with different configs
        configs: list[tuple] = list(itertools.product(
            [('force_rtn', v) for v in [force_rtn]],
            [('transform_type', v) for v in [transform_type]],
        ))
        print(configs)
        quant_layers: dict[tuple, dict[str, torch.nn.Module]] = {config: {} for config in configs}
        quant_results: dict[tuple, dict[str, dict]] = {config: {} for config in configs}
        for layer_sequence in layer_sequences:
            hessian: torch.Tensor = hessian_wrappers[layer_sequence[0]].hessian / hessian_wrappers[layer_sequence[0]].num_samples  # mean Hessian: the accumulated X^T X divided by the number of calibration rows
            weights = [block.get_submodule(layer_name).weight for layer_name in layer_sequence]
            for config in configs:
                quant_sequence_layers, quant_sequence_results = quantize_layers(
                    *weights,
                    hessian=hessian,
                    quantize_format_name=quantize_format_name,
                    dampen_ratio=dampen_ratio,
                    **dict(config),
                )
                quant_layers[config].update({layer_name: qlinear_layer for layer_name, qlinear_layer in zip(layer_sequence, quant_sequence_layers)})
                quant_results[config].update({layer_name: quant_result for layer_name, quant_result in zip(layer_sequence, quant_sequence_results)})

        # Score each config by the block's output reconstruction error against the fp reference, and -- in the
        # SAME forward pass -- capture that config's outputs (which are exactly the next-block inputs). The best
        # config's captured outputs are then reused for propagation, so the metric and the data come from one pass
        # (no separate propagation forward). Only the best-so-far outputs are kept, so memory stays at one output
        # set regardless of the number of configs.
        best_config: tuple | None = None
        best_l2: float = float('inf')
        best_outputs: list[torch.Tensor] | None = None
        for config in configs:
            for layer_name, quant_layer in quant_layers[config].items():
                block.set_submodule(layer_name, quant_layer)
            # opt-in: calibrate THIS config's nvfp4 activation global scales BEFORE scoring, so both the L2 metric
            # and the captured outputs reflect the calibrated model. Default (None) keeps the inherited weight gs
            # and runs no extra pass. Configs own distinct QuantLinear objects, so this never leaks across configs.
            if activation_quantile is not None:
                calibrate_activation_global_scale(
                    block=block,
                    layer_names=list(quant_layers[config]),
                    input_args=input_args,
                    input_kwargs=input_kwargs,
                    device=device,
                    activation_quantile=activation_quantile,
                )
            _losses: list[torch.Tensor] = []
            _outputs: list[torch.Tensor] = []
            for inp_args, inp_kwargs, out_ref in zip(input_args, input_kwargs, output_ref):
                out: torch.Tensor = maybe_first_element(block(*to(inp_args, device=device), **to(inp_kwargs, device=device)))
                _losses.append(torch.linalg.vector_norm(out - out_ref.to(device=out.device), dtype=torch.float32).pow(2.) / out_ref.numel())
                _outputs.append(out.to(device=act_offload_device))
            l2_loss: float = torch.stack(_losses).mean().item()
            print('l2_loss', config, l2_loss)
            if l2_loss < best_l2:  # strict '<' keeps the first min, matching the previous min(l2_losses, key=...)
                best_config, best_l2, best_outputs = config, l2_loss, _outputs

        # install the best config and record its results (its gs is already calibrated on its own QuantLinear
        # objects, so re-installing restores the exact state its outputs were captured under)
        best_quant_layers: dict[str, torch.nn.Module] = quant_layers[best_config]
        best_quant_results: dict[str, dict] = quant_results[best_config]
        print('best_config', best_config, 'l2_loss', best_l2)
        for layer_name, quant_layer in best_quant_layers.items():
            block.set_submodule(layer_name, quant_layer)
            quantized_state_dict[f'model.layers.{block_idx}.{layer_name}'] = best_quant_results[layer_name]

        # propagate the quantized block's outputs as the inputs to the next block -- reuse the best config's
        # outputs captured during scoring (no extra forward pass)
        for inp_args, inp_kwargs, out in zip(input_args, input_kwargs, best_outputs):
            if len(inp_args) > 0:
                inp_args[0].data = out
            elif 'hidden_states' in inp_kwargs:
                inp_kwargs['hidden_states'] = out
            else:
                raise ValueError('Unsupported block input format.')

        layer_id2name.clear()
        hessian_wrappers.clear()
        if cpu_offload_modules:
            block.cpu()
        clear_device_cache(garbage_collection=True)

    print('Quantization finished.')
    return quantized_state_dict
