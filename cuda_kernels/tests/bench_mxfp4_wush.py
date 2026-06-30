#
# Copyright (C) 2025 Roberto L. Castro (Roberto.LopezCastro@ist.ac.at). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import torch
import triton
from scipy.linalg import hadamard

import qutlass
from qutlass import fusedQuantizeWushMx, fusedQuantizeMx
from qutlass.utils import to_blocked

PROVIDER_CFGS = {
    "torch-bf16": dict(enabled=True),
    "mxfp4-had": dict(transform="hadamard", enabled=True),
    "mxfp4-wush": dict(transform="cwush", enabled=True)
}

_enabled = [k for k, v in PROVIDER_CFGS.items() if v["enabled"]]


def get_hadamard_matrix(group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.tensor(
        hadamard(group_size) * group_size**-0.5, dtype=dtype, device=device
    )

def get_wush_matrix(K: int, group_size: int, dtype: torch.dtype, device: torch.device):
    return torch.randn(K//group_size, group_size, group_size, device="cuda", dtype=torch.bfloat16)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[
            1,
            4,
            8,
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            24576,
            32768,
            65536
        ],
        x_log=False,
        line_arg="provider",
        line_vals=_enabled,
        line_names=_enabled,
        ylabel="Latency [us] (smaller is better)",
        plot_name="BF16 vs MXFP4 GEMMs",
        args={},
    )
)
def benchmark(batch_size, provider, N, K, had_size):
    M = batch_size
    device = "cuda"
    dtype = torch.bfloat16

    a = torch.randn((M, K), device=device, dtype=dtype)

    forward_transform_matrix = get_wush_matrix(K, had_size, dtype, device)

    forward_hadamard_matrix = get_hadamard_matrix(had_size, dtype, device)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch-bf16":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.linear(a.view(-1, 32), forward_transform_matrix[0]), rep=200, quantiles=quantiles
        )
    elif provider == "mxfp4-wush":  
        forward_transform_matrix = forward_transform_matrix.view(-1, 32).T.contiguous()
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: fusedQuantizeWushMx(
                a, forward_transform_matrix
            ),
            rep=200, quantiles=quantiles
        )
    elif provider == "mxfp4-had":
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            lambda: fusedQuantizeMx(
                a, forward_hadamard_matrix, method="abs_max"
            ),
            rep=200, quantiles=quantiles
        )

    # Convert ms to us for better readability at small batch sizes
    to_us = lambda t_ms: t_ms * 1000
    return to_us(ms), to_us(max_ms), to_us(min_ms)


MODELS = {
    #'Llama7B': [
    #    (4096, 3 * 4096),
    #    (4096, 4096),
    #    (4096, 2 * 10752),
    #    (10752, 4096)
    # ],
    #'Llama13B': [
    #    (5120, 3 * 5120),
    #    (5120, 5120),
    #    (5120, 2 * 13568),
    #    (13568, 5120)
    # ],
    #'Llama33B': [
    #    (6656, 3 * 6656),
    #    (6656, 6656),
    #    (6656, 2 * 17664),
    #    (17664, 6656)
    # ],
    #'Llama65B': [
    #    (8192, 3 * 8192),
    #    (8192, 8192),
    #    (8192, 2 * 21760),
    #    (21760, 8192)
    # ],
    #'Qwen3-0.6B': ((1024, 2048), (1024, 1024), (1024, 6144), (3072, 1024)),
    #'Qwen3-1.7B': ((2048, 4096), (2048, 2048), (2048, 12288), (6144, 2048)),
    #'Qwen3-4B': ((2560, 2560), (2560, 2560), (2560, 19456), (9728, 2560)),
    'Qwen3-8B': [(4096, 4096), (4096, 4096), (4096, 24576), (12288, 4096)],
    'Qwen3-14B': [(5120, 5120), (5120, 5120), (5120, 34816), (17408, 5120)],
    #'Qwen3-32B': [(5120, 5120), (5120, 51200), (25600, 5120)],
    'Llama-3.1-70B': [(8192, 8192), (8192, 57344), (28672, 8192)]
    #"Llama-3.1-70B": [(8192, 57344)]
}

for model, layers in MODELS.items():
    for K, N in layers:
        for had_size in [32]:
            print(f"{model}, N={N} K={K}, HAD={had_size}, BF16 vs MXFP4 GEMMs latency [us]:")
            save_path = f"benchmarks_output/bench_mxfp4_res_n{N}_k{K}_had{had_size}_sm100"
            os.makedirs(save_path, exist_ok=True)
            benchmark.run(
                print_data=True,
                show_plots=True,
                save_path=save_path,
                N=N,
                K=K,
                had_size=had_size,
            )
