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

import unittest
import torch
import numpy as np
from scipy.linalg import hadamard

import qutlass
from qutlass import matmul_mxf8_bf16_tn, matmul_mxf8_bf16_nn
from qutlass.utils import to_blocked

# The "nn" (column-major x column-major) FP8 GEMM is only compiled for sm_100
# (Blackwell datacenter, e.g. B100/B200); see the `#if TARGET_CUDA_ARCH == 100`
# guard in qutlass/csrc/gemm.cu. On any other GPU it raises
# "Unsupported CUDA arch", so the nn case is skipped there.
_MXF8_NN_SUPPORTED = (
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)
)

@torch.compile(fullgraph=True)
def _pseudoquant_mxfp8(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    x = x.reshape(-1, 32)

    absmax = x.abs().max(dim=-1, keepdim=True).values
    shared_exps = torch.where(
        absmax > 0,
        torch.log2(x.abs().max(dim=-1, keepdim=True).values).floor().to(torch.uint8)
        - 8
        + 128,
        128,
    ).view(torch.float8_e8m0fnu)

    xq = torch.clamp(x / shared_exps.to(x.dtype), -448.0, 448.0).to(torch.float8_e4m3fn)

    xdq = xq.to(x.dtype) * shared_exps.to(x.dtype)

    return xdq.reshape(
        orig_shape
    ), (xq.reshape(orig_shape), shared_exps.reshape(orig_shape[:-1] + (orig_shape[-1] // 32,)))

@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for these tests")
class Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = 0
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        cls.dtype = torch.bfloat16
        cls.device = torch.device("cuda")

    def run_problem_tn(self, m, n, k):
        print(m, n, k)

        a = torch.rand(m, k, dtype=self.dtype, device=self.device) * 25.
        b = torch.rand(n, k, dtype=self.dtype, device=self.device) * 25.
        alpha = torch.Tensor([1.]).to(self.device)

        a_dq, (a_e4m3, a_e8m0) = _pseudoquant_mxfp8(a)
        b_dq, (b_e4m3, b_e8m0) = _pseudoquant_mxfp8(b)

        out_ref = a_dq @ b_dq.transpose(-2, -1).to(dtype=a_dq.dtype)

        a_scale_block = to_blocked(a_e8m0, True)
        b_scale_block = to_blocked(b_e8m0, True)

        out = matmul_mxf8_bf16_tn(a_e4m3, b_e4m3, a_scale_block, b_scale_block, alpha)

        torch.testing.assert_close(out, out_ref.to(dtype=out.dtype), atol=1e-1, rtol=1e-1)

    def run_problem_nn(self, m, n, k):
        print(m, n, k)

        a = torch.randn(m, k, dtype=self.dtype, device=self.device) * 25.
        b = torch.randn(n, k, dtype=self.dtype, device=self.device) * 25.
        alpha = torch.Tensor([1.]).to(self.device)

        a_dq, (a_e4m3, a_e8m0) = _pseudoquant_mxfp8(a)
        b_dq, (b_e4m3, b_e8m0) = _pseudoquant_mxfp8(b)

        out_ref = a_dq @ b_dq.to(dtype=a_dq.dtype).transpose(-1, -2)

        a_scale_block = to_blocked(a_e8m0, True)
        b_scale_block = to_blocked(b_e8m0, True)

        a_e4m3 = a_e4m3.T.contiguous().view((k,m))

        out = matmul_mxf8_bf16_nn(a_e4m3, b_e4m3, a_scale_block, b_scale_block, alpha)

        torch.testing.assert_close(out, out_ref.to(dtype=out.dtype), atol=1e-1, rtol=1e-1)

    def _llama_shapes(self):
        MODELS = {
            ' 7B': [
                (4096, 3 * 4096),
                (4096, 4096),
                (4096, 2 * 10752),
                (10752, 4096)
            ],
            '13B': [
                (5120, 3 * 5120),
                (5120, 5120),
                (5120, 2 * 13568),
                (13568, 5120)
            ],
            '33B': [
                (6656, 3 * 6656),
                (6656, 6656),
                (6656, 2 * 17664),
                (17664, 6656)
            ],
            '70B': [
                (8192, 3 * 8192),
                (8192, 8192),
                (8192, 2 * 21760),
                (21760, 8192)
            ]
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for batch in [16]:
                    yield batch, layer[1], layer[0]

    def test_llama_shapes_tn(self):
        print()
        for m, n, k in self._llama_shapes():
            self.run_problem_tn(m, n, k)

    @unittest.skipUnless(
        _MXF8_NN_SUPPORTED,
        "matmul_mxf8_bf16_nn requires sm_100 (Blackwell datacenter); "
        "not built for this GPU",
    )
    def test_llama_shapes_nn(self):
        print()
        for m, n, k in self._llama_shapes():
            self.run_problem_nn(m, n, k)

if __name__ == '__main__':
    unittest.main()
