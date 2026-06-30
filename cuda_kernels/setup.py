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

import torch.utils.cpp_extension as torch_cpp_ext
import os
import pathlib
import torch
import re

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent
torch_version = torch.__version__


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


def detect_cc():
    dev = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(dev)
    return major * 10 + minor


cc = detect_cc()


def get_cuda_arch_flags():
    flags = [
        "-gencode",
        "arch=compute_120a,code=sm_120a",
        "-gencode",
        "arch=compute_100a,code=sm_100a",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler",
        "-funroll-loops",
        "-Xcompiler",
        "-ffast-math",
        "-Xcompiler",
        "-finline-functions",
    ]
    return flags


def third_party_cmake():
    import subprocess
    import sys
    import shutil

    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("Cannot find CMake executable.")

    # Disable CUTLASS's library/test/example targets: their kernel generation
    # emits ~35k .cu files under third_party/cutlass/tools/library/generated/
    # that the qutlass CUDAExtension build never uses. Skipping it speeds up the
    # build and keeps the submodule working tree clean.
    build_dir = HERE / "build"                                                                                                              
    build_dir.mkdir(exist_ok=True)                                                                                                          
    retcode = subprocess.call([
        cmake, str(HERE),                                                                                                                   
        "-DCUTLASS_ENABLE_LIBRARY=OFF",
        "-DCUTLASS_ENABLE_TESTS=OFF",
        "-DCUTLASS_ENABLE_EXAMPLES=OFF",
        "-DCUTLASS_ENABLE_PROFILER=OFF",
    ], cwd=build_dir)
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.cuda.current_device()
    print(f"Current device: {torch.cuda.get_device_name(device)}")
    print(f"Current CUDA capability: {torch.cuda.get_device_capability(device)}")
    assert torch.cuda.get_device_capability(device)[0] >= 10, (
        f"CUDA capability must be >= 10.0, yours is {torch.cuda.get_device_capability(device)}"
    )

    print(f"PyTorch version: {torch_version}")
    m = re.match(r"^(\d+)\.(\d+)", torch_version)
    if not m:
        raise RuntimeError(f"Cannot parse PyTorch version '{torch_version}'")
    major, minor = map(int, m.groups())
    if major < 2 or (major == 2 and minor < 7):
        raise RuntimeError(f"PyTorch version must be >= 2.7, but found {torch_version}")

    third_party_cmake()
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        ext_modules=[
            CUDAExtension(
                name="qutlass._CUDA",
                sources=[
                    "qutlass/csrc/bindings.cpp",
                    "qutlass/csrc/gemm.cu",
                    "qutlass/csrc/gemm_ada.cu",
                    "qutlass/csrc/fused_quantize_mx.cu",
                    "qutlass/csrc/fused_quantize_wush_mx.cu",
                    "qutlass/csrc/fused_quantize_mx_mask.cu",
                    "qutlass/csrc/fused_quantize_nv.cu",
                    "qutlass/csrc/fused_quantize_mx_sm100.cu",
                    "qutlass/csrc/fused_quantize_nv_sm100.cu",
                    "qutlass/csrc/quartet_bwd_sm120.cu",
                ],
                include_dirs=[
                    # WUSH-patched CUTLASS headers. This dir mirrors the cutlass/
                    # include layout (cutlass_patch/cutlass/gemm/{threadblock,
                    # warp}/...) and MUST come before third_party/cutlass/include
                    # so it overrides the stock mma_multistage.h / mma_tensor_op.h
                    # via include precedence, leaving the third_party/cutlass
                    # submodule unmodified.
                    os.path.join(setup_dir, "qutlass/csrc/cutlass_patch"),
                    os.path.join(setup_dir, "qutlass/csrc/include"),
                    os.path.join(setup_dir, "qutlass/csrc/include/cutlass_extensions"),
                    os.path.join(setup_dir, "third_party/cutlass/include"),
                    os.path.join(setup_dir, "third_party/cutlass/tools/util/include"),
                ],
                define_macros=[("TARGET_CUDA_ARCH", str(cc))],
                extra_compile_args={
                    "cxx": ["-std=c++17"],
                    "nvcc": get_cuda_arch_flags(),
                },
                extra_link_args=[
                    "-lcudart",
                    "-lcuda",
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
