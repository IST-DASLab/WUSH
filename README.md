## WUSH
The official implementation for the paper [WUSH: Near-Optimal Adaptive Transforms for LLM Quantization](https://arxiv.org/abs/2512.00956). 


## Environment setup


```bash
#!/bin/bash
conda create -n wush python=3.12 ipykernel ipywidgets cmake --yes

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wush

pip install --pre torch==2.11.0.dev20260122+cu128 --index-url https://download.pytorch.org/whl/nightly/cu128
pip install numpy pandas datasets

git clone git@github.com:vahe1994/transformers.git && cd transformers && git checkout wush_tmp && pip install -e . && cd ..

git clone git@github.com:Dao-AILab/fast-hadamard-transform
cd fast-hadamard-transform
pip install -e .
cd ..


git clone git@github.com/isT-DASLab/qutlass
cd qutlass
# Installing QuTLASS (run from the repository root that contains the submodule)
mv mma_multistage.h third_party/cutlass/include/cutlass/gemm/threadblock/
mv mma_tensor_op.h third_party/cutlass/include/cutlass/gemm/warp/

pip install --no-build-isolation .

# Installing the fp-quant linear layer with WUSH support
cd ../WUSH/inference_lib
pip install -e .
cd ../../

pip install lm_eval==0.4.9
```

---

## How to run

The script to run WUSH quantization is in `scripts/wush.sh`.

```bash
#!/bin/bash
CUDA_VISIBLE_DEVICES=0 GPTQ=1 TRANSFORM_CLASS=wush bash scripts/wush.sh
```

## Kernel Benchmarks

```bash
python qutlass/benchmarks/bench_mxfp4_sm100.py
```

## Citation
```
@misc{chen2026wushnearoptimaladaptivetransforms,
      title={WUSH: Near-Optimal Adaptive Transforms for LLM Quantization}, 
      author={Jiale Chen and Vage Egiazarian and Roberto L. Castro and Torsten Hoefler and Dan Alistarh},
      year={2026},
      eprint={2512.00956},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.00956}, 
}
```

**Note:** The  repository is built on the codebase from the paper [Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization](https://github.com/IST-DASLab/FP-Quant/edit/master).