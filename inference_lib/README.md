# fp_quant

A library that wraps [`qutlass`](https://github.com/IST-DASLab/qutlass) kernels with linear layer wrappers for integrations into training and inference engines.

## Installation

```bash
pip install .
```

## Usage

```python
from fp_quant import replace_with_fp_quant_linear, FPQuantConfig

# Replace nn.Linear layers with fp_quant.FPQuantLinear
replace_with_fp_quant_linear(
    model,
    fp_quant_linear_config=FPQuantConfig(),
)
``` 