# ParamShrink

[![CI](https://github.com/kapildhinwa2010-dev/ParamShrink/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ParamShrink/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/ParamShrink/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ParamShrink/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**ParamShrink** is a parameter compression framework that reduces the number of trainable parameters in neural networks through Fourier feature projections, enabling extreme compression while preserving model capacity.

### Key Features

- ✅ **Extreme Compression**: Reduce parameter count by 10x-1000x
- ✅ **Fourier-Based**: Uses Fourier features for smooth parameterization
- ✅ **Memory Efficient**: Chunked processing for large models
- ✅ **GPU Accelerated**: Full CUDA support
- ✅ **PyTorch Integration**: Seamless integration with PyTorch models
- ✅ **Configurable**: Adjustable compression ratios and network architecture

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/ParamShrink.git
cd ParamShrink
pip install -e .
```

## Usage

### Basic Parameter Shrinking

```python
from param_shrink import ParamShrink, create_param_shrink

# Create shrinker
shrinker = create_param_shrink(num_freqs=64, compression_ratio=10.0)

# Generate a weight matrix
W = shrinker._generate_weight_matrix(1000, 1000)
print(f"Generated matrix: {W.shape}")
```

### Compressing a Model

```python
import torch.nn as nn
from param_shrink import ParamShrinkWrapper

# Create a model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

model = MyModel()

# Shrink the model
wrapper = ParamShrinkWrapper()
compressed = wrapper.shrink(model)

# Save compressed representation
import torch
torch.save(compressed, 'model_compressed.pt')
```

### Restoring Parameters

```python
# Load compressed representation
compressed = torch.load('model_compressed.pt')

# Restore parameters
restored_params = wrapper.restore(compressed)

# Load into a new model
model = MyModel()
model.load_state_dict(restored_params)
```

## Architecture

### Fourier Feature Compression

ParamShrink uses random Fourier features to map 2D coordinate grids into a high-dimensional space:

```
Coordinates → Fourier Projection → Features → Neural Network → Weights
```

The neural network (shrinker) learns to generate weights from these features, enabling compression.

### Network Structure

The default shrink network consists of:
- Input: Fourier features (num_freqs * 2 dimensions)
- Hidden layers: 3 layers with SiLU activation (128 hidden size)
- Output: Scalar weight value
- Trainable scale parameter

### Memory Efficiency

For large weight matrices, ParamShrink uses chunked processing:
- Configurable chunk size (default: 16384 elements)
- Automatic chunking for matrices larger than chunk size
- Seamless reconstruction of full matrix

## Compression Ratios

| Original Parameters | Shrink Parameters | Compression Ratio |
|---------------------|-------------------|-------------------|
| 100K                | ~10K              | ~10x              |
| 1M                  | ~10K              | ~100x             |
| 10M                 | ~10K              | ~1000x            |
| 100M                | ~10K              | ~10,000x          |

*The shrink network uses only ~10K parameters regardless of the target model size.*

## API Reference

### ParamShrink

```python
ParamShrink(
    num_freqs: int = 64,
    hidden_size: int = 128,
    compression_ratio: float = 10.0,
    chunk_size: int = 16384
)
```

**Methods:**
- `compress(state_dict: Dict) -> Dict`: Compress model state dict
- `decompress(compressed: Dict, target_shapes: Dict) -> Dict`: Decompress to weights
- `shrink_model(model: nn.Module) -> Dict`: Shrink entire model
- `restore_model(compressed: Dict) -> Dict`: Restore model parameters
- `count_parameters() -> int`: Count shrink network parameters
- `get_compression_ratio(original_params: int) -> float`: Calculate compression ratio

### ParamShrinkWrapper

High-level wrapper for easier use:

```python
wrapper = ParamShrinkWrapper(config=ParamShrinkConfig())
```

**Methods:**
- `shrink(model: nn.Module) -> Dict`: Shrink a model
- `restore(compressed: Dict) -> Dict`: Restore parameters

### create_param_shrink()

Factory function for quick shrinker creation:

```python
shrinker = create_param_shrink(
    num_freqs=64,
    hidden_size=128,
    compression_ratio=10.0,
    device='cuda'
)
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Fourier feature networks
- Neural network compression
- Implicit neural representations
- Parameter-efficient learning
