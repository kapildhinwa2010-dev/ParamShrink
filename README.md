# ParamShrink

[![CI](https://github.com/kapildhinwa2010-dev/ParamShrink/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ParamShrink/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/ParamShrink/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/ParamShrink/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Fourier-feature parameterization that replaces weight matrices with a small generator network.

## What it does

ParamShrink maps 2D coordinate grids through random Fourier features and
a 3-layer SiLU MLP with a trainable scale, generating weight matrices
from roughly 10K generator parameters. compress()/decompress() wrap a
full PyTorch state dict.

## Installation

    pip install torch numpy pytest

## Usage

    from param_shrink import ParamShrinkWrapper

    wrapper = ParamShrinkWrapper()
    compressed = wrapper.shrink(model)
    restored = wrapper.restore(compressed)
    model.load_state_dict(restored)

## API

- ParamShrink(num_freqs=64, hidden_size=128, compression_ratio=10.0,
  chunk_size=16384): compress, decompress, shrink_model, restore_model,
  count_parameters, get_compression_ratio
- ParamShrinkWrapper / ParamShrinkConfig: shrink, restore
- create_param_shrink(...) factory

## Honest results and limitations

- The ratio table (10x to 10,000x) is parameter arithmetic only. No
  reconstruction-error or task-accuracy measurement backs it.
- Without fitting the generator to actual trained weights, decompress
  returns a smooth approximation - useful weights require per-matrix
  optimization, which is the real cost.
- compression_ratio is a config knob that controls generator sizing,
  not a measured outcome.
- No fine-tuning pipeline is included; accuracy after shrink+restore is
  unmeasured.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that Fourier-parameterized
  reconstructions of trained weights were not acceptable without
  per-layer training, and the related carrier-mode recall failed about
  half of noise trials. The robust design kept in the toolkit is the
  Reed-Solomon slot lattice (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
