# ParamShrink

ParamShrink is an experimental PyTorch module for generating two-dimensional weight matrices from Fourier features and a small trainable neural network. Despite its name and compression-oriented API, the current implementation does not fit that generator to a model's supplied weights, so it does not yet perform faithful model compression or restoration.

## Status

The repository contains a working weight-generation module, a wrapper, and unit tests. It is not a complete or validated model-compression workflow: there is no fitting/training routine for target model weights, and no reconstruction-quality or task-accuracy evaluation is included. The project metadata labels the release as Beta.

## Implemented

- `ParamShrink` maps a two-coordinate grid through fixed random Fourier projections, sine/cosine features, and a trainable SiLU multilayer perceptron to generate 2D weight matrices. Large matrices are processed in chunks.
- `compress()` serializes the shrinker's Fourier buffer, scale, network weights, and configuration. `decompress()` loads those generator values and generates tensors for supplied target shapes: matrices for 2D shapes and zero-filled vectors for 1D shapes.
- `shrink_model()` / `restore_model()` and `ParamShrinkWrapper` wrap that serialization and shape-based generation; they do not load a restored state dictionary into a model.
- `count_parameters()`, `get_compression_ratio()`, a configuration dataclass, and a factory function are provided.

## Important limitations

- `compress(state_dict)` does not use or fit to `state_dict`. The returned data represents the shrinker's current generator, not the source model's weights. As the source docstring notes, useful reconstruction would first require training the generator to fit those weights; that fitting process is not implemented.
- `compression_ratio` is stored and reported as a setting but does not change the generator architecture or perform compression.
- Restoration handles only 2D matrices and 1D vectors. Other target ranks are skipped, and 1D outputs are zeros rather than learned model biases.
- `restore_model()` returns a dictionary of generated tensors, not a restored model or a quality-checked checkpoint.
- The repository's `pyproject.toml` declares package metadata, but a wheel built from the current source was verified to contain no `param_shrink.py` module. Installing the built distribution therefore does not provide the importable implementation; use the source checkout until packaging is corrected.
- In a local run of the included example, the example model had 40,300 parameters and the reported shrinker state had 49,794 parameters, for a displayed ratio of 0.8×. This is a parameter-count comparison, not a reconstruction benchmark, and in this example the generator has more parameters than the source model.
- No reconstruction-error, downstream accuracy, memory, or speed measurements are provided.

## Use from a source checkout

Requires Python 3.10 or later, PyTorch 2.0 or later, and NumPy 1.24 or later. From the repository root, install the listed dependencies and run the included demonstration:

```bash
python -m pip install -r requirements.txt
python param_shrink.py
```

The demonstration exercises weight generation and the wrapper's serialization path; it does not train the generator or demonstrate accurate weight reconstruction. The module can also be imported from the checkout, for example:

```python
import torch
from param_shrink import ParamShrink

shrinker = ParamShrink(num_freqs=64, hidden_size=128)
weight = shrinker._generate_weight_matrix(100, 100)
print(weight.shape)
```

`_generate_weight_matrix` is an internal method; the repository does not currently expose a stable high-level API for fitting or faithfully compressing a model.

## Tests and CI

On the cloned repository commit, `python3 -m pytest -q` completed with **24 passed**. The included `param_shrink.py` demonstration also ran successfully; its output reported the 0.8× parameter-count ratio noted above. The GitHub Actions workflow is configured to run pytest and a compile/smoke check on Python 3.10, 3.11, and 3.12. A remote CI run result was not verified for this commit.

## License

MIT License. See [LICENSE](LICENSE).
