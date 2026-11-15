"""
Tests for ParamShrink module
"""

import pytest
import torch
import torch.nn as nn
from param_shrink import (
    ParamShrink,
    ParamShrinkWrapper,
    ParamShrinkConfig,
    create_param_shrink
)


class TestParamShrinkInitialization:
    """Test ParamShrink initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        shrinker = ParamShrink()
        assert shrinker.num_freqs == 64
        assert shrinker.hidden_size == 128
        assert shrinker.compression_ratio == 10.0
        assert shrinker.chunk_size == 16384
    
    def test_custom_initialization(self):
        """Test custom initialization"""
        shrinker = ParamShrink(
            num_freqs=32,
            hidden_size=64,
            compression_ratio=100.0,
            chunk_size=8192
        )
        assert shrinker.num_freqs == 32
        assert shrinker.hidden_size == 64
        assert shrinker.compression_ratio == 100.0
        assert shrinker.chunk_size == 8192
    
    def test_buffer_registration(self):
        """Test that B is registered as buffer"""
        shrinker = ParamShrink()
        assert hasattr(shrinker, 'B')
        assert isinstance(shrinker.B, torch.Tensor)
        assert shrinker.B.shape == (64, 2)
    
    def test_parameter_count(self):
        """Test parameter counting"""
        shrinker = ParamShrink(num_freqs=32, hidden_size=64)
        count = shrinker.count_parameters()
        # 32*2 input + 4 layers (64, 64, 64, 1) + scale
        expected_params = (
            32 * 2 * 64 + 64 +  # Layer 1
            64 * 64 + 64 +      # Layer 2
            64 * 64 + 64 +      # Layer 3
            64 * 1 + 1 +        # Layer 4
            1                   # Scale
        )
        assert count == expected_params


class TestParamShrinkWeightGeneration:
    """Test weight generation"""
    
    def test_generate_small_matrix(self):
        """Test generating small weight matrix"""
        shrinker = ParamShrink()
        W = shrinker._generate_weight_matrix(10, 10)
        assert W.shape == (10, 10)
        assert W.dtype == torch.float32
    
    def test_generate_large_matrix(self):
        """Test generating large weight matrix"""
        shrinker = ParamShrink(chunk_size=16384)
        W = shrinker._generate_weight_matrix(100, 100)
        assert W.shape == (100, 100)
    
    def test_generate_matrix_device(self):
        """Test weight generation on CPU"""
        shrinker = ParamShrink()
        W = shrinker._generate_weight_matrix(10, 10, device=torch.device('cpu'))
        assert W.device.type == 'cpu'
    
    def test_weight_matrix_properties(self):
        """Test weight matrix statistical properties"""
        shrinker = ParamShrink()
        W = shrinker._generate_weight_matrix(100, 100)
        # Check that weights are not all zeros
        assert W.abs().sum() > 0
        # Check that weights have reasonable scale
        assert W.std() > 0


class TestParamShrinkChunkedProcessing:
    """Test chunked processing"""
    
    def test_chunked_vs_unchunked(self):
        """Test that chunked and unchunked produce consistent results"""
        shrinker1 = ParamShrink(chunk_size=10000)
        shrinker2 = ParamShrink(chunk_size=100)
        
        # Copy parameters from shrinker1 to shrinker2
        shrinker2.net.load_state_dict(shrinker1.net.state_dict())
        shrinker2.B.data = shrinker1.B.data.clone()
        shrinker2.scale.data = shrinker1.scale.data.clone()
        
        W1 = shrinker1._generate_weight_matrix(100, 100)
        W2 = shrinker2._generate_weight_matrix(100, 100)
        
        # Should be close (may differ slightly due to chunking)
        assert torch.allclose(W1, W2, rtol=1e-4, atol=1e-4)
    
    def test_large_matrix_chunking(self):
        """Test that large matrices are processed correctly"""
        shrinker = ParamShrink(chunk_size=100)
        W = shrinker._generate_weight_matrix(500, 500)
        assert W.shape == (500, 500)
        # Check consistency
        assert not torch.isnan(W).any()
        assert not torch.isinf(W).any()


class TestParamShrinkCompression:
    """Test compression and decompression"""
    
    def test_compress_decompress(self):
        """Test basic compress/decompress cycle"""
        shrinker = ParamShrink(num_freqs=32, hidden_size=64)
        
        # Create a simple state dict
        state_dict = {
            'weight1': torch.randn(10, 10),
            'bias1': torch.randn(10),
        }
        
        # Compress
        compressed = shrinker.compress(state_dict)
        assert 'B' in compressed
        assert 'net' in compressed
        assert 'scale' in compressed
        assert 'config' in compressed
    
    def test_decompress_shapes(self):
        """Test decompression with shape information"""
        shrinker = ParamShrink(num_freqs=32, hidden_size=64)
        
        target_shapes = {
            'weight1': (10, 10),
            'bias1': (10,),
        }
        
        # Create dummy compressed state
        compressed = shrinker.compress({})
        
        # Decompress
        reconstructed = shrinker.decompress(compressed, target_shapes)
        
        assert reconstructed['weight1'].shape == (10, 10)
        assert reconstructed['bias1'].shape == (10,)


class TestParamShrinkModel:
    """Test model shrinking"""
    
    def test_shrink_model(self):
        """Test shrinking a model"""
        shrinker = ParamShrink()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        compressed = shrinker.shrink_model(model)
        assert 'B' in compressed
        assert 'net' in compressed
        assert 'shapes' in compressed
        assert 'model_class' in compressed
    
    def test_restore_model(self):
        """Test restoring a model"""
        shrinker = ParamShrink()
        
        # Create and shrink a model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        compressed = shrinker.shrink_model(model)
        reconstructed = shrinker.restore_model(compressed)
        
        # Check that we get weights back
        assert len(reconstructed) > 0


class TestParamShrinkWrapper:
    """Test ParamShrinkWrapper"""
    
    def test_shrink(self):
        """Test shrinking through wrapper"""
        wrapper = ParamShrinkWrapper()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        compressed = wrapper.shrink(model)
        assert 'B' in compressed
        assert 'net' in compressed
    
    def test_restore(self):
        """Test restoring through wrapper"""
        wrapper = ParamShrinkWrapper()
        
        # Create and shrink a model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        compressed = wrapper.shrink(model)
        reconstructed = wrapper.restore(compressed)
        
        assert len(reconstructed) > 0


class TestCreateParamShrink:
    """Test create_param_shrink factory function"""
    
    def test_default_creation(self):
        """Test default shrinker creation"""
        shrinker = create_param_shrink()
        assert isinstance(shrinker, ParamShrink)
        assert shrinker.num_freqs == 64
        assert shrinker.hidden_size == 128
    
    def test_custom_creation(self):
        """Test custom shrinker creation"""
        shrinker = create_param_shrink(
            num_freqs=32,
            hidden_size=64,
            compression_ratio=100.0,
            chunk_size=8192
        )
        assert shrinker.num_freqs == 32
        assert shrinker.hidden_size == 64


class TestParamShrinkConfig:
    """Test ParamShrinkConfig"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = ParamShrinkConfig()
        assert config.num_freqs == 64
        assert config.hidden_size == 128
        assert config.compression_ratio == 10.0
        assert config.chunk_size == 16384
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ParamShrinkConfig(
            num_freqs=32,
            hidden_size=64,
            compression_ratio=100.0
        )
        assert config.num_freqs == 32
        assert config.hidden_size == 64
        assert config.compression_ratio == 100.0


class TestCompressionRatio:
    """Test compression ratio calculations"""
    
    def test_compression_ratio(self):
        """Test compression ratio calculation"""
        shrinker = ParamShrink(num_freqs=64, hidden_size=128)
        
        # Shrinker has: 64*2*128 + 128 + 128*128 + 128 + 128*1 + 1 + 1
        shrink_params = shrinker.count_parameters()
        
        # For a 1M parameter model
        original_params = 1_000_000
        ratio = shrinker.get_compression_ratio(original_params)
        
        expected_ratio = original_params / shrink_params
        assert abs(ratio - expected_ratio) < 0.01


class TestGradientFlow:
    """Test gradient flow through shrinker"""
    
    def test_gradients_exist(self):
        """Test that gradients flow through shrinker"""
        shrinker = ParamShrink()
        
        # Generate weights with gradient tracking
        W = shrinker._generate_weight_matrix(10, 10)
        loss = W.sum()
        loss.backward()
        
        # Check that parameters have gradients
        assert shrinker.scale.grad is not None
        for param in shrinker.net.parameters():
            assert param.grad is not None


class TestDeviceAndDtype:
    """Test device and dtype handling"""
    
    def test_device_consistency(self):
        """Test that all tensors are on same device"""
        shrinker = ParamShrink()
        W = shrinker._generate_weight_matrix(10, 10)
        assert W.device == shrinker.B.device
        assert W.device == shrinker.scale.device
        for param in shrinker.net.parameters():
            assert param.device == shrinker.B.device
    
    def test_dtype_consistency(self):
        """Test that all tensors have same dtype"""
        shrinker = ParamShrink()
        W = shrinker._generate_weight_matrix(10, 10)
        assert W.dtype == shrinker.B.dtype
        assert W.dtype == shrinker.scale.dtype


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
