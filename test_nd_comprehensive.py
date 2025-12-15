"""
test_nd_comprehensive.py

Comprehensive test suite for all dimensions (1D, 2D, 3D) and all tasks.

Tests:
- Classification
- Regression
- Segmentation
- Map-to-map image translation
- Radiomics features
- Extra parameters
- Skip connection alignment
"""

import torch
import pytest
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from models import (
    EMUNet, EMUNetPP, EMUNetMapToMap, EMUNetPPMapToMap,
    SkipConnectionAligner, calculate_fos_features, calculate_simple_glcm_features
)
from unet_1d_optimized import UNet1DOptimized, EMUNet1D


class TestDimensionSupport:
    """Test basic dimension support across all models."""
    
    @pytest.mark.parametrize("dimension,input_shape", [
        (1, (2, 3, 128)),
        (2, (2, 3, 64, 64)),
        (3, (2, 3, 32, 32, 32)),
    ])
    def test_emunet_dimensions(self, dimension, input_shape):
        """Test EMUNet with 1D, 2D, 3D inputs."""
        model = EMUNet(
            in_channels=3, out_channels=10,
            dimension=dimension, task='classification'
        )
        x = torch.randn(*input_shape)
        output = model(x)
        assert output.shape == (2, 10), f"Expected shape (2, 10), got {output.shape}"
    
    @pytest.mark.parametrize("dimension,input_shape", [
        (1, (2, 1, 256)),
        (2, (2, 1, 128, 128)),
        (3, (2, 1, 64, 64, 64)),
    ])
    def test_emunetpp_dimensions(self, dimension, input_shape):
        """Test EMUNet++ with all dimensions."""
        model = EMUNetPP(
            in_channels=1, out_channels=5,
            dimension=dimension, task='regression'
        )
        x = torch.randn(*input_shape)
        output = model(x)
        assert output.shape == (2, 5), f"Expected shape (2, 5), got {output.shape}"


class TestTaskSupport:
    """Test all task types across dimensions."""
    
    @pytest.mark.parametrize("task", ['classification', 'regression', 'segmentation', 'map-to-map'])
    def test_emunet_2d_tasks(self, task):
        """Test EMUNet with all task types in 2D."""
        if task == 'map-to-map':
            model = EMUNetMapToMap(
                in_channels=1, out_channels=1, dimension=2
            )
            x = torch.randn(2, 1, 64, 64)
            output = model(x)
            assert output.shape == (2, 1, 64, 64)
        else:
            model = EMUNet(
                in_channels=1, out_channels=5 if task == 'classification' else 1,
                dimension=2, task=task
            )
            x = torch.randn(2, 1, 64, 64)
            output = model(x)
            if task == 'segmentation':
                assert output.shape == (2, 5, 64, 64)
            else:
                assert output.shape[0] == 2
    
    @pytest.mark.parametrize("task,output_channels", [
        ('classification', 3),
        ('regression', 1),
        ('map-to-map', 1),
    ])
    def test_emunet_1d_tasks(self, task, output_channels):
        """Test 1D EMUNet with different tasks."""
        if task == 'map-to-map':
            model = UNet1DOptimized(
                in_channels=1, out_channels=1, task='map-to-map'
            )
            x = torch.randn(2, 1, 256)
            output = model(x)
            assert output.shape == (2, 1, 256)
        else:
            model = EMUNet1D(
                in_channels=1, out_channels=output_channels,
                task=task
            )
            x = torch.randn(2, 1, 256)
            output = model(x)
            assert output.shape[0] == 2


class TestRadiomicsFeatures:
    """Test radiomics feature computation across dimensions."""
    
    def test_fos_features_1d(self):
        """Test first-order statistics on 1D signal."""
        x = torch.randn(100)
        fos = calculate_fos_features(x)
        assert fos.shape[0] >= 20  # Should have at least 20 features
    
    def test_fos_features_2d(self):
        """Test first-order statistics on 2D image."""
        x = torch.randn(64, 64)
        fos = calculate_fos_features(x)
        assert fos.shape[0] >= 20
    
    def test_fos_features_3d(self):
        """Test first-order statistics on 3D volume."""
        x = torch.randn(32, 32, 32)
        fos = calculate_fos_features(x)
        assert fos.shape[0] >= 20
    
    @pytest.mark.parametrize("dimension", [1, 2, 3])
    def test_glcm_features_nd(self, dimension):
        """Test GLCM-like features across dimensions."""
        shapes = {1: (100,), 2: (64, 64), 3: (32, 32, 32)}
        x = torch.randn(*shapes[dimension])
        
        glcm = calculate_simple_glcm_features(x, radii=[1, 2], dimension=dimension)
        
        # Should have features for each radius and direction
        expected_directions = 1 if dimension == 1 else 2 if dimension == 2 else 3
        expected_features = 2 * expected_directions * 3  # 2 radii, directions, 3 GLCM features
        
        assert glcm.shape[0] == expected_features, \
            f"Expected {expected_features} features, got {glcm.shape[0]} for {dimension}D"
    
    def test_radiomics_in_model_2d(self):
        """Test radiomics computation in model forward pass."""
        model = EMUNet(
            in_channels=2, out_channels=3, dimension=2,
            task='classification', use_radiomics=True
        )
        x = torch.randn(2, 2, 64, 64)
        output = model(x)
        assert output.shape == (2, 3)
    
    def test_radiomics_in_model_3d(self):
        """Test radiomics in 3D model."""
        model = EMUNet(
            in_channels=1, out_channels=5, dimension=3,
            task='regression', use_radiomics=True, num_bins=32
        )
        x = torch.randn(2, 1, 32, 32, 32)
        output = model(x)
        assert output.shape == (2, 5)


class TestExtraParameters:
    """Test models with extra parameters (age, TR, TE, etc.)."""
    
    @pytest.mark.parametrize("dimension", [1, 2, 3])
    def test_extra_params_classification(self, dimension):
        """Test extra parameters in classification."""
        shapes = {1: (2, 1, 128), 2: (2, 1, 64, 64), 3: (2, 1, 32, 32, 32)}
        
        model = EMUNet(
            in_channels=1, out_channels=4, dimension=dimension,
            task='classification', extra_params_dim=3
        )
        x = torch.randn(*shapes[dimension])
        extra = torch.randn(2, 3)
        
        output = model(x, extra)
        assert output.shape == (2, 4)
    
    @pytest.mark.parametrize("dimension", [1, 2, 3])
    def test_extra_params_segmentation(self, dimension):
        """Test extra parameters in segmentation."""
        shapes = {1: (2, 1, 128), 2: (2, 1, 64, 64), 3: (2, 1, 32, 32, 32)}
        output_shapes = {1: (2, 5, 128), 2: (2, 5, 64, 64), 3: (2, 5, 32, 32, 32)}
        
        model = EMUNet(
            in_channels=1, out_channels=5, dimension=dimension,
            task='segmentation', extra_params_dim=2
        )
        x = torch.randn(*shapes[dimension])
        extra = torch.randn(2, 2)
        
        output = model(x, extra)
        assert output.shape == output_shapes[dimension]


class TestMapToMap:
    """Test image-to-image translation models."""
    
    @pytest.mark.parametrize("in_c,out_c,dimension,shape", [
        (1, 1, 1, (2, 1, 256)),
        (3, 3, 2, (2, 3, 64, 64)),
        (1, 1, 3, (2, 1, 32, 32, 32)),
    ])
    def test_emunet_maptomap(self, in_c, out_c, dimension, shape):
        """Test EMUNetMapToMap across dimensions."""
        model = EMUNetMapToMap(
            in_channels=in_c, out_channels=out_c, dimension=dimension
        )
        x = torch.randn(*shape)
        output = model(x)
        assert output.shape == shape
    
    @pytest.mark.parametrize("in_c,out_c,dimension,shape", [
        (1, 1, 2, (2, 1, 64, 64)),
        (2, 2, 3, (2, 2, 32, 32, 32)),
    ])
    def test_emunetpp_maptomap(self, in_c, out_c, dimension, shape):
        """Test EMUNetPPMapToMap."""
        model = EMUNetPPMapToMap(
            in_channels=in_c, out_channels=out_c, dimension=dimension
        )
        x = torch.randn(*shape)
        output = model(x)
        assert output.shape == shape
    
    def test_maptomap_activation(self):
        """Test different activation functions in map-to-map."""
        activations = ['sigmoid', 'tanh', 'none']
        
        for act in activations:
            model = EMUNetMapToMap(
                in_channels=1, out_channels=1, dimension=2,
                activation_final=act
            )
            x = torch.randn(2, 1, 64, 64)
            output = model(x)
            
            if act == 'sigmoid':
                assert (output >= 0).all() and (output <= 1).all()
            elif act == 'tanh':
                assert (output >= -1).all() and (output <= 1).all()


class TestSkipConnectionAligner:
    """Test skip connection alignment strategies."""
    
    @pytest.mark.parametrize("strategy", ['pad', 'crop', 'interpolate'])
    def test_aligner_strategies_1d(self, strategy):
        """Test aligner with different strategies in 1D."""
        aligner = SkipConnectionAligner(strategy=strategy)
        encoder_feat = torch.randn(2, 64, 128)
        decoder_feat = torch.randn(2, 64, 64)
        
        aligned = aligner(encoder_feat, decoder_feat, dimension=1)
        assert aligned.shape == encoder_feat.shape
    
    @pytest.mark.parametrize("strategy", ['pad', 'crop', 'interpolate'])
    def test_aligner_strategies_2d(self, strategy):
        """Test aligner with different strategies in 2D."""
        aligner = SkipConnectionAligner(strategy=strategy)
        encoder_feat = torch.randn(2, 128, 64, 64)
        decoder_feat = torch.randn(2, 128, 32, 32)
        
        aligned = aligner(encoder_feat, decoder_feat, dimension=2)
        assert aligned.shape == encoder_feat.shape
    
    @pytest.mark.parametrize("strategy", ['pad', 'crop', 'interpolate'])
    def test_aligner_strategies_3d(self, strategy):
        """Test aligner with different strategies in 3D."""
        aligner = SkipConnectionAligner(strategy=strategy)
        encoder_feat = torch.randn(2, 64, 32, 32, 32)
        decoder_feat = torch.randn(2, 64, 16, 16, 16)
        
        aligned = aligner(encoder_feat, decoder_feat, dimension=3)
        assert aligned.shape == encoder_feat.shape


class TestUNet1D:
    """Test 1D-optimized U-Net models."""
    
    def test_unet1d_basic(self):
        """Test basic UNet1DOptimized creation and forward pass."""
        model = UNet1DOptimized(in_channels=1, out_channels=1, depth=3)
        x = torch.randn(2, 1, 256)
        output = model(x)
        assert output.shape == (2, 1, 256)
    
    def test_unet1d_dilated(self):
        """Test 1D U-Net with dilated convolutions."""
        model = UNet1DOptimized(
            in_channels=1, out_channels=1, depth=4,
            dilation_schedule=[1, 2, 4, 8]
        )
        x = torch.randn(2, 1, 512)
        output = model(x)
        assert output.shape == (2, 1, 512)
    
    def test_emunet1d_with_radiomics(self):
        """Test EMUNet1D with radiomics features."""
        model = EMUNet1D(
            in_channels=1, out_channels=3,
            task='classification', use_radiomics=True
        )
        x = torch.randn(2, 1, 256)
        output = model(x)
        assert output.shape == (2, 3)
    
    def test_emunet1d_with_extra_params(self):
        """Test EMUNet1D with extra parameters."""
        model = EMUNet1D(
            in_channels=2, out_channels=1,
            task='regression', extra_params_dim=2
        )
        x = torch.randn(2, 2, 256)
        extra = torch.randn(2, 2)
        output = model(x, extra)
        assert output.shape == (2, 1)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_invalid_dimension(self):
        """Test that invalid dimensions raise errors."""
        with pytest.raises(ValueError):
            EMUNet(in_channels=1, out_channels=1, dimension=4)
    
    def test_invalid_input_channels(self):
        """Test error on mismatched input channels."""
        model = EMUNet(in_channels=3, out_channels=1, dimension=2)
        x = torch.randn(2, 1, 64, 64)  # Only 1 channel, but model expects 3
        
        with pytest.raises(ValueError):
            model(x)
    
    def test_invalid_extra_params(self):
        """Test error on mismatched extra parameters."""
        model = EMUNet(
            in_channels=1, out_channels=1, dimension=2,
            extra_params_dim=3
        )
        x = torch.randn(2, 1, 64, 64)
        extra = torch.randn(2, 5)  # Wrong dimension
        
        with pytest.raises(ValueError):
            model(x, extra)
    
    def test_device_compatibility(self):
        """Test GPU compatibility (if available)."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = EMUNet(in_channels=1, out_channels=1, dimension=2).to(device)
        x = torch.randn(2, 1, 64, 64).to(device)
        output = model(x)
        
        assert output.device == device


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
