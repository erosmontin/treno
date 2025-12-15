"""
Treno - Deep Learning Architectures for Medical Imaging

This package provides neural network architectures and data loaders
for medical image analysis tasks.
"""

from .loaders import (
    # Modern classes (using pyable-dataloader)
    TrenoDataset,
    create_manifest_from_csv,
    save_manifest,
    load_manifest,
    create_treno_dataset_from_csv,
    # Flags
    PYABLE_DATALOADER_AVAILABLE,
)

# 1D-optimized models for time-series
try:
    from .unet_1d_optimized import (
        UNet1DOptimized,
        EMUNet1D,
        DilatedConv1dBlock,
        TemporalPooling,
    )
    _1D_AVAILABLE = True
except ImportError:
    _1D_AVAILABLE = False

from .models import (
    # Main architectures
    EMUNet,
    EMLeNet,
    EMUNetPP,
    EMResNet,
    # New: Map-to-map models for image translation
    EMUNetMapToMap,
    EMUNetPPMapToMap,
    MapToMapHead,
    # Building blocks
    CBAM,
    BaseConvBlock,
    NetworkHead,
    SimpleAttention,
    # New: Skip connection utilities
    SkipConnectionAligner,
    # Radiomics and feature computation
    calculate_fos_features,
    calculate_simple_glcm_features,
    # Training utilities
    EarlyStopping,
    ModelCheckpoint,
    TrainingHistory,
    # Model I/O (imported from pyable-ml)
    save_model,
    load_model,
    save_checkpoint,
    load_checkpoint,
)

from .utils import (
    # Feature selection
    feature_selection,
    filterFeaturesByScore,
    filterFeaturesByCorrelation,
    filterFeaturesByMAD,
    rankFeaturesByRepeatedGini,
    zScoreFeatures,
    # Evaluation metrics
    compute_metrics,
    compute_binary_metrics,
    compute_multilabel_sensitivity_specificity,
    testPrediction,
    # Data splitting (medical imaging aware)
    extract_patient_groups,
    stratified_group_split,
    # Visualization
    visualize_embeddings,
    write_confusion_matrix_to_tensorboard,
    # Explainability
    GradCAM,
    compute_saliency_map,
    postprocess_cam,
    # Medical imaging utilities
    resize_image,
    store_3d_array,
    # General utilities
    remove_nans,
    train,
)

__version__ = "3.5.0.0"

__all__ = [
    # Modern Loaders
    "TrenoDataset",
    "create_manifest_from_csv",
    "save_manifest",
    "load_manifest",
    "create_treno_dataset_from_csv",
    # Main Architectures (2D/3D)
    "EMUNet",
    "EMUNetPP",
    "EMLeNet",
    "EMResNet",
    # New: Map-to-Map Models (ND image translation)
    "EMUNetMapToMap",
    "EMUNetPPMapToMap",
    "MapToMapHead",
    # 1D Models (time-series)
    "UNet1DOptimized",
    "EMUNet1D",
    "DilatedConv1dBlock",
    "TemporalPooling",
    # Building Blocks
    "CBAM",
    "BaseConvBlock",
    "NetworkHead",
    "SimpleAttention",
    "SkipConnectionAligner",
    # Features
    "calculate_fos_features",
    "calculate_simple_glcm_features",
    # Training Utilities
    "EarlyStopping",
    "ModelCheckpoint",
    "TrainingHistory",
    "save_model",
    "load_model",
    "save_checkpoint",
    "load_checkpoint",
    # Utils
    "feature_selection",
    "compute_metrics",
    "GradCAM",
    "train",
    # Flags
    "PYABLE_DATALOADER_AVAILABLE",
    "_1D_AVAILABLE",
]

