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

from .models import (
    # Main architectures
    EMUNet,
    EMLeNet,
    # Building blocks
    CBAM,
    BaseConvBlock,
    NetworkHead,
    SimpleAttention,
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

__version__ = "3.0.5.0"

__all__ = [
    # Modern API
    "TrenoDataset",
    "create_manifest_from_csv",
    "save_manifest",
    "load_manifest",
    "create_treno_dataset_from_csv",
    # Legacy API
    "ImageImageDataset",
    "ImageLabelmapDataset",
    "ImaImaDataset",
    "ImaRoiDataset",
    "normalize",
    "labelMapToChannel",
    "possibletransforms",
    "ImaginableDataloader",
    # Flags
    "PYABLE_DATALOADER_AVAILABLE",
    "PYABLE_AVAILABLE",
]
