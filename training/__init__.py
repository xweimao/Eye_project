"""
Training module for Eye State Classification Project
"""

from .trainer import (
    EyeDataset,
    PyTorchTrainer,
    TensorFlowTrainer,
    load_processed_data,
    plot_training_history
)

__all__ = [
    'EyeDataset',
    'PyTorchTrainer',
    'TensorFlowTrainer',
    'load_processed_data',
    'plot_training_history'
]
