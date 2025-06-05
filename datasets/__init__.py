# datasets/__init__.py
"""
Change Detection Datasets Package

This package contains dataset classes and utilities for remote sensing change detection.
"""

from .CD_dataset import CDDataset, ImageDataset
from .data_utils import CDDataAugmentation

__all__ = ['CDDataset', 'ImageDataset', 'CDDataAugmentation']
__version__ = '1.0.0'