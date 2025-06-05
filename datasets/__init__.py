# datasets/__init__.py
"""
Change Detection Datasets Package
This package contains dataset classes and utilities for remote sensing change detection.
"""

from .CD_dataset import CDDataset, ImageDataset
from .data_utils import CDDataAugmentation

__all__ = ['CDDataset', 'ImageDataset', 'CDDataAugmentation']
__version__ = '1.0.0'


# models/__init__.py
"""
Change Detection Models Package
This package contains model architectures and components for ELGC-Net.
"""

from .elgcnet import ELGCNet

__all__ = ['ELGCNet']
__version__ = '1.0.0'


# models/networks/__init__.py
"""
Network Components Package
This package contains network building blocks and utilities.
"""

# 如果有具体的网络组件，可以在这里导入
# from .resnet import ResNet
# from .attention import ELGCAModule

__all__ = []
__version__ = '1.0.0'


# 根目录的 __init__.py (可选)
"""
ELGC-Net: Efficient Local-Global Context Aggregation for Remote Sensing Change Detection
A PyTorch implementation for remote sensing change detection.
"""

__version__ = '1.0.0'
__author__ = 'Mubashir Noman'
__email__ = 'mubashir.noman@mbzuai.ac.ae'

# 如果需要的话，可以导入主要的类和函数
# from .models import ELGCNet
# from .utils import get_loader