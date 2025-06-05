"""
Colab 优化配置
"""
import os

class ColabConfig:
    # GPU设置
    GPU_IDS = '0'
    
    # 数据设置
    BATCH_SIZE = 8  # T4 GPU适用
    NUM_WORKERS = 2  # Colab优化
    
    # 训练设置
    MAX_EPOCHS = 100  # 减少训练时间
    LEARNING_RATE = 0.00031
    
    # 项目设置
    PROJECT_NAME = 'elgcnet_levir_colab'
    CHECKPOINT_ROOT = './checkpoints'
    VIS_ROOT = './vis'
    
    # 模型设置
    NET_G = 'ELGCNet'
    DEC_EMBED_DIM = 256
    N_CLASS = 2
    
    # 数据集设置
    DATA_NAME = 'LEVIR'
    IMG_SIZE = 256
