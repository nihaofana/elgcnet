# ELGC-Net Colab 使用指南

本目录包含了在Google Colab上运行ELGC-Net的优化脚本。

## 文件说明

- `colab_config.py` - Colab环境优化的配置文件
- `train_colab.py` - Colab训练脚本
- `eval_colab.py` - Colab评估脚本
- `prepare_data.py` - 数据集准备脚本
- `save_results.py` - 结果保存脚本
- `monitor_training.py` - GPU监控工具
- `ELGC-Net_Quick.ipynb` - 快速开始的Jupyter Notebook

## 快速开始

1. 在Google Colab中打开 `ELGC-Net_Quick.ipynb`
2. 运行所有单元格
3. 等待训练完成

## 配置说明

默认配置已针对Colab的T4 GPU优化：
- Batch Size: 8
- Workers: 2
- Epochs: 100

如需修改，请编辑 `colab_config.py` 中的参数。

## 数据集准备

确保数据集已上传到Google Drive：
```
/content/drive/My Drive/Change_Detection/LEVIR-CD-256/
├── A/
├── B/
├── label/
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## 使用方法

### 方法1：使用Notebook（推荐）
直接在Colab中打开并运行 `ELGC-Net_Quick.ipynb`

### 方法2：命令行方式
```bash
# 克隆项目
!git clone https://github.com/techmn/elgcnet.git
%cd elgcnet

# 安装依赖
!pip install -r requirements.txt

# 准备数据
!python colab/prepare_data.py

# 运行训练
!python colab/train_colab.py

# 评估模型
!python colab/eval_colab.py

# 保存结果
!python colab/save_results.py
```

## 注意事项

1. 确保已挂载Google Drive
2. 检查GPU是否可用
3. 如遇到内存不足，请减小batch_size
4. 训练中断后可以从最新检查点恢复（需要修改代码支持resume）
EOF

# 创建colab/colab_config.py
echo "创建 colab/colab_config.py..."
cat > colab/colab_config.py << 'EOF'
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
EOF