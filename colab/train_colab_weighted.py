"""
ELGC-Net Colab训练脚本 - 使用加权损失处理数据不平衡
"""
import os
import sys
import torch
import torch.nn as nn
from argparse import ArgumentParser

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from colab.colab_config import ColabConfig
from models.trainer import CDTrainer
import utils

class WeightedCDTrainer(CDTrainer):
    """支持加权损失的变化检测训练器"""
    
    def __init__(self, args, dataloaders):
        super().__init__(args, dataloaders)
        
        # 设置类别权重 - 给变化类别更高权重
        self.class_weights = torch.tensor([1.0, 15.0])  # 无变化:变化 = 1:15
        if torch.cuda.is_available():
            self.class_weights = self.class_weights.cuda()
        
        print(f"🎯 使用类别权重: 无变化={self.class_weights[0]:.1f}, 变化={self.class_weights[1]:.1f}")
    
    def _get_loss(self, loss_name):
        """获取加权损失函数"""
        if loss_name == 'ce':
            # 使用加权交叉熵损失
            return nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            # 其他损失函数保持原样
            return super()._get_loss(loss_name)

def train_colab():
    # 创建参数
    parser = ArgumentParser()
    args = parser.parse_args([])
    
    # 使用Colab配置
    config = ColabConfig()
    
    # 基本设置
    args.gpu_ids = config.GPU_IDS
    args.project_name = config.PROJECT_NAME + '_weighted'  # 区分加权版本
    args.checkpoint_root = config.CHECKPOINT_ROOT
    args.vis_root = config.VIS_ROOT
    
    # 数据设置
    args.num_workers = config.NUM_WORKERS
    args.dataset = 'CDDataset'
    args.data_name = config.DATA_NAME
    args.batch_size = config.BATCH_SIZE
    args.split = "train"
    args.split_val = "val"
    args.img_size = config.IMG_SIZE
    
    # 模型设置
    args.n_class = config.N_CLASS
    args.dec_embed_dim = config.DEC_EMBED_DIM
    args.net_G = config.NET_G
    args.loss = 'ce'  # 使用加权交叉熵
    args.pretrain = None
    
    # 优化器设置
    args.optimizer = 'adamw'
    args.lr = config.LEARNING_RATE * 0.5  # 使用较低的学习率
    args.max_epochs = config.MAX_EPOCHS
    args.lr_policy = 'linear'
    args.lr_decay_iters = [100]
    
    # 设置设备
    utils.get_device(args)
    
    # 创建输出目录
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    print("="*50)
    print("ELGC-Net Colab 加权损失训练配置")
    print("="*50)
    print(f"GPU: {args.gpu_ids}")
    print(f"批次大小: {args.batch_size}")
    print(f"最大轮数: {args.max_epochs}")
    print(f"学习率: {args.lr}")
    print(f"数据集: {args.data_name}")
    print(f"检查点目录: {args.checkpoint_dir}")
    print(f"损失函数: 加权交叉熵 (变化类别权重x15)")
    print("="*50)
    
    # 获取数据加载器
    dataloaders = utils.get_loaders(args)
    
    # 创建加权训练器并开始训练
    try:
        model = WeightedCDTrainer(args=args, dataloaders=dataloaders)
        model.train_models()
        print(f"\n✅ 训练完成！模型保存在: {args.checkpoint_dir}")
        return args, True
    except KeyboardInterrupt:
        print("\n⚠️ 训练被用户中断")
        return args, False
    except Exception as e:
        print(f"\n❌ 训练出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return args, False

if __name__ == '__main__':
    args, success = train_colab()
    if success:
        print("\n🎯 可以运行以下命令评估模型:")
        print(f"python colab/eval_colab.py --project_name {args.project_name}")