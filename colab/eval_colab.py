"""
ELGC-Net Colab评估脚本
"""
import os
import sys
from argparse import ArgumentParser

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from colab.colab_config import ColabConfig
from models.evaluator import CDEvaluator
import utils

def eval_colab():
    parser = ArgumentParser()
    args = parser.parse_args([])
    
    # 使用Colab配置
    config = ColabConfig()
    
    # 设置参数
    args.gpu_ids = config.GPU_IDS
    args.project_name = config.PROJECT_NAME
    args.checkpoints_root = config.CHECKPOINT_ROOT
    args.vis_root = config.VIS_ROOT
    args.num_workers = config.NUM_WORKERS
    args.dataset = 'CDDataset'
    args.data_name = config.DATA_NAME
    args.batch_size = 1  # 评估时使用batch_size=1
    args.split = 'test'
    args.img_size = config.IMG_SIZE
    args.n_class = config.N_CLASS
    args.dec_embed_dim = config.DEC_EMBED_DIM
    args.net_G = config.NET_G
    # args.checkpoint_name = 'checkpoint_best.pt'
    args.checkpoint_name = 'best_ckpt.pt'
    args.print_models = False
    
    utils.get_device(args)
    
    # 设置目录
    args.checkpoint_dir = os.path.join(args.checkpoints_root, args.project_name)
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    
    print("="*50)
    print("ELGC-Net 模型评估")
    print("="*50)
    print(f"数据集: {args.data_name}")
    print(f"检查点: {os.path.join(args.checkpoint_dir, args.checkpoint_name)}")
    print("="*50)
    
    # 检查模型文件是否存在
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    if not os.path.exists(checkpoint_path):
        print(f"错误：找不到模型文件 {checkpoint_path}")
        print("请先运行训练脚本: python colab/train_colab.py")
        return
    
    # 创建数据加载器
    dataloader = utils.get_loader(
        args.data_name, 
        img_size=args.img_size,
        batch_size=args.batch_size, 
        is_train=False,
        split=args.split
    )
    
    # 评估模型
    model = CDEvaluator(args=args, dataloader=dataloader)
    model.eval_models(checkpoint_name=args.checkpoint_name)

if __name__ == '__main__':
    eval_colab()