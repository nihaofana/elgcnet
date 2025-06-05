"""
数据集准备脚本 - 从Google Drive复制部分数据到本地
"""
import os
import time
import shutil
import argparse
from pathlib import Path

def prepare_colab_dataset(train_samples=200, val_samples=50, test_samples=50):
    """准备Colab训练数据集"""
    # 设置路径
    drive_dataset_path = Path("/content/drive/My Drive/Change_Detection/LEVIR-CD-256")
    local_dataset_path = Path("/content/elgcnet/datasets/CD/LEVIR-CD-256")
    
    # 检查数据集是否存在
    if not drive_dataset_path.exists():
        print(f"错误：数据集路径不存在: {drive_dataset_path}")
        print("请确保数据集已上传到Google Drive的正确位置")
        return False
    
    print("开始准备数据集...")
    start_time = time.time()
    
    # 创建本地目录
    for folder in ['A', 'B', 'label', 'list']:
        (local_dataset_path / folder).mkdir(parents=True, exist_ok=True)
    
    # 配置样本数量
    splits_config = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    all_files = set()
    
    # 读取和创建列表文件
    for split_name, num_samples in splits_config.items():
        src_list = drive_dataset_path / 'list' / f'{split_name}.txt'
        dst_list = local_dataset_path / 'list' / f'{split_name}.txt'
        
        with open(src_list, 'r') as f:
            files = f.read().strip().split('\n')[:num_samples]
        
        with open(dst_list, 'w') as f:
            f.write('\n'.join(files))
        
        all_files.update(files)
        print(f"{split_name}: {len(files)} 样本")
    
    # 复制图片文件
    total_files = len(all_files)
    print(f"\n总共需要复制 {total_files} 组图片...")
    
    for idx, filename in enumerate(all_files):
        if idx % 50 == 0:
            print(f"进度: {idx}/{total_files}")
        
        for folder in ['A', 'B', 'label']:
            src = drive_dataset_path / folder / filename
            dst = local_dataset_path / folder / filename
            if src.exists():
                shutil.copy2(src, dst)
    
    # 完成统计
    end_time = time.time()
    print(f"\n数据集准备完成！用时: {end_time - start_time:.2f} 秒")
    
    for folder in ['A', 'B', 'label']:
        count = len(list((local_dataset_path / folder).glob('*.png')))
        print(f"{folder}: {count} 个文件")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='准备ELGC-Net Colab数据集')
    parser.add_argument('--train_samples', type=int, default=200, 
                        help='训练集样本数量 (默认: 200)')
    parser.add_argument('--val_samples', type=int, default=50, 
                        help='验证集样本数量 (默认: 50)')
    parser.add_argument('--test_samples', type=int, default=50, 
                        help='测试集样本数量 (默认: 50)')
    args = parser.parse_args()
    
    prepare_colab_dataset(args.train_samples, args.val_samples, args.test_samples)