"""
改进的数据集准备脚本 - 平衡变化样本分布
"""
import os
import time
import shutil
import argparse
import random
import numpy as np
from pathlib import Path
from PIL import Image

def prepare_balanced_colab_dataset(train_samples=200, val_samples=50, test_samples=50):
    """准备平衡的Colab训练数据集"""
    drive_dataset_path = Path("/content/drive/My Drive/Change_Detection/LEVIR-CD-256")
    local_dataset_path = Path("./datasets/CD/LEVIR-CD-256")
    
    if not drive_dataset_path.exists():
        print(f"错误：数据集路径不存在: {drive_dataset_path}")
        return False
    
    print("开始智能选择数据集样本...")
    start_time = time.time()
    
    # 创建本地目录
    for folder in ['A', 'B', 'label', 'list']:
        (local_dataset_path / folder).mkdir(parents=True, exist_ok=True)
    
    def analyze_and_select_samples(split_name, target_count):
        """分析并智能选择样本"""
        src_list = drive_dataset_path / 'list' / f'{split_name}.txt'
        
        print(f"\n分析 {split_name} 集...")
        
        # 读取所有样本
        with open(src_list, 'r') as f:
            all_files = f.read().strip().split('\n')
        
        # 分析每个样本的变化比例
        samples_with_ratio = []
        
        for i, filename in enumerate(all_files):
            if i % 100 == 0:
                print(f"  分析进度: {i}/{len(all_files)}")
                
            label_path = drive_dataset_path / 'label' / filename
            if label_path.exists():
                try:
                    label = np.array(Image.open(label_path))
                    change_ratio = np.sum(label > 0) / label.size * 100
                    samples_with_ratio.append((filename, change_ratio))
                except:
                    continue
        
        print(f"  成功分析 {len(samples_with_ratio)} 个样本")
        
        # 按变化比例排序
        samples_with_ratio.sort(key=lambda x: x[1], reverse=True)
        
        # 智能选择策略：确保包含不同变化程度的样本
        high_change = [item for item in samples_with_ratio if item[1] >= 1.0]  # ≥1%
        medium_change = [item for item in samples_with_ratio if 0.3 <= item[1] < 1.0]  # 0.3-1%
        low_change = [item for item in samples_with_ratio if item[1] < 0.3]  # <0.3%
        
        print(f"  高变化样本(≥1%): {len(high_change)}")
        print(f"  中变化样本(0.3-1%): {len(medium_change)}")
        print(f"  低变化样本(<0.3%): {len(low_change)}")
        
        # 分配样本数量
        high_count = min(len(high_change), max(target_count // 2, 20))  # 至少一半或20个
        medium_count = min(len(medium_change), target_count // 4)
        low_count = target_count - high_count - medium_count
        
        selected = []
        
        # 随机选择各类样本
        if high_change:
            selected.extend(random.sample(high_change, min(high_count, len(high_change))))
        
        if medium_change and medium_count > 0:
            selected.extend(random.sample(medium_change, min(medium_count, len(medium_change))))
        
        if low_change and low_count > 0:
            selected.extend(random.sample(low_change, min(low_count, len(low_change))))
        
        # 如果样本不够，从剩余样本中随机选择
        if len(selected) < target_count:
            remaining = [item for item in samples_with_ratio if item not in selected]
            if remaining:
                additional = target_count - len(selected)
                selected.extend(random.sample(remaining, min(additional, len(remaining))))
        
        # 最终选择的文件名
        selected_files = [filename for filename, _ in selected[:target_count]]
        
        # 统计选择结果
        selected_ratios = [ratio for _, ratio in selected[:target_count]]
        print(f"  最终选择: {len(selected_files)} 个样本")
        print(f"  变化比例: 平均 {np.mean(selected_ratios):.3f}%, 范围 {np.min(selected_ratios):.3f}%-{np.max(selected_ratios):.3f}%")
        
        return selected_files
    
    # 配置样本数量
    splits_config = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    all_files = set()
    
    # 为每个分割智能选择样本
    for split_name, num_samples in splits_config.items():
        selected_files = analyze_and_select_samples(split_name, num_samples)
        
        # 保存列表文件
        dst_list = local_dataset_path / 'list' / f'{split_name}.txt'
        with open(dst_list, 'w') as f:
            f.write('\n'.join(selected_files))
        
        all_files.update(selected_files)
    
    # 复制图片文件
    total_files = len(all_files)
    print(f"\n开始复制 {total_files} 组图片...")
    
    for idx, filename in enumerate(all_files):
        if idx % 50 == 0:
            print(f"复制进度: {idx}/{total_files}")
        
        for folder in ['A', 'B', 'label']:
            src = drive_dataset_path / folder / filename
            dst = local_dataset_path / folder / filename
            if src.exists():
                shutil.copy2(src, dst)
    
    end_time = time.time()
    print(f"\n数据集准备完成！用时: {end_time - start_time:.2f} 秒")
    
    # 最终统计
    for folder in ['A', 'B', 'label']:
        count = len(list((local_dataset_path / folder).glob('*.png')))
        print(f"{folder}: {count} 个文件")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='智能准备ELGC-Net数据集')
    parser.add_argument('--train_samples', type=int, default=200)
    parser.add_argument('--val_samples', type=int, default=50) 
    parser.add_argument('--test_samples', type=int, default=50)
    args = parser.parse_args()
    
    # 设置随机种子确保可复现
    random.seed(42)
    np.random.seed(42)
    
    prepare_balanced_colab_dataset(args.train_samples, args.val_samples, args.test_samples)