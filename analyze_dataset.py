"""
数据集分析脚本 - 诊断变化像素分布问题
"""
import numpy as np
from PIL import Image
import os

def analyze_current_dataset():
    """分析当前数据集的变化像素分布"""
    dataset_path = "./datasets/CD/LEVIR-CD-256"
    
    print("🔍 分析当前数据集的变化像素分布...")
    
    for split in ['train', 'val', 'test']:
        list_file = f"{dataset_path}/list/{split}.txt"
        
        if not os.path.exists(list_file):
            print(f"❌ {split}.txt 不存在")
            continue
            
        with open(list_file, 'r') as f:
            files = [line.strip() for line in f.readlines()]
        
        print(f"\n📊 {split.upper()} 集分析:")
        print(f"样本数量: {len(files)}")
        
        change_ratios = []
        valid_samples = 0
        
        for i, filename in enumerate(files):
            label_path = f"{dataset_path}/label/{filename}"
            
            if os.path.exists(label_path):
                try:
                    label = np.array(Image.open(label_path))
                    
                    # 计算变化像素比例
                    total_pixels = label.size
                    change_pixels = np.sum(label > 0)
                    change_ratio = change_pixels / total_pixels * 100
                    
                    change_ratios.append(change_ratio)
                    valid_samples += 1
                    
                    if i < 5:  # 显示前5个样本的详细信息
                        print(f"  样本 {i+1} ({filename}): {change_ratio:.3f}% 变化")
                        
                except Exception as e:
                    print(f"  ❌ 无法读取 {filename}: {e}")
            else:
                print(f"  ❌ 标签文件不存在: {filename}")
        
        if change_ratios:
            print(f"  📈 统计结果:")
            print(f"    - 有效样本: {valid_samples}/{len(files)}")
            print(f"    - 平均变化比例: {np.mean(change_ratios):.3f}%")
            print(f"    - 最小变化比例: {np.min(change_ratios):.3f}%")
            print(f"    - 最大变化比例: {np.max(change_ratios):.3f}%")
            print(f"    - 标准差: {np.std(change_ratios):.3f}%")
            
            # 检查是否有足够的变化样本
            meaningful_change = [r for r in change_ratios if r > 1.0]  # >1%变化
            print(f"    - 有意义变化样本(>1%): {len(meaningful_change)}/{len(change_ratios)}")
            
            if np.mean(change_ratios) < 0.5:
                print("    ⚠️ 警告：平均变化比例过低，建议重新选择数据")
            elif np.mean(change_ratios) < 1.0:
                print("    🔄 提示：变化比例偏低，建议使用加权损失")
            else:
                print("    ✅ 变化比例正常")
        else:
            print(f"  ❌ 无法分析任何样本")

if __name__ == '__main__':
    analyze_current_dataset()