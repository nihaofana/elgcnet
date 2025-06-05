"""
保存训练结果到Google Drive
"""
import os
import shutil
from pathlib import Path
from datetime import datetime

def save_results_to_drive():
    """保存模型和日志到Google Drive"""
    # 设置路径
    checkpoint_dir = Path('./checkpoints/elgcnet_levir_colab')
    vis_dir = Path('./vis/elgcnet_levir_colab')
    
    # Drive保存路径（包含时间戳）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    drive_save_path = Path(f'/content/drive/My Drive/ELGC-Net-Results/{timestamp}')
    drive_save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    saved_files = []
    
    # 最佳模型
    best_model = checkpoint_dir / 'checkpoint_best.pt'
    if best_model.exists():
        dst = drive_save_path / 'checkpoint_best.pt'
        shutil.copy2(best_model, dst)
        saved_files.append(('最佳模型', dst))
    
    # 最新模型
    latest_model = checkpoint_dir / 'checkpoint_latest.pt'
    if latest_model.exists():
        dst = drive_save_path / 'checkpoint_latest.pt'
        shutil.copy2(latest_model, dst)
        saved_files.append(('最新模型', dst))
    
    # 训练日志
    log_file = vis_dir / 'log.txt'
    if log_file.exists():
        dst = drive_save_path / 'training_log.txt'
        shutil.copy2(log_file, dst)
        saved_files.append(('训练日志', dst))
    
    # 创建README
    readme_content = f"""# ELGC-Net 训练结果
    
训练时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 文件列表
"""
    for name, path in saved_files:
        readme_content += f"- {name}: {path.name}\n"
    
    with open(drive_save_path / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print(f"结果已保存到: {drive_save_path}")
    for name, path in saved_files:
        print(f"  - {name}: {path}")
    
    return str(drive_save_path)

if __name__ == '__main__':
    save_results_to_drive()
