"""
训练监控工具
"""
import subprocess
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

def monitor_gpu(duration=60, interval=2):
    """监控GPU使用情况
    
    Args:
        duration: 监控持续时间（秒）
        interval: 更新间隔（秒）
    """
    gpu_usage = []
    memory_usage = []
    timestamps = []
    
    start_time = time.time()
    
    print(f"开始监控GPU (持续{duration}秒)...")
    
    while time.time() - start_time < duration:
        # 获取GPU信息
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', 
             '--format=csv,noheader,nounits'], 
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            output = result.stdout.strip().split(', ')
            gpu_usage.append(int(output[0]))
            memory_usage.append(int(output[1]))
            timestamps.append(time.time() - start_time)
            
            # 实时绘图
            clear_output(wait=True)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            
            ax1.plot(timestamps, gpu_usage, 'b-', linewidth=2)
            ax1.set_ylabel('GPU 使用率 (%)')
            ax1.set_ylim(0, 100)
            ax1.grid(True, alpha=0.3)
            ax1.set_title('GPU 监控')
            
            ax2.plot(timestamps, memory_usage, 'r-', linewidth=2)
            ax2.set_ylabel('显存使用 (MB)')
            ax2.set_xlabel('时间 (秒)')
            ax2.grid(True, alpha=0.3)
            
            # 显示当前值
            ax1.text(0.02, 0.98, f'当前: {gpu_usage[-1]}%', 
                    transform=ax1.transAxes, va='top')
            ax2.text(0.02, 0.98, f'当前: {memory_usage[-1]}MB', 
                    transform=ax2.transAxes, va='top')
            
            plt.tight_layout()
            plt.show()
        
        time.sleep(interval)
    
    print("监控完成！")
    return timestamps, gpu_usage, memory_usage

if __name__ == '__main__':
    # 默认监控60秒
    monitor_gpu(60)