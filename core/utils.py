import os
import glob
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def find_latest_checkpoint(checkpoints_dir):
    """查找最新的检查点文件
    
    Args:
        checkpoints_dir: 检查点目录路径
        
    Returns:
        str: 最新检查点的完整路径，如果没有找到则返回None
    """
    if not os.path.exists(checkpoints_dir):
        return None
        
    # 支持的检查点文件名模式
    patterns = [
        "enhanced_model_final.ckpt",
        "enhanced_model-epoch=*.ckpt",
        "model-*.ckpt",
        "*.ckpt"
    ]
    
    latest_checkpoint = None
    latest_time = 0
    
    for pattern in patterns:
        # 使用glob查找匹配的文件
        checkpoints = glob.glob(os.path.join(checkpoints_dir, pattern))
        
        for checkpoint in checkpoints:
            # 获取文件的修改时间
            mtime = os.path.getmtime(checkpoint)
            if mtime > latest_time:
                latest_time = mtime
                latest_checkpoint = checkpoint
                
    return latest_checkpoint
        
def ensure_dir(directory):
    """确保目录存在，如不存在则创建
    
    Args:
        directory: 目录路径
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    
def list_available_checkpoints(checkpoint_dir, pattern="model-*.ckpt"):
    """列出所有可用的检查点
    
    Args:
        checkpoint_dir: 检查点目录路径
        pattern: 检查点文件名匹配模式
        
    Returns:
        list: 检查点文件列表
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return []
        
    return sorted(
        [str(p) for p in checkpoint_dir.glob(pattern)],
        key=lambda x: os.path.getmtime(x),
        reverse=True
    ) 