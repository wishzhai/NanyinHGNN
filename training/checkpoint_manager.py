import os
import yaml
import torch
from datetime import datetime

class CheckpointManager:
    def __init__(self, save_dir="pretrained_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_checkpoint(self, model, epoch, val_loss, config):
        """保存完整检查点"""
        timestamp = datetime.now().strftime("%m%d-%H%M")
        filename = f"model-epoch{epoch}-{val_loss:.2f}-{timestamp}.pth"
        
        # 保存模型状态
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'val_loss': val_loss,
            'config': config
        }, os.path.join(self.save_dir, filename))
        
        # 备份配置
        self._backup_config(config, epoch)
        
        # 更新最佳模型
        if val_loss < self._get_best_loss():
            torch.save(model.state_dict(), 
                      os.path.join(self.save_dir, "best_model.pth"))
    
    def load_checkpoint(self, path, model):
        """加载检查点"""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state'])
        return checkpoint
    
    def _backup_config(self, config, epoch):
        """备份配置文件"""
        config_dir = os.path.join(self.save_dir, "config_backup")
        os.makedirs(config_dir, exist_ok=True)
        
        with open(f"{config_dir}/epoch_{epoch}_config.yaml", 'w') as f:
            yaml.dump(config, f)
    
    def _get_best_loss(self):
        """获取历史最佳损失"""
        try:
            return torch.load(
                os.path.join(self.save_dir, "best_loss.pt")
            )['val_loss']
        except:
            return float('inf')