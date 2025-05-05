import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import traceback

logger = logging.getLogger(__name__)

class LearningRateWarmupCallback(Callback):
    """学习率预热回调
    
    在训练的前几个步骤中逐渐增加学习率，以提高训练稳定性。
    """
    
    def __init__(self, warmup_steps=100, verbose=True):
        """初始化
        
        Args:
            warmup_steps: 预热步数
            verbose: 是否打印日志
        """
        super().__init__()
        self.warmup_steps = warmup_steps
        self.verbose = verbose
        self.current_step = 0
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        """在每个训练批次开始时调用
        
        Args:
            trainer: PyTorch Lightning 训练器
            pl_module: PyTorch Lightning 模块
            batch: 当前批次数据
            batch_idx: 当前批次索引
        """
        # 获取优化器
        optimizers = trainer.optimizers
        if not optimizers:
            return
            
        # 获取当前学习率
        optimizer = optimizers[0]
        if not optimizer.param_groups:
            return
            
        # 计算预热因子
        if self.current_step < self.warmup_steps:
            warmup_factor = min(1.0, float(self.current_step) / float(self.warmup_steps))
            
            # 应用预热因子
            for param_group in optimizer.param_groups:
                if 'initial_lr' not in param_group:
                    param_group['initial_lr'] = param_group['lr']
                
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
                
            if self.verbose and self.current_step % 10 == 0:
                logger.info(f"学习率预热: 步骤 {self.current_step}/{self.warmup_steps}, 因子 {warmup_factor:.4f}, 学习率 {optimizer.param_groups[0]['lr']:.6f}")
                
        self.current_step += 1
        
    def on_train_epoch_start(self, trainer, pl_module):
        """在每个训练轮次开始时调用
        
        Args:
            trainer: PyTorch Lightning 训练器
            pl_module: PyTorch Lightning 模块
        """
        # 记录当前轮次
        if self.verbose:
            logger.info(f"开始训练轮次 {trainer.current_epoch + 1}")
            
    def state_dict(self):
        """获取状态字典，用于保存检查点
        
        Returns:
            dict: 状态字典
        """
        return {
            "current_step": self.current_step,
            "warmup_steps": self.warmup_steps
        }
        
    def load_state_dict(self, state_dict):
        """从状态字典加载，用于恢复检查点
        
        Args:
            state_dict: 状态字典
        """
        self.current_step = state_dict.get("current_step", 0)
        self.warmup_steps = state_dict.get("warmup_steps", self.warmup_steps) 