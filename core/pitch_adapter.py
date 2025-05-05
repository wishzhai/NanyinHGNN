#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音高预测的Adapter模块
用于在LoRA之上添加针对音高预测的局部任务适配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class AdapterGNN(nn.Module):
    """
    针对音高预测的轻量级Adapter模块
    
    Args:
        input_dim (int): 输入特征维度
        hidden_dim (int): Adapter隐藏层维度
        dropout (float): Dropout比率
        bottleneck_dim (int): 瓶颈层维度
    """
    def __init__(self, input_dim, hidden_dim=64, dropout=0.1, bottleneck_dim=16):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 初始化参数
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
        
    def forward(self, x):
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [batch_size, seq_len, input_dim]
            
        Returns:
            torch.Tensor: 适配后的特征 [batch_size, seq_len, input_dim]
        """
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        x = residual + x  # 残差连接
        return self.layer_norm(x)

class PitchPredictorWithAdapter(nn.Module):
    """
    带有Adapter的音高预测器
    
    Args:
        original_predictor: 原始音高预测器
        adapter_hidden_dim (int): Adapter隐藏层维度
        bottleneck_dim (int): Adapter瓶颈层维度
        dropout (float): Dropout比率
    """
    def __init__(self, original_predictor, adapter_hidden_dim=64, bottleneck_dim=16, dropout=0.1):
        super().__init__()
        self.original_predictor = original_predictor
        
        # 冻结原始预测器参数
        for param in self.original_predictor.parameters():
            param.requires_grad = False
            
        # 获取输入维度
        self.input_dim = self._get_input_dim()
        logger.info(f"检测到音高预测器输入维度: {self.input_dim}")
        
        # 添加Adapter
        self.pitch_adapter = AdapterGNN(
            input_dim=self.input_dim,
            hidden_dim=adapter_hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout
        )
        
        logger.info(f"已创建音高Adapter: hidden_dim={adapter_hidden_dim}, bottleneck_dim={bottleneck_dim}")
        
    def _get_input_dim(self):
        """获取原始预测器的输入维度"""
        # 遍历模型的所有模块，查找最后一个线性层的输入维度
        input_dim = None
        for name, module in reversed(list(self.original_predictor.named_modules())):
            if isinstance(module, nn.Linear):
                input_dim = module.in_features
                break
        
        if input_dim is None:
            logger.warning("无法自动检测输入维度，使用默认值256")
            input_dim = 256
            
        return input_dim
        
    def forward(self, x, *args, **kwargs):
        """
        前向传播
        
        Args:
            x: 输入特征
            *args, **kwargs: 传递给原始预测器的其他参数
            
        Returns:
            原始预测器的输出
        """
        # 应用Adapter
        adapted_features = self.pitch_adapter(x)
        
        # 使用原始预测器进行预测
        return self.original_predictor(adapted_features, *args, **kwargs)
        
    def train(self, mode=True):
        """
        设置训练模式
        
        Args:
            mode (bool): 是否为训练模式
        """
        # 保持原始预测器在评估模式
        self.original_predictor.eval()
        # 只设置Adapter的训练模式
        self.pitch_adapter.train(mode)
        return self 