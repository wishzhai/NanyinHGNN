import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import logging
import math

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MusicFeatureEnhancer:
    def __init__(self, config):
        """初始化特征增强器
        
        Args:
            config: 配置字典
        """
        self.config = config
        model_config = config.get('model', {})
        self.hidden_dim = model_config.get('hidden_dim', 128)
    
    def __call__(self, graph):
        """应用特征增强
        
        Args:
            graph: DGL图
            
        Returns:
            dgl.DGLGraph: 处理后的图
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return graph
            
        # 获取基础特征
        features = self.get_node_features(graph)
        
        if features is None:
            return graph
            
        # 更新图中的特征
        graph.nodes['note'].data['feat'] = features
        
        return graph
    
    def get_node_features(self, graph):
        """获取节点特征
        
        Args:
            graph: DGL图
            
        Returns:
            torch.Tensor: 节点特征
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return None
            
        # 如果已经有特征，直接返回
        if 'feat' in graph.nodes['note'].data:
            return graph.nodes['note'].data['feat']
            
        # 获取音符属性
        pitch = graph.nodes['note'].data.get('pitch', None)
        duration = graph.nodes['note'].data.get('duration', None)
        velocity = graph.nodes['note'].data.get('velocity', None)
        position = graph.nodes['note'].data.get('position', None)
        
        if any(x is None for x in [pitch, duration, velocity, position]):
            return None
            
        # 标准化特征
        pitch = (pitch - 60) / 24  # 以中央C为中心归一化
        duration = duration / 2  # 假设最大时值为2
        velocity = velocity / 127  # MIDI力度范围0-127
        position = position / 16  # 假设一个小节长度为16
        
        # 组合特征
        features = torch.cat([
            pitch.float().unsqueeze(-1),
            duration.float().unsqueeze(-1),
            velocity.float().unsqueeze(-1),
            position.float().unsqueeze(-1)
        ], dim=-1)
        
        # 扩展到128维
        expanded_features = torch.zeros(features.shape[0], self.hidden_dim, device=features.device)
        expanded_features[:, :4] = features
        
        # 添加位置编码
        positions = torch.arange(features.shape[0], device=features.device)
        pos_encoding = self.get_positional_encoding(positions, self.hidden_dim - 4)
        expanded_features[:, 4:] = pos_encoding
        
        return expanded_features
        
    def get_positional_encoding(self, positions, dim):
        """生成位置编码
        
        Args:
            positions: 位置索引
            dim: 编码维度
            
        Returns:
            torch.Tensor: 位置编码
        """
        device = positions.device
        max_len = positions.shape[0]
        
        pe = torch.zeros(max_len, dim, device=device)
        position = positions.float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe.size(1)//2])  # 确保不超出维度
        
        return pe


class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model, max_len=2000):
        """初始化
        
        Args:
            d_model: 模型维度
            max_len: 最大序列长度
        """
        super().__init__()
        
        # 创建位置编码表
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, positions):
        """前向传播
        
        Args:
            positions: 位置索引，形状为 [batch_size]
            
        Returns:
            torch.Tensor: 位置编码，形状为 [batch_size, d_model]
        """
        return self.pe[positions]


class PitchIntervalEncoding(nn.Module):
    """音高和音程编码"""
    
    def __init__(self, d_model, num_pitches=128):
        """初始化
        
        Args:
            d_model: 模型维度
            num_pitches: 音高数量
        """
        super().__init__()
        
        # 创建音高嵌入
        self.pitch_embedding = nn.Embedding(num_pitches, d_model)
        
        # 初始化参数
        nn.init.normal_(self.pitch_embedding.weight, mean=0, std=0.02)
    
    def forward(self, pitches):
        """前向传播
        
        Args:
            pitches: 音高索引，形状为 [batch_size]
            
        Returns:
            torch.Tensor: 音高编码，形状为 [batch_size, d_model]
        """
        # 确保在Embedding范围内
        pitches = torch.clamp(pitches, min=0, max=127)
        return self.pitch_embedding(pitches)


class FeatureEnhancer(nn.Module):
    """特征增强器：通过多尺度特征提取和上下文感知来增强输入特征"""
    
    def __init__(self, config=None):
        super().__init__()
        self.hidden_dim = 256  # 改为256维度
        
        # 特征提取器
        self.feature_extractors = nn.ModuleList([
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        ])
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 上下文编码器
        self.context_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=2,
            dropout=0.1,
            bidirectional=True,
            batch_first=True
        )
        
        # 输出投影
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, x):
        try:
            # 特征提取
            x1 = self.feature_extractors[0](x)
            x2 = self.feature_extractors[1](x)
            fused_features = (x1 + x2) / 2
            fused_features = self.norm1(fused_features)
            
            # 注意力处理
            attn_output, _ = self.attention(fused_features, fused_features, fused_features)
            
            # 上下文编码
            context_output, _ = self.context_encoder(attn_output)
            
            # 输出投影
            output = self.output_projection(context_output)
            output = self.norm2(output)
            
            return output
            
        except Exception as e:
            logger.error(f"特征增强失败: {str(e)}")
            logger.error(f"输入特征形状: {x.shape}")
            return x 