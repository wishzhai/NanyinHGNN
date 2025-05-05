import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class LoRAAdapter(nn.Module):
    """使用LoRA技术的特征适配器，替代原有的GNNAdapter"""
    
    def __init__(self, in_dim, hidden_dim, r=4, alpha=8, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 输入投影层（使用LoRA）
        self.input_proj = LoRALinear(in_dim, hidden_dim, r=r, alpha=alpha, dropout=dropout)
        
        # 特征转换层（使用LoRA）
        self.transform_layers = nn.ModuleList([
            LoRALinear(hidden_dim, hidden_dim, r=r, alpha=alpha, dropout=dropout)
            for _ in range(2)  # 使用2层转换
        ])
        
        # 层归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(2)
        ])
        
        # 位置编码
        self.max_seq_len = 2000
        self.pos_encoder = PositionalEncoding(hidden_dim, self.max_seq_len)
        
        # 残差连接的dropout
        self.dropout = nn.Dropout(dropout)
        
    def add_positional_encoding(self, features, positions=None):
        """添加位置编码
        
        Args:
            features: 输入特征 [num_nodes, hidden_dim]
            positions: 位置信息 [num_nodes]
            
        Returns:
            torch.Tensor: 添加位置编码后的特征
        """
        if positions is None:
            positions = torch.arange(features.size(0), device=features.device)
        
        # 确保位置在合理范围内
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        
        # 获取位置编码并添加到特征中
        pos_encoding = self.pos_encoder(positions)
        return features + pos_encoding
        
    def forward(self, g, features):
        """前向传播
        
        Args:
            g: DGL图对象（为了保持接口一致，但不使用）
            features: 节点特征
            
        Returns:
            torch.Tensor: 处理后的特征
        """
        try:
            # 确保特征是浮点类型
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            else:
                features = features.float()
            
            # 通过LoRA投影输入特征
            h = self.input_proj(features)
            
            # 获取节点位置信息（如果存在）
            positions = None
            if hasattr(g, 'nodes') and 'note' in g.ntypes:
                positions = g.nodes['note'].data.get('position', None)
                if positions is not None and positions.dim() > 1:
                    positions = positions.squeeze()
            
            # 添加位置编码
            try:
                h = self.add_positional_encoding(h, positions)
            except Exception as e:
                logger.warning(f"位置编码应用失败: {str(e)}")
            
            # 特征转换
            for transform, norm in zip(self.transform_layers, self.norms):
                h_new = transform(h)
                h_new = norm(h_new)
                h_new = F.gelu(h_new)  # 使用GELU激活函数
                h_new = self.dropout(h_new)
                h = h + h_new  # 残差连接
            
            return h
            
        except Exception as e:
            logger.error(f"LoRAAdapter前向传播出错: {str(e)}")
            return features

class LoRALinear(nn.Module):
    """LoRA线性层实现"""
    
    def __init__(self, in_features, out_features, r=4, alpha=8, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 原始线性层（冻结）
        self.linear = nn.Linear(in_features, out_features)
        for param in self.linear.parameters():
            param.requires_grad = False
            
        # LoRA低秩分解
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(dropout)
        
        # 初始化LoRA权重
        self.reset_lora_parameters()
        
    def reset_lora_parameters(self):
        """初始化LoRA参数"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        """前向传播"""
        # 原始线性变换 + LoRA路径
        return self.linear(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling

class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe)
    
    def forward(self, positions):
        """
        Args:
            positions: [batch_size] 或 [seq_len] 的位置索引
            
        Returns:
            [batch_size/seq_len, d_model] 位置编码
        """
        return self.pe[positions] 