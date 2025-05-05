import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
import logging
import traceback
import math

logger = logging.getLogger(__name__)

class LoRALinear(nn.Module):
    """
    实现LoRA (Low-Rank Adaptation) 线性层
    """
    def __init__(self, in_features, out_features, r=4, alpha=8, dropout=0.1):
        """
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            r: LoRA的秩 (rank)
            alpha: 缩放因子
            dropout: Dropout率
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 原始线性层 (冻结)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        for param in self.linear.parameters():
            param.requires_grad = False
            
        # LoRA低秩分解
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        
        # 初始化LoRA权重
        self.reset_lora_parameters()
        
    def reset_lora_parameters(self):
        """初始化LoRA参数"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        """前向传播"""
        # 原始线性变换 + LoRA路径
        return self.linear(x) + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRAGraphConv(nn.Module):
    """
    应用LoRA技术的图卷积层
    """
    def __init__(self, in_feats, out_feats, r=4, alpha=8, dropout=0.1, allow_zero_in_degree=False):
        super().__init__()
        self.gnn = GraphConv(in_feats, out_feats, allow_zero_in_degree=allow_zero_in_degree)
        
        # 冻结GNN参数
        for param in self.gnn.parameters():
            param.requires_grad = False
            
        # 添加LoRA路径
        self.lora_linear = LoRALinear(in_feats, out_feats, r=r, alpha=alpha, dropout=dropout)
        
    def forward(self, graph, feat):
        """
        结合GNN和LoRA的前向传播
        """
        # GNN路径
        h_gnn = self.gnn(graph, feat)
        
        # LoRA路径（直接特征变换，不经过图结构）
        h_lora = self.lora_linear(feat)
        
        # 合并两个路径的结果
        return h_gnn + h_lora


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


class LoRAGNNAdapter(nn.Module):
    """结合LoRA和GNN的适配器，用于小数据集上的高效特征增强"""
    
    def __init__(self, in_dim, hidden_dim, num_layers=2, lora_r=4, lora_alpha=8, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 使用LoRA处理输入投影
        self.input_proj = LoRALinear(in_dim, hidden_dim, r=lora_r, alpha=lora_alpha, dropout=dropout)
        
        # 使用LoRA增强的GNN层
        self.gnn_layers = nn.ModuleList([
            LoRAGraphConv(
                hidden_dim, 
                hidden_dim, 
                r=lora_r, 
                alpha=lora_alpha, 
                dropout=dropout,
                allow_zero_in_degree=True
            )
            for _ in range(num_layers)
        ])
        
        # 层归一化
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 位置编码
        self.max_seq_len = 2000
        self.pos_encoder = PositionalEncoding(hidden_dim, self.max_seq_len)
        
        # 残差丢弃
        self.dropout = nn.Dropout(dropout)
        
        # 输出投影 - 使用LoRA
        self.output_proj = LoRALinear(hidden_dim, hidden_dim, r=lora_r, alpha=lora_alpha, dropout=dropout)
        
        logger.info(f"初始化LoRAGNN适配器: 输入维度={in_dim}, 隐藏维度={hidden_dim}, 层数={num_layers}, LoRA秩={lora_r}")
        
    def add_positional_encoding(self, features, positions=None):
        """添加位置编码
        
        Args:
            features: 输入特征 [num_nodes, hidden_dim]
            positions: 位置信息 [num_nodes]
            
        Returns:
            torch.Tensor: 添加位置编码后的特征
        """
        if positions is None:
            # 如果没有提供位置信息，使用节点索引作为位置
            positions = torch.arange(features.size(0), device=features.device)
        
        # 确保位置在合理范围内
        positions = torch.clamp(positions, 0, self.max_seq_len - 1)
        
        # 获取位置编码并添加到特征中
        pos_encoding = self.pos_encoder(positions)
        return features + pos_encoding
        
    def forward(self, g, features):
        """前向传播
        
        Args:
            g: DGL图对象
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
            
            # 通过LoRA处理输入特征
            h = self.input_proj(features)
            
            # 获取节点位置信息（如果存在）
            positions = None
            if hasattr(g, 'ntypes') and 'note' in g.ntypes:
                positions = g.nodes['note'].data.get('position', None)
                if positions is not None and positions.dim() > 1:
                    positions = positions.squeeze()
            
            # 添加位置编码
            try:
                h = self.add_positional_encoding(h, positions)
            except Exception as e:
                logger.warning(f"位置编码应用失败: {str(e)}")
            
            # 准备图
            sg = self._prepare_graph(g)
            
            # GNN+LoRA层处理
            for i, (conv, norm) in enumerate(zip(self.gnn_layers, self.norms)):
                h_new = conv(sg, h)
                h_new = norm(h_new)
                h_new = F.gelu(h_new)  # 使用GELU作为激活函数
                h_new = self.dropout(h_new)
                h = h + h_new  # 残差连接
            
            # 输出投影
            h = self.output_proj(h)
            
            return h
            
        except Exception as e:
            logger.error(f"LoRAGNNAdapter前向传播出错: {str(e)}\n{traceback.format_exc()}")
            return features
    
    def _prepare_graph(self, g):
        """处理图结构，准备用于GNN计算
        
        Args:
            g: 输入图
            
        Returns:
            dgl.DGLGraph: 处理后的图
        """
        # 检查图的边类型和边数量
        if hasattr(g, 'etypes') and len(g.etypes) > 0:
            # 选择边数最多的边类型
            max_edges = 0
            selected_type = None
            for etype in g.etypes:
                num_edges = g.num_edges(etype)
                if num_edges > max_edges:
                    max_edges = num_edges
                    selected_type = etype
            
            if selected_type and max_edges > 0:
                logger.debug(f"使用边类型: {selected_type}")
                sg = g.edge_type_subgraph([selected_type])
            else:
                logger.debug("没有找到有效的边，使用原始图")
                sg = g
        else:
            sg = g
        
        # 添加自环边
        if not sg.is_homogeneous:
            # 如果是异构图，转换为同构图
            sg = dgl.to_homogeneous(sg)
        sg = dgl.remove_self_loop(sg)  # 先移除已有的自环边
        sg = dgl.add_self_loop(sg)     # 添加新的自环边
        
        return sg 