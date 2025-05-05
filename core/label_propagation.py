import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import logging

logger = logging.getLogger(__name__)

class LabelPropagation(nn.Module):
    def __init__(self, num_layers=5, alpha=0.9):
        super().__init__()
        self.num_layers = num_layers
        self.alpha = alpha
        
    def forward(self, g, labels, mask=None):
        with g.local_scope():
            # 初始化节点特征
            y = labels.float()
            if mask is not None:
                y = y * mask.float().unsqueeze(-1)
            
            # 计算度矩阵的逆
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).unsqueeze(-1)
            
            # 标签传播
            for _ in range(self.num_layers):
                # 计算邻居平均值
                g.ndata['h'] = y * norm
                g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                y = g.ndata.pop('h') * norm
                
                # 应用平滑因子
                y = self.alpha * y + (1 - self.alpha) * labels.float()
                
                if mask is not None:
                    y = y * mask.float().unsqueeze(-1)
            
            return y

class AdaptiveLabelPropagation(nn.Module):
    """自适应标签传播模块，支持异构图"""
    
    def __init__(self, in_dim, num_layers=5, alpha=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.num_layers = num_layers
        self.alpha = alpha
        
        # 为不同类型的边添加权重
        self.edge_weights = nn.ParameterDict({
            'connect': nn.Parameter(torch.ones(1)),
            'decorate': nn.Parameter(torch.ones(1)),
            'next': nn.Parameter(torch.ones(1))
        })
        
        # 特征转换层
        self.feature_transform = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, g, init_logits, features, edge_dict=None):
        """前向传播
        
        Args:
            g: DGL异构图
            init_logits: 初始预测概率
            features: 节点特征
            edge_dict: 边类型到边的映射字典
            
        Returns:
            torch.Tensor: 优化后的预测概率
        """
        try:
            # 检查输入
            if not isinstance(g, dgl.DGLGraph):
                raise ValueError("输入必须是DGL图")
                
            if edge_dict is None:
                logger.warning("未提供边字典，将尝试直接从图中获取边")
                edge_dict = {etype: g.edges(etype=etype) for etype in g.etypes}
            
            # 转换特征
            transformed_features = self.feature_transform(features)
            
            # 初始化传播结果
            current_logits = init_logits
            
            # 对每一层进行传播
            for _ in range(self.num_layers):
                next_logits = torch.zeros_like(current_logits)
                total_weight = torch.zeros(current_logits.size(0), current_logits.size(1), device=current_logits.device)
                
                # 对每种边类型进行传播
                for etype, (src, dst) in edge_dict.items():
                    if etype in self.edge_weights:
                        # 获取边权重
                        edge_weight = F.sigmoid(self.edge_weights[etype])
                        
                        # 计算特征相似度
                        src_feat = transformed_features[src]
                        dst_feat = transformed_features[dst]
                        similarity = F.cosine_similarity(src_feat, dst_feat, dim=1).unsqueeze(1)
                        
                        # 扩展相似度维度以匹配logits
                        similarity = similarity.expand(-1, current_logits.size(1))
                        
                        # 应用边权重和相似度
                        weight = edge_weight * similarity
                        
                        # 传播标签
                        next_logits.index_add_(0, src, weight * current_logits[dst])
                        total_weight.index_add_(0, src, weight)
                
                # 归一化
                mask = total_weight > 0
                next_logits[mask] = next_logits[mask] / total_weight[mask]
                
                # 更新当前预测
                current_logits = self.alpha * next_logits + (1 - self.alpha) * init_logits
            
            return current_logits
            
        except Exception as e:
            logger.error(f"标签传播失败: {str(e)}")
            return init_logits  # 出错时返回初始预测 