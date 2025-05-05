import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math
import logging
from typing import Dict, List, Tuple, Union, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedGATv2Conv(nn.Module):
    """增强版的GATv2卷积层，支持异构图和多种边类型"""
    
    def __init__(self, in_feats, hidden_dim, num_heads, dropout=0.1, feat_drop=0.1, attn_drop=0.1, negative_slope=0.2):
        """初始化
        
        Args:
            in_feats: 输入特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout比率
            feat_drop: 特征dropout率
            attn_drop: 注意力dropout率
            negative_slope: LeakyReLU的负斜率
        """
        super().__init__()
        # 强制使用384维度以匹配预训练模型
        self.hidden_dim = 384
        self.out_feats = 384
        self.num_heads = num_heads
        self.dropout = dropout
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        
        # 特征变换
        self.fc = nn.Linear(in_feats, 384)
        
        # 注意力投影
        self.q_proj = nn.Linear(384, 384)
        self.k_proj = nn.Linear(384, 384)
        self.v_proj = nn.Linear(384, 384)
        
        # 输出投影
        self.out_proj = nn.Linear(384, 384)
        
        # Dropout层
        self.feat_dropout = nn.Dropout(feat_drop)
        self.attn_dropout = nn.Dropout(attn_drop)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(384)
        self.norm2 = nn.LayerNorm(384)
        
        # 初始化参数
        self._init_params()
    
    def _init_params(self):
        """重置模型参数"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.q_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.k_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.v_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.out_proj.weight, gain=gain)
        
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.q_proj.bias, 0)
        nn.init.constant_(self.k_proj.bias, 0)
        nn.init.constant_(self.v_proj.bias, 0)
        nn.init.constant_(self.out_proj.bias, 0)
    
    def _process_features(self, feat):
        """处理输入特征
        
        Args:
            feat: 输入特征
            
        Returns:
            torch.Tensor: 处理后的特征
        """
        # 应用特征投影
        feat = self.fc(feat)
        
        # 应用特征dropout
        feat = self.feat_dropout(feat)
        
        return feat
    
    def compute_attention(self, q, k):
        """计算注意力权重
        
        Args:
            q: 查询张量
            k: 键张量
            
        Returns:
            torch.Tensor: 注意力权重
        """
        # 确保q和k的形状正确
        # q和k应该是形状为[num_nodes, num_heads, head_dim]的张量
        if q.dim() != 3 or k.dim() != 3:
            logger.warning(f"注意力计算输入维度不正确: q.shape={q.shape}, k.shape={k.shape}")
            # 尝试修复维度
            head_dim = self.out_feats // self.num_heads
            if q.dim() == 2:
                q = q.view(-1, self.num_heads, head_dim)
            if k.dim() == 2:
                k = k.view(-1, self.num_heads, head_dim)
        
        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # 应用注意力缩放和偏置
        # 确保attn_bias的形状与attn兼容
        if attn.dim() == 3:  # [batch, heads, seq_len]
            attn = attn * self.attn_scale + self.attn_bias.squeeze(-1)
        else:
            attn = attn * self.attn_scale + self.attn_bias
        
        # 应用softmax
        attn = F.softmax(attn, dim=-1)
        
        # 应用注意力dropout
        attn = self.attn_dropout(attn)
        
        return attn
    
    def forward(self, graph, feat=None):
        """前向传播
        
        Args:
            graph: DGL图
            feat: 输入特征，如果为None则使用图中的'feat'特征
            
        Returns:
            torch.Tensor: 更新后的节点特征
        """
        with graph.local_scope():
            # 处理输入
            if feat is None:
                feat = graph.nodes['note'].data.get('feat')
            
            if feat is None:
                logger.error("图中没有'feat'特征，无法进行前向传播")
                return None
                
            # 检查输入特征维度
            if feat.dim() != 2:
                logger.error(f"输入特征维度不正确: feat.shape={feat.shape}")
                return None
                
            if feat.size(1) != self.hidden_dim:
                logger.error(f"输入特征维度不匹配: feat.size(1)={feat.size(1)}, expected={self.hidden_dim}")
                return None
            
            # 处理特征
            h = self._process_features(feat)
            
            # 计算查询、键、值
            head_dim = self.out_feats // self.num_heads
            try:
                q = self.q_proj(h).view(-1, self.num_heads, head_dim)
                k = self.k_proj(h).view(-1, self.num_heads, head_dim)
                v = self.v_proj(h).view(-1, self.num_heads, head_dim)
            except RuntimeError as e:
                logger.error(f"计算QKV时出错: {str(e)}")
                return None
            
            # 保存原始特征用于残差连接
            h_orig = h
            
            # 对每种边类型应用注意力机制
            h_list = []
            dst_nodes = set()  # 记录目标节点
            
            # 检查图是否为异构图
            if isinstance(graph, dgl.DGLHeteroGraph):
                # 处理异构图
                for etype in self.etypes:
                    if etype in graph.canonical_etypes:
                        src_type, edge_type, dst_type = etype
                        
                        # 检查是否有边
                        if graph.num_edges(etype) == 0:
                            continue
                        
                        src, dst = graph.edges(etype=etype)
                        dst_nodes.update(dst.tolist())
                        
                        try:
                            attn = self.compute_attention(q[dst], k[src])
                            h_etype = torch.matmul(attn, v[src])
                            h_etype = h_etype.view(-1, self.out_feats)  # 直接调整为输出维度
                            h_list.append((h_etype, dst))
                        except RuntimeError as e:
                            logger.error(f"处理边类型 {etype} 时出错: {str(e)}")
                            continue
            else:
                # 处理同构图
                src, dst = graph.edges()
                dst_nodes.update(dst.tolist())
                
                try:
                    attn = self.compute_attention(q[dst], k[src])
                    h_homo = torch.matmul(attn, v[src])
                    h_homo = h_homo.view(-1, self.out_feats)  # 直接调整为输出维度
                    h_list.append((h_homo, dst))
                except RuntimeError as e:
                    logger.error(f"处理同构图时出错: {str(e)}")
                    return h_orig
            
            # 如果没有边，返回原始特征
            if not h_list:
                return h_orig
            
            # 创建一个与原始特征相同大小的张量，用于存储更新后的特征
            h_updated = h_orig.clone()
            
            # 更新每个目标节点的特征
            for h_part, dst_indices in h_list:
                h_updated[dst_indices] = h_part
            
            # 应用第一个残差连接和层归一化
            try:
                h = self.norm1(h_orig + h_updated * self.res_scale)
                h_ffn = self.ffn(h)
                h = self.norm2(h + h_ffn * self.res_scale)
            except RuntimeError as e:
                logger.error(f"应用残差连接和层归一化时出错: {str(e)}")
                return h_orig
            
            return h


class EnhancedGATv2Model(nn.Module):
    """增强版GATv2模型"""
    
    def __init__(self, in_feats, hidden_dim, num_heads, num_layers, dropout=0.2, feat_drop=0.1, attn_drop=0.1, etypes=None):
        """初始化
        
        Args:
            in_feats: 输入特征维度
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            num_layers: GAT层数
            dropout: Dropout比率
            feat_drop: 特征dropout率
            attn_drop: 注意力dropout率
            etypes: 边类型列表
        """
        super().__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.etypes = etypes or [('note', 'temporal', 'note')]
        
        # 灵活的输入特征转换层
        self.input_transform = None
        
        # GAT层
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(
            EnhancedGATv2Conv(
                in_feats=in_feats,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                feat_drop=feat_drop,
                attn_drop=attn_drop
            )
        )
        
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(
                EnhancedGATv2Conv(
                    in_feats=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    feat_drop=feat_drop,
                    attn_drop=attn_drop
                )
            )
        
        # 最后一层
        self.layers.append(
            EnhancedGATv2Conv(
                in_feats=hidden_dim,
                hidden_dim=hidden_dim,
                num_heads=1,
                dropout=dropout,
                feat_drop=feat_drop,
                attn_drop=attn_drop
            )
        )
        
        # 解码器
        self.pitch_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 88)  # 88个音高
        )
        
        self.duration_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)  # 32种时值
        )
        
        self.velocity_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32)  # 32种力度
        )
    
    def forward(self, graph, feat):
        """前向传播
        
        Args:
            graph: DGL图
            feat: 输入特征
            
        Returns:
            dict: 包含节点特征和预测结果的字典
        """
        if feat is None:
            logger.error("输入特征为空")
            return None
            
        try:
            # 检查输入特征维度并动态创建转换层（如果需要）
            if feat.size(1) != self.in_feats:
                logger.info(f"输入特征维度调整: {feat.size(1)} -> {self.in_feats}")
                
                # 首次遇到不匹配的特征时创建转换层
                if self.input_transform is None or self.input_transform[0].in_features != feat.size(1):
                    logger.info(f"创建新的特征转换层: {feat.size(1)} -> {self.hidden_dim} -> {self.in_feats}")
                    self.input_transform = nn.Sequential(
                        nn.Linear(feat.size(1), self.hidden_dim // 2),
                        nn.ReLU(),
                        nn.Dropout(self.dropout.p),
                        nn.Linear(self.hidden_dim // 2, self.in_feats)
                    ).to(feat.device)
                
                feat = self.input_transform(feat)
            
            h = feat
            
            # 应用GAT层
            for i, layer in enumerate(self.layers):
                h_new = layer(graph, h)
                if h_new is None:
                    logger.error(f"第 {i+1} 层GAT处理失败")
                    return None
                h = h_new
                if i < len(self.layers) - 1:  # 非最后一层
                    h = F.elu(h)
                    h = self.dropout(h)
            
            # 生成预测
            pitch_logits = self.pitch_decoder(h)
            duration_logits = self.duration_decoder(h)
            velocity_logits = self.velocity_decoder(h)
            
            return {
                'node_features': h,
                'pitch_logits': pitch_logits,
                'duration_logits': duration_logits,
                'velocity_logits': velocity_logits
            }
            
        except Exception as e:
            logger.error(f"GAT模型处理失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def forward_batch(self, graphs, feats):
        """处理批量数据
        
        Args:
            graphs: 图列表
            feats: 特征列表
            
        Returns:
            tuple: (处理后的图列表, 输出特征列表)
        """
        outputs = []
        for g, f in zip(graphs, feats):
            outputs.append(self.forward(g, f))
        return outputs

    def preprocess_features(self, graph):
        """预处理图特征
        
        Args:
            graph: DGL图
            
        Returns:
            dgl.DGLGraph: 处理后的图
        """
        # 检查图是否为批处理图
        if isinstance(graph, dgl.DGLHeteroGraph):
            graphs = [graph]
        elif isinstance(graph, list):
            graphs = graph
        else:
            try:
                graphs = dgl.unbatch(graph)
            except:
                logger.error(f"无法处理输入类型: {type(graph)}")
                return None
        
        # 处理每个图
        for i, g in enumerate(graphs):
            # 检查是否有'feat'特征
            if 'note' in g.ntypes and 'feat' not in g.nodes['note'].data:
                # 创建特征向量
                num_nodes = g.num_nodes('note')
                if num_nodes > 0:
                    # 使用其他特征创建特征向量
                    features = []
                    
                    # 添加位置特征
                    if 'position' in g.nodes['note'].data:
                        pos = g.nodes['note'].data['position']
                        if pos.dim() == 1:
                            pos = pos.unsqueeze(1)
                        features.append(pos)
                    else:
                        features.append(torch.zeros(num_nodes, 1, device=g.device))
                    
                    # 添加音高特征
                    if 'pitch' in g.nodes['note'].data:
                        pitch = g.nodes['note'].data['pitch']
                        if pitch.dim() == 1:
                            pitch = pitch.unsqueeze(1)
                        # 归一化音高
                        pitch = (pitch - 60.0) / 24.0  # 归一化到[-1, 1]左右
                        features.append(pitch)
                    else:
                        features.append(torch.zeros(num_nodes, 1, device=g.device))
                    
                    # 添加时值特征
                    if 'duration' in g.nodes['note'].data:
                        duration = g.nodes['note'].data['duration']
                        if duration.dim() == 1:
                            duration = duration.unsqueeze(1)
                        # 归一化时值
                        duration = torch.log1p(duration) / 2.0  # 对数归一化
                        features.append(duration)
                    else:
                        features.append(torch.zeros(num_nodes, 1, device=g.device))
                    
                    # 添加力度特征
                    if 'velocity' in g.nodes['note'].data:
                        velocity = g.nodes['note'].data['velocity']
                        if velocity.dim() == 1:
                            velocity = velocity.unsqueeze(1)
                        # 归一化力度
                        velocity = velocity / 127.0  # 归一化到[0, 1]
                        features.append(velocity)
                    else:
                        features.append(torch.zeros(num_nodes, 1, device=g.device))
                    
                    # 合并特征
                    combined_features = torch.cat(features, dim=1)
                    
                    # 创建初始特征向量
                    g.nodes['note'].data['feat'] = combined_features
        
        # 返回处理后的图
        if isinstance(graph, dgl.DGLHeteroGraph):
            return graphs[0]
        elif isinstance(graph, list):
            return graphs
        else:
            return dgl.batch(graphs) 