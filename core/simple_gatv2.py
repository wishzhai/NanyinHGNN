import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import math

class SimpleGATv2Layer(nn.Module):
    """简化版的GATv2卷积层，只处理note节点类型"""
    
    def __init__(self, in_feats, out_feats, num_heads=4, feat_drop=0.2, attn_drop=0.1):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.head_dim = out_feats // num_heads
        
        # 特征转换层（处理输入特征）
        self.feat_proj = nn.Linear(in_feats, out_feats)
        
        # 多头注意力
        self.q_proj = nn.Linear(out_feats, out_feats)
        self.k_proj = nn.Linear(out_feats, out_feats)
        self.v_proj = nn.Linear(out_feats, out_feats)
        
        # 注意力正则化参数
        self.attn_scale = nn.Parameter(torch.FloatTensor([0.1]))
        self.attn_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1))
        
        # Dropout层
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出变换
        self.out_proj = nn.Linear(out_feats, out_feats)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(out_feats)
        self.norm2 = nn.LayerNorm(out_feats)
        
        # 前馈网络（使用两层，增加表达能力）
        self.ffn = nn.Sequential(
            nn.Linear(out_feats, out_feats * 4),
            nn.GELU(),  # 使用GELU激活函数
            nn.Dropout(feat_drop),
            nn.Linear(out_feats * 4, out_feats),
            nn.Dropout(feat_drop)
        )
        
        # 残差缩放因子
        self.res_scale = nn.Parameter(torch.FloatTensor([0.1]))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.feat_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.q_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.k_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.v_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.out_proj.weight, gain=gain)
        
        # 初始化注意力参数
        nn.init.constant_(self.attn_scale, 0.1)
        nn.init.zeros_(self.attn_bias)
        nn.init.constant_(self.res_scale, 0.1)
    
    def forward(self, g):
        """前向传播
        
        Args:
            g: 异构图，需要包含'note'节点类型
            
        Returns:
            g: 处理后的图
        """
        with g.local_scope():
            # 验证图中有'note'节点
            if 'note' not in g.ntypes:
                print("错误: 图中没有'note'节点类型")
                return g
                
            # 验证'note'节点有'feat'特征
            if 'feat' not in g.nodes['note'].data:
                print("错误: 'note'节点缺少'feat'特征")
                return g
                
            # 获取节点特征
            h = g.nodes['note'].data['feat']
            identity = h  # 保存原始特征用于残差连接
            
            # 特征投影
            h = self.feat_proj(h)
            h = F.gelu(h)  # 使用GELU激活函数
            h = self.feat_drop(h)
            
            # 多头注意力计算
            batch_size = h.size(0)
            
            # 计算Q、K、V
            q = self.q_proj(h).view(batch_size, -1, self.num_heads, self.head_dim)
            k = self.k_proj(h).view(batch_size, -1, self.num_heads, self.head_dim)
            v = self.v_proj(h).view(batch_size, -1, self.num_heads, self.head_dim)
            
            # 转置以便进行注意力计算
            q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            k = k.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, seq_len, head_dim]
            
            # 计算注意力分数
            scale = math.sqrt(self.head_dim) * self.attn_scale
            attn = torch.matmul(q, k.transpose(-2, -1)) / scale + self.attn_bias
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            
            # 加权求和
            h_out = torch.matmul(attn, v)  # [batch_size, num_heads, seq_len, head_dim]
            h_out = h_out.permute(0, 2, 1, 3).contiguous()  # [batch_size, seq_len, num_heads, head_dim]
            h_out = h_out.view(batch_size, -1, self.out_feats)  # [batch_size, seq_len, out_feats]
            
            # 输出投影
            h_out = self.out_proj(h_out)
            
            # 第一个残差连接与层归一化
            h = self.norm1(h + h_out * self.res_scale)
            
            # 前馈网络
            h_ffn = self.ffn(h)
            
            # 第二个残差连接与层归一化
            h = self.norm2(h + h_ffn * self.res_scale)
            
            # 最后的残差连接（如果输入输出维度相同）
            if h.shape == identity.shape:
                h = h + identity * self.res_scale
            
            # 更新节点特征
            g.nodes['note'].data['feat'] = h
            
            return g


class SimpleGATv2Model(nn.Module):
    """简化版的南音GATv2模型"""
    
    def __init__(self, in_feats=4, hidden_dim=128, num_heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        self.in_feats = in_feats
        self.hidden_dim = hidden_dim
        
        # 输入特征变换
        self.input_transform = nn.Sequential(
            nn.Linear(in_feats, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # GATv2层堆叠
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                SimpleGATv2Layer(
                    in_feats=hidden_dim,
                    out_feats=hidden_dim,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout
                )
            )
    
    def preprocess_features(self, graph):
        """预处理节点特征
        
        Args:
            graph: 输入图
            
        Returns:
            处理后的图
        """
        # 验证图中有'note'节点
        if 'note' not in graph.ntypes:
            return graph
        
        # 获取设备
        device = graph.device
            
        # 初始化必要的特征
        if 'feat' not in graph.nodes['note'].data:
            # 如果没有feat特征，检查是否有基础特征
            if all(f in graph.nodes['note'].data for f in ['pitch', 'duration', 'velocity', 'position']):
                # 如果有基础特征，从中构建feat
                pitch = graph.nodes['note'].data['pitch'].float().view(-1, 1)
                duration = graph.nodes['note'].data['duration'].float().view(-1, 1)
                velocity = graph.nodes['note'].data['velocity'].float().view(-1, 1)
                position = graph.nodes['note'].data['position'].float().view(-1, 1)
                
                # 合并特征
                feat = torch.cat([pitch, duration, velocity, position], dim=1)
                graph.nodes['note'].data['feat'] = feat
            else:
                # 如果没有基础特征，创建零特征（使用正确的设备）
                num_nodes = graph.num_nodes('note')
                graph.nodes['note'].data['feat'] = torch.zeros((num_nodes, self.in_feats), device=device)
                print(f"警告: 为'note'节点创建了零特征，设备: {device}")
        
        # 转换特征维度
        feat = graph.nodes['note'].data['feat']
        if feat.dim() == 1:
            feat = feat.view(-1, 1)
        
        # 确保特征维度正确
        if feat.size(1) != self.in_feats:
            # 简单调整维度（实际应用中可能需要更复杂的方法）
            if feat.size(1) < self.in_feats:
                # 特征扩展（确保使用正确的设备）
                padded = torch.zeros((feat.size(0), self.in_feats), device=feat.device)
                padded[:, :feat.size(1)] = feat
                feat = padded
            else:
                # 特征裁剪
                feat = feat[:, :self.in_feats]
            graph.nodes['note'].data['feat'] = feat
        
        # 应用特征转换
        h = self.input_transform(graph.nodes['note'].data['feat'])
        graph.nodes['note'].data['feat'] = h
        
        return graph
    
    def forward(self, graph):
        """模型前向传播
        
        Args:
            graph: 输入图或图列表
            
        Returns:
            处理后的图或图列表
        """
        # 处理可能的批处理情况
        if isinstance(graph, list):
            return [self.forward_single(g) for g in graph]
        else:
            return self.forward_single(graph)
    
    def forward_single(self, graph):
        """处理单个图
        
        Args:
            graph: 单个输入图
            
        Returns:
            处理后的图
        """
        # 预处理特征
        graph = self.preprocess_features(graph)
        
        # 依次通过每一层
        for layer in self.layers:
            graph = layer(graph)
        
        return graph 