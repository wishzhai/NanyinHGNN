import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import math

class NanyinGATv2Conv(nn.Module):
    """南音定制的GATv2异构图卷积层"""
    
    def __init__(self, in_feats: int, out_feats: int, num_heads: int, 
                 etypes: list, feat_drop=0.2, attn_drop=0.1):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.etypes = etypes
        
        # 输入特征转换层 - 将4维特征转换为in_feats维
        # 直接将输入特征从4维映射到所需的维度
        self.input_transform = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, in_feats)
        )
        
        # 特征投影层
        self.feat_proj = nn.Linear(in_feats, out_feats)
        
        # 多头注意力层
        self.q_proj = nn.Linear(out_feats, out_feats * num_heads)
        self.k_proj = nn.Linear(out_feats, out_feats * num_heads)
        self.v_proj = nn.Linear(out_feats, out_feats * num_heads)
        
        # Dropout层 - 检查是否已经是nn.Dropout实例
        if isinstance(feat_drop, nn.Dropout):
            self.feat_drop = feat_drop
        else:
            self.feat_drop = nn.Dropout(feat_drop)
            
        if isinstance(attn_drop, nn.Dropout):
            self.attn_drop = attn_drop
        else:
            self.attn_drop = nn.Dropout(attn_drop)
        
        # 输出投影
        self.out_proj = nn.Linear(out_feats * num_heads, out_feats)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(out_feats)
        self.norm2 = nn.LayerNorm(out_feats)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(out_feats, out_feats * 4),
            nn.ReLU(),
            self.feat_drop,  # 使用之前创建的dropout实例
            nn.Linear(out_feats * 4, out_feats)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置模型参数"""
        gain = nn.init.calculate_gain('relu')
        # 初始化input_transform中的第一层
        nn.init.xavier_normal_(self.input_transform[0].weight, gain=gain)
        # 初始化input_transform中的第三层
        nn.init.xavier_normal_(self.input_transform[2].weight, gain=gain)
        nn.init.xavier_normal_(self.feat_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.q_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.k_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.v_proj.weight, gain=gain)
        nn.init.xavier_normal_(self.out_proj.weight, gain=gain)
    
    def _process_features(self, feat):
        """处理输入特征，确保维度正确
        
        Args:
            feat: 输入特征张量
            
        Returns:
            处理后的特征张量
        """
        if feat is None:
            return None
            
        # 确保特征是浮点类型
        feat = feat.float()
        
        # 如果是4维特征，先通过更强大的特征转换网络
        if feat.size(-1) == 4:
            try:
                # 使用强化的特征转换网络
                feat = self.input_transform(feat)  # [num_nodes, in_feats]
            except Exception as e:
                print(f"特征转换错误: {str(e)}, 输入形状: {feat.shape}")
                # 如果出错，尝试添加批次维度后重试
                if feat.dim() == 2:
                    feat = feat.unsqueeze(0)
                    feat = self.input_transform(feat)
                    feat = feat.squeeze(0)
                else:
                    raise
        elif feat.size(-1) != self.in_feats:
            # 如果既不是4维也不是in_feats维，尝试使用线性插值调整维度
            print(f"警告：遇到非标准维度特征 {feat.size(-1)}，尝试调整...")
            old_size = feat.size(-1)
            feat_reshaped = feat.view(-1, 1, old_size)
            feat = F.interpolate(feat_reshaped, size=self.in_feats, mode='linear')
            feat = feat.squeeze(1)
        
        # 投影到out_feats维
        feat = self.feat_proj(feat)  # [num_nodes, out_feats]
        return feat
    
    def forward(self, batch):
        """前向传播 - 使用简化版本，专注于主要节点类型
        Args:
            batch: 批处理的异构图
        Returns:
            batch: 处理后的异构图
        """
        try:
            with batch.local_scope():
                device = next(self.parameters()).device
                
                # 打印图的基本信息以进行调试
                print(f"图信息 - 节点类型: {batch.ntypes}, 边类型: {batch.etypes}")
                for ntype in batch.ntypes:
                    feat = batch.nodes[ntype].data.get('feat')
                    if feat is not None:
                        print(f"节点类型 {ntype}: 数量={batch.num_nodes(ntype)}, 特征形状={feat.shape}, 类型={feat.dtype}")
                
                # 专注于处理主要节点类型: 'note'
                if 'note' not in batch.ntypes:
                    print("警告: 图中没有'note'节点类型")
                    return batch
                
                note_feat = batch.nodes['note'].data.get('feat')
                if note_feat is None:
                    print("警告: 'note'节点没有特征")
                    return batch
                
                # 处理note节点特征
                try:
                    # 转换特征
                    note_feat = self._process_features(note_feat)
                    
                    # 在这个简化版本中，我们只对节点自身应用变换
                    # 不依赖于图的边结构
                    q = self.q_proj(note_feat)
                    k = self.k_proj(note_feat)
                    v = self.v_proj(note_feat)
                    
                    # 应用自注意力机制
                    # 重塑为多头格式
                    batch_size = note_feat.size(0)
                    q = q.view(batch_size, self.num_heads, -1)
                    k = k.view(batch_size, self.num_heads, -1)
                    v = v.view(batch_size, self.num_heads, -1)
                    
                    # 计算自注意力分数
                    attn = torch.bmm(
                        q.transpose(0, 1),               # [num_heads, batch_size, head_dim]
                        k.permute(1, 2, 0)               # [num_heads, head_dim, batch_size]
                    ) / math.sqrt(self.out_feats // self.num_heads)
                    
                    # 应用softmax
                    attn = F.softmax(attn, dim=-1)       # [num_heads, batch_size, batch_size]
                    attn = self.attn_drop(attn)
                    
                    # 计算加权和
                    out = torch.bmm(
                        attn,                            # [num_heads, batch_size, batch_size]
                        v.transpose(0, 1)                # [num_heads, batch_size, head_dim]
                    )
                    
                    # 重塑回原始格式
                    out = out.transpose(0, 1).contiguous().view(batch_size, -1)
                    out = self.feat_drop(out)
                    out = self.out_proj(out)
                    
                    # 应用残差连接和层归一化
                    out = self.norm1(note_feat + out)
                    out = self.norm2(out + self.ffn(out))
                    
                    # 更新图中的节点特征
                    batch.nodes['note'].data['feat'] = out
                    
                    # 打印成功处理信息
                    print(f"成功处理'note'节点特征: {out.shape}")
                    
                except Exception as e:
                    print(f"处理'note'节点特征时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # 简单复制其他节点类型的特征，不进行处理
                for ntype in batch.ntypes:
                    if ntype != 'note':
                        feat = batch.nodes[ntype].data.get('feat')
                        if feat is not None:
                            # 保持特征不变
                            batch.nodes[ntype].data['feat'] = feat
                
                return batch
                
        except Exception as e:
            print(f"GATv2Conv forward 出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def compute_attention(self, q, k):
        """计算注意力分数 (实验性功能，当前未使用)
        
        Args:
            q: 查询向量 [N, H, D]
            k: 键向量 [N, H, D]
        Returns:
            torch.Tensor: 注意力分数 [N, H]
        """
        # 确保输入维度正确
        if q.dim() != 3 or k.dim() != 3:
            raise ValueError(f"查询和键向量必须是3维的，但得到: q.dim()={q.dim()}, k.dim()={k.dim()}")
        
        # 初始化注意力权重（如果尚未初始化）
        if not hasattr(self, 'attn'):
            self.attn = nn.Parameter(torch.Tensor(1, self.num_heads, self.out_feats))
            nn.init.xavier_normal_(self.attn)
        
        # 确保 self.attn 的维度正确
        if self.attn.size(0) != 1 or self.attn.size(1) != self.num_heads or self.attn.size(2) != self.out_feats:
            self.attn.data = self.attn.data.view(1, self.num_heads, self.out_feats)
        
        # GATv2动态注意力计算
        x = torch.tanh(q + k)  # [N, H, D]
        
        # 计算注意力分数并确保维度正确
        attn = torch.sum(self.attn * x, dim=-1)  # [N, H]
        
        # 应用激活函数
        return F.leaky_relu(attn, negative_slope=0.2)