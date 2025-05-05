import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from typing import Dict, List, Tuple, Optional

class NanyinContrastiveLearning(nn.Module):
    """南音对比学习模块
    
    使用对比学习方法让模型自动发现数据中的模式，而不是依赖于显式的规则。
    主要思想是：
    1. 对同一个音乐片段的不同视角（如不同增强方式）应该在特征空间中接近
    2. 不同音乐片段的特征应该在特征空间中远离
    3. 通过这种方式，模型可以自动学习到音乐的内在结构和模式
    """
    
    def __init__(self, config: Dict):
        """初始化对比学习模块
        
        Args:
            config: 配置字典，包含对比学习的参数
        """
        super().__init__()
        
        # 获取配置参数
        self.hidden_dim = config.get('hidden_dim', 128)
        self.temperature = config.get('temperature', 0.1)
        self.queue_size = config.get('queue_size', 8192)
        self.momentum = config.get('momentum', 0.999)
        
        # 创建编码器
        self.encoder_q = self._build_encoder()
        self.encoder_k = self._build_encoder()
        
        # 初始化动量编码器
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # 初始化队列和指针
        self.register_buffer("queue", torch.randn(self.hidden_dim, self.queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # 创建投影头
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 数据增强方法
        self.augmentations = [
            self._pitch_shift,
            self._time_stretch,
            self._velocity_change,
            self._add_noise,
            self._drop_notes
        ]
    
    def _build_encoder(self) -> nn.Module:
        """构建编码器
        
        Returns:
            nn.Module: 编码器模型
        """
        # 这里可以使用GNN或其他适合音乐数据的编码器
        # 简单起见，这里使用一个多层感知机
        return nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """更新动量编码器"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列
        
        Args:
            keys: 编码后的特征
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的键
        if ptr + batch_size <= self.queue_size:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            # 处理队列环绕
            first_part = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first_part].T
            self.queue[:, :batch_size - first_part] = keys[first_part:].T
        
        # 更新指针
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr
    
    def _pitch_shift(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """音高偏移增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return graph
        
        # 复制图以避免修改原始图
        new_graph = graph.clone()
        
        # 随机音高偏移量，范围为[-2, 2]半音
        shift = torch.randint(-2, 3, (1,), device=graph.device).item()
        
        # 应用音高偏移
        if 'pitch' in new_graph.nodes['note'].data:
            pitches = new_graph.nodes['note'].data['pitch']
            new_pitches = pitches + shift
            # 确保音高在有效范围内
            new_pitches = torch.clamp(new_pitches, min=21, max=108)
            new_graph.nodes['note'].data['pitch'] = new_pitches
        
        return new_graph
    
    def _time_stretch(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """时间伸缩增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return graph
        
        # 复制图以避免修改原始图
        new_graph = graph.clone()
        
        # 随机时间伸缩因子，范围为[0.9, 1.1]
        stretch_factor = 0.9 + 0.2 * torch.rand(1, device=graph.device).item()
        
        # 应用时间伸缩
        if 'position' in new_graph.nodes['note'].data and 'duration' in new_graph.nodes['note'].data:
            positions = new_graph.nodes['note'].data['position']
            durations = new_graph.nodes['note'].data['duration']
            
            # 伸缩位置和持续时间
            new_positions = positions * stretch_factor
            new_durations = durations * stretch_factor
            
            new_graph.nodes['note'].data['position'] = new_positions
            new_graph.nodes['note'].data['duration'] = new_durations
        
        return new_graph
    
    def _velocity_change(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """力度变化增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return graph
        
        # 复制图以避免修改原始图
        new_graph = graph.clone()
        
        # 随机力度变化因子，范围为[0.8, 1.2]
        velocity_factor = 0.8 + 0.4 * torch.rand(1, device=graph.device).item()
        
        # 应用力度变化
        if 'velocity' in new_graph.nodes['note'].data:
            velocities = new_graph.nodes['note'].data['velocity']
            new_velocities = velocities * velocity_factor
            # 确保力度在有效范围内
            new_velocities = torch.clamp(new_velocities, min=1, max=127)
            new_graph.nodes['note'].data['velocity'] = new_velocities
        
        return new_graph
    
    def _add_noise(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """添加噪声增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return graph
        
        # 复制图以避免修改原始图
        new_graph = graph.clone()
        
        # 添加特征噪声
        if 'feat' in new_graph.nodes['note'].data:
            feats = new_graph.nodes['note'].data['feat']
            noise = torch.randn_like(feats) * 0.05  # 5%的噪声
            new_feats = feats + noise
            new_graph.nodes['note'].data['feat'] = new_feats
        
        return new_graph
    
    def _drop_notes(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """随机丢弃音符增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return graph
        
        # 复制图以避免修改原始图
        new_graph = graph.clone()
        
        # 随机丢弃概率，范围为[0, 0.1]
        drop_prob = 0.1 * torch.rand(1, device=graph.device).item()
        
        # 随机丢弃音符
        num_nodes = new_graph.num_nodes('note')
        drop_mask = torch.rand(num_nodes, device=graph.device) >= drop_prob
        
        # 如果所有音符都被丢弃，保留至少一个
        if not drop_mask.any():
            drop_mask[0] = True
        
        # 创建子图
        new_graph = dgl.node_subgraph(new_graph, {'note': drop_mask})
        
        return new_graph
    
    def _augment_graph(self, graph: dgl.DGLHeteroGraph) -> Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]:
        """对图进行两种不同的增强
        
        Args:
            graph: 输入图
            
        Returns:
            Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]: 两种不同增强后的图
        """
        # 随机选择两种不同的增强方法
        aug_indices = torch.randperm(len(self.augmentations))[:2]
        aug1, aug2 = self.augmentations[aug_indices[0]], self.augmentations[aug_indices[1]]
        
        # 应用增强
        graph1 = aug1(graph)
        graph2 = aug2(graph)
        
        return graph1, graph2
    
    def _extract_features(self, graph: dgl.DGLHeteroGraph) -> torch.Tensor:
        """从图中提取特征
        
        Args:
            graph: 输入图
            
        Returns:
            torch.Tensor: 提取的特征
        """
        device = graph.device
        
        # 检查图是否有note节点类型
        if 'note' not in graph.ntypes:
            # 如果没有note节点，返回零向量
            return torch.zeros(self.hidden_dim, device=device)
            
        num_nodes = graph.num_nodes('note')
        
        if num_nodes == 0:
            # 如果没有节点，返回零向量
            return torch.zeros(self.hidden_dim, device=device)
        
        if 'feat' in graph.nodes['note'].data:
            # 如果已经有特征，直接使用
            node_feats = graph.nodes['note'].data['feat']
            
            # 确保特征是浮点类型
            node_feats = node_feats.float()
            
            # 确保特征维度正确
            if node_feats.dim() == 1:
                node_feats = node_feats.unsqueeze(1)  # 转为 [num_nodes, 1]
                
            if node_feats.size(1) != self.hidden_dim:
                # 创建投影层调整维度
                projection = nn.Linear(node_feats.size(1), self.hidden_dim, device=device)
                node_feats = projection(node_feats)
        else:
            # 获取原始特征
            pitches = graph.nodes['note'].data.get('pitch', torch.zeros(num_nodes, device=device)).float()
            positions = graph.nodes['note'].data.get('position', torch.arange(num_nodes, device=device)).float()
            durations = graph.nodes['note'].data.get('duration', torch.ones(num_nodes, device=device)).float()
            velocities = graph.nodes['note'].data.get('velocity', torch.ones(num_nodes, device=device) * 64).float()
            
            # 归一化
            pitches = (pitches - 60.0) / 24.0  # 归一化到[-1, 1]左右
            
            # 确保所有特征都是二维的
            if pitches.dim() == 1:
                pitches = pitches.view(num_nodes, 1)
            if positions.dim() == 1:
                positions = positions.view(num_nodes, 1)
            if durations.dim() == 1:
                durations = durations.view(num_nodes, 1)
            if velocities.dim() == 1:
                velocities = velocities.view(num_nodes, 1)
            
            # 合并特征
            node_feats = torch.cat([pitches, positions, durations, velocities], dim=1)
            
            # 创建一个线性层将4维特征投影到隐藏维度
            projection = nn.Linear(4, self.hidden_dim, device=device)
            node_feats = projection(node_feats)
        
        # 将特征赋值给图的feat属性
        graph.nodes['note'].data['feat'] = node_feats
        
        # 全局平均池化
        graph_feat = dgl.mean_nodes(graph, 'feat', ntype='note')
        
        # 确保输出维度正确
        if graph_feat.dim() == 1:
            # 如果是一维向量，确保长度为hidden_dim
            if graph_feat.size(0) != self.hidden_dim:
                # 调整维度
                if graph_feat.size(0) < self.hidden_dim:
                    graph_feat = F.pad(graph_feat, (0, self.hidden_dim - graph_feat.size(0)))
                else:
                    graph_feat = graph_feat[:self.hidden_dim]
        
        return graph_feat
    
    def forward(self, graphs: List[dgl.DGLHeteroGraph]) -> Dict:
        """前向传播
        
        Args:
            graphs: 输入图列表
            
        Returns:
            Dict: 包含对比损失和其他信息的字典
        """
        device = next(self.parameters()).device
        batch_size = len(graphs)
        
        # 结果字典
        result = {
            'loss': torch.tensor(0.0, device=device, requires_grad=True),
            'accuracy': 0.0
        }
        
        if batch_size == 0:
            return result
        
        # 对每个图进行两种不同的增强
        q_graphs = []
        k_graphs = []
        
        for graph in graphs:
            try:
                graph1, graph2 = self._augment_graph(graph)
                q_graphs.append(graph1)
                k_graphs.append(graph2)
            except Exception as e:
                print(f"图增强出错: {str(e)}")
                # 如果增强失败，使用原始图
                q_graphs.append(graph)
                k_graphs.append(graph)
        
        try:
            # 提取特征
            q_feats = []
            for g in q_graphs:
                feat = self._extract_features(g)
                # 确保特征是二维的 [1, hidden_dim]
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                q_feats.append(feat)
            
            # 将特征拼接成批次
            q_feats = torch.cat(q_feats, dim=0)  # [batch_size, hidden_dim]
            
            # 编码查询特征
            q = self.encoder_q(q_feats)
            q = self.projector(q)
            q = F.normalize(q, dim=1)
            
            # 计算键特征
            with torch.no_grad():
                # 更新动量编码器
                self._momentum_update_key_encoder()
                
                # 提取特征
                k_feats = []
                for g in k_graphs:
                    feat = self._extract_features(g)
                    # 确保特征是二维的 [1, hidden_dim]
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)
                    k_feats.append(feat)
                
                # 将特征拼接成批次
                k_feats = torch.cat(k_feats, dim=0)  # [batch_size, hidden_dim]
                
                # 编码键特征
                k = self.encoder_k(k_feats)
                k = self.projector(k)
                k = F.normalize(k, dim=1)
            
            # 计算正样本对的相似度
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            
            # 计算负样本对的相似度
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            
            # 计算InfoNCE损失
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits /= self.temperature
            
            # 正样本对的标签为0
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            
            # 计算交叉熵损失
            loss = F.cross_entropy(logits, labels)
            
            # 计算准确率
            pred = torch.argmax(logits, dim=1)
            acc = (pred == labels).float().mean()
            
            # 更新队列
            self._dequeue_and_enqueue(k)
            
            # 更新结果
            result['loss'] = loss
            result['accuracy'] = acc.item()
            
        except Exception as e:
            print(f"对比学习前向传播出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回零损失
            result['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
            result['accuracy'] = 0.0
        
        return result
    
    def extract_representations(self, graphs: List[dgl.DGLHeteroGraph]) -> torch.Tensor:
        """提取图的表示
        
        Args:
            graphs: 输入图列表
            
        Returns:
            torch.Tensor: 图的表示
        """
        if not graphs:
            return torch.zeros(0, self.hidden_dim, device=next(self.parameters()).device)
        
        # 提取特征
        feats = torch.cat([self._extract_features(g).unsqueeze(0) for g in graphs], dim=0)
        
        # 编码特征
        with torch.no_grad():
            representations = self.encoder_q(feats)
        
        return representations 