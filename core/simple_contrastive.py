import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
import logging
from typing import Dict, List, Tuple
import random

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleContrastiveLearning(nn.Module):
    """简化版南音对比学习模块
    
    使用对比学习方法让模型自动发现数据中的模式，而不是依赖于显式的规则。
    主要思想是：
    1. 对同一个音乐片段的不同视角（如不同增强方式）应该在特征空间中接近
    2. 不同音乐片段的特征应该在特征空间中远离
    """
    
    def __init__(self, config: Dict):
        """初始化对比学习模块
        
        Args:
            config: 配置字典，包含对比学习的参数
        """
        super().__init__()
        
        # 获取配置参数
        self.hidden_dim = config.get('hidden_dim', 128)
        self.temperature = config.get('temperature', 0.07)  # 降低温度
        self.queue_size = config.get('queue_size', 4096)   # 减小队列大小
        self.momentum = config.get('momentum', 0.999)
        
        # 输入特征维度
        self.input_dim = config.get('input_dim', 128)
        
        # 数据增强配置
        self.augment_prob = config.get('augment_prob', 0.8)
        self.augment_methods = config.get('augment_methods', ['pitch_shift', 'time_stretch'])
        
        # 增强特征提取器
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),  # 增加dropout
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 动量编码器
        self.momentum_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 初始化动量编码器
        for param_q, param_k in zip(self.encoder.parameters(), 
                                  self.momentum_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        # 特征队列
        self.register_buffer("queue", torch.randn(self.queue_size, self.hidden_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # 创建投影头
        self.projector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        logger.info(f"初始化SimpleContrastiveLearning，输入维度: {self.input_dim}, 隐藏维度: {self.hidden_dim}, 队列大小: {self.queue_size}")
    
    def _pitch_shift(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """音高偏移增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes or 'pitch' not in graph.nodes['note'].data:
            return graph
        
        # 复制图
        new_graph = graph.clone()
        
        # 随机选择偏移量 (-3, -2, -1, 1, 2, 3)
        shifts = [-3, -2, -1, 1, 2, 3]
        shift = shifts[torch.randint(0, len(shifts), (1,)).item()]
        
        # 应用偏移
        pitches = new_graph.nodes['note'].data['pitch'].clone()
        pitches = pitches + shift
        
        # 确保音高在有效范围内
        pitches = torch.clamp(pitches, min=36, max=84)
        
        # 更新图
        new_graph.nodes['note'].data['pitch'] = pitches
        
        return new_graph
    
    def _time_stretch(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """时间拉伸增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes or 'position' not in graph.nodes['note'].data:
            return graph
        
        # 复制图
        new_graph = graph.clone()
        
        # 随机选择拉伸因子 (0.8, 0.9, 1.1, 1.2)
        factors = [0.8, 0.9, 1.1, 1.2]
        factor = factors[torch.randint(0, len(factors), (1,)).item()]
        
        # 应用拉伸
        positions = new_graph.nodes['note'].data['position'].clone()
        positions = positions * factor
        
        # 更新图
        new_graph.nodes['note'].data['position'] = positions
        
        return new_graph
    
    def _velocity_change(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """力度变化增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes or 'velocity' not in graph.nodes['note'].data:
            return graph
        
        # 复制图
        new_graph = graph.clone()
        
        # 随机选择力度变化范围 (0.8-1.2)
        factor = 0.8 + torch.rand(1).item() * 0.4
        
        # 应用力度变化
        velocities = new_graph.nodes['note'].data['velocity']
        new_velocities = velocities * factor
        
        # 确保在有效范围内 (1-127)
        new_velocities = torch.clamp(new_velocities, min=1, max=127)
        
        # 更新图
        new_graph.nodes['note'].data['velocity'] = new_velocities
        
        return new_graph
    
    def _add_noise(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """添加特征噪声增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 增强后的图
        """
        if 'note' not in graph.ntypes:
            return graph
        
        # 复制图
        new_graph = graph.clone()
        
        # 为每个特征添加小噪声
        for feat_name in ['pitch', 'velocity', 'duration']:
            if feat_name in new_graph.nodes['note'].data:
                feat = new_graph.nodes['note'].data[feat_name].float()
                noise = torch.randn_like(feat) * 0.05  # 5%的噪声
                new_feat = feat + noise
                
                # 根据特征类型进行范围限制
                if feat_name == 'pitch':
                    new_feat = torch.clamp(new_feat, min=36, max=84)
                elif feat_name == 'velocity':
                    new_feat = torch.clamp(new_feat, min=1, max=127)
                elif feat_name == 'duration':
                    new_feat = torch.clamp(new_feat, min=0)
                    
                new_graph.nodes['note'].data[feat_name] = new_feat
        
        return new_graph
    
    def _augment_graph(self, graph: dgl.DGLHeteroGraph) -> Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]:
        """对图进行两种不同的增强
        
        Args:
            graph: 输入图
            
        Returns:
            Tuple[dgl.DGLHeteroGraph, dgl.DGLHeteroGraph]: 两种增强后的图
        """
        # 可用的增强方法
        method_map = {
            'pitch_shift': self._pitch_shift,
            'time_stretch': self._time_stretch,
            'velocity_change': self._velocity_change,
            'add_noise': self._add_noise
        }
        
        # 随机选择两种不同的增强方法
        available_methods = [method_map[m] for m in self.augment_methods if m in method_map]
        if len(available_methods) < 2:
            return graph.clone(), graph.clone()
        
        # 对每个视图应用1-2个随机增强
        def apply_random_augments(g):
            num_augments = torch.randint(1, 3, (1,)).item()  # 1或2个增强
            methods = random.sample(available_methods, num_augments)
            for method in methods:
                if torch.rand(1).item() < self.augment_prob:
                    g = method(g)
            return g
        
        return apply_random_augments(graph.clone()), apply_random_augments(graph.clone())
    
    def _extract_features(self, graph: dgl.DGLGraph) -> torch.Tensor:
        """从图中提取特征
        
        Args:
            graph: 输入图
            
        Returns:
            torch.Tensor: 提取的特征向量
        """
        if 'note' not in graph.ntypes:
            if 'feat' in graph.ndata:
                # 直接使用图中的特征
                node_feats = graph.ndata['feat']
                if node_feats.dim() == 2:  # [num_nodes, feat_dim]
                    # 使用平均池化得到图级特征
                    graph_feat = torch.mean(node_feats, dim=0)
                    return graph_feat
                else:
                    return node_feats
            return torch.zeros(self.input_dim, device=next(self.parameters()).device)
        
        num_nodes = graph.num_nodes('note')
        if num_nodes == 0:
            return torch.zeros(self.input_dim, device=next(self.parameters()).device)
        
        # 获取设备
        device = next(self.parameters()).device
        
        # 检查是否已经有特征
        if 'feat' in graph.nodes['note'].data:
            node_feats = graph.nodes['note'].data['feat']
            if node_feats.dim() == 2:  # [num_nodes, feat_dim]
                # 使用平均池化得到图级特征
                graph_feat = torch.mean(node_feats, dim=0)
                return graph_feat
            return node_feats
        
        # 如果没有预计算的特征，则构建基础特征
        pitches = graph.nodes['note'].data.get('pitch', torch.zeros(num_nodes, device=device)).float()
        positions = graph.nodes['note'].data.get('position', torch.arange(num_nodes, device=device)).float()
        durations = graph.nodes['note'].data.get('duration', torch.ones(num_nodes, device=device)).float()
        velocities = graph.nodes['note'].data.get('velocity', torch.ones(num_nodes, device=device) * 64).float()
        
        # 计算音程关系
        intervals = torch.zeros(num_nodes, device=device)
        if num_nodes > 1:
            intervals[1:] = pitches[1:] - pitches[:-1]
        
        # 计算节奏特征
        rhythm_ratios = torch.ones(num_nodes, device=device)
        if num_nodes > 1:
            rhythm_ratios[1:] = durations[1:] / (durations[:-1] + 1e-6)
        
        # 计算力度变化
        velocity_changes = torch.zeros(num_nodes, device=device)
        if num_nodes > 1:
            velocity_changes[1:] = velocities[1:] - velocities[:-1]
        
        # 计算音符密度
        window_size = 4.0
        note_density = torch.zeros(num_nodes, device=device)
        for i in range(num_nodes):
            start_time = positions[i]
            end_time = start_time + window_size
            mask = (positions >= start_time) & (positions < end_time)
            note_density[i] = mask.sum().float() / window_size
        
        # 归一化特征
        pitches = (pitches - 60.0) / 24.0
        positions = positions / (positions.max() + 1e-6)
        durations = durations / (durations.max() + 1e-6)
        velocities = velocities / 127.0
        intervals = intervals / 12.0
        rhythm_ratios = torch.clamp(rhythm_ratios, 0.1, 10.0) / 10.0
        velocity_changes = velocity_changes / 127.0
        note_density = note_density / note_density.max() if note_density.max() > 0 else note_density
        
        # 合并所有特征
        features_list = [
            pitches.view(num_nodes, 1),
            positions.view(num_nodes, 1),
            durations.view(num_nodes, 1),
            velocities.view(num_nodes, 1),
            intervals.view(num_nodes, 1),
            rhythm_ratios.view(num_nodes, 1),
            velocity_changes.view(num_nodes, 1),
            note_density.view(num_nodes, 1)
        ]
        
        # 合并特征
        node_feats = torch.cat(features_list, dim=1)
        
        # 投影到目标维度
        if not hasattr(self, 'raw_projection'):
            self.raw_projection = nn.Linear(len(features_list), self.input_dim, device=device)
        node_feats = self.raw_projection(node_feats)
        
        # 使用平均池化得到图级特征
        graph_feat = torch.mean(node_feats, dim=0)
        
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
        
        try:
            # 对每个图进行两种不同的增强
            q_graphs = []
            k_graphs = []
            
            for graph in graphs:
                try:
                    graph1, graph2 = self._augment_graph(graph)
                    q_graphs.append(graph1)
                    k_graphs.append(graph2)
                except Exception as e:
                    logger.warning(f"图增强出错: {str(e)}")
                    q_graphs.append(graph)
                    k_graphs.append(graph)
            
            # 提取特征
            q_feats = []
            for g in q_graphs:
                feat = self._extract_features(g)
                if feat.dim() == 1:
                    feat = feat.unsqueeze(0)
                q_feats.append(feat)
            
            # 将特征拼接成批次
            q_feats = torch.cat(q_feats, dim=0)
            
            # 编码查询特征
            q = self.encoder(q_feats)
            q = self.projector(q)
            # 使用clone避免原地操作
            q = F.normalize(q.clone(), dim=1)
            
            # 计算动量特征
            with torch.no_grad():
                self._momentum_update_key_encoder()
                k = self.momentum_encoder(q_feats)
                k = self.projector(k)
                k = F.normalize(k.clone(), dim=1)
            
            # 计算正样本对的相似度
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            
            # 使用clone避免修改queue
            queue = self.queue.clone()
            l_neg = torch.einsum('nc,ck->nk', [q, queue.t()])
            
            # 计算InfoNCE损失
            logits = torch.cat([l_pos, l_neg], dim=1)
            logits = logits / self.temperature
            
            # 正样本对的标签为0
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)
            
            # 计算交叉熵损失
            loss = F.cross_entropy(logits, labels)
            
            # 计算准确率
            with torch.no_grad():
                pred = torch.argmax(logits, dim=1)
                acc = (pred == labels).float().mean()
            
            # 更新队列
            self._dequeue_and_enqueue(k.detach())
            
            # 更新结果
            result['loss'] = loss
            result['accuracy'] = acc.item()
            
        except Exception as e:
            logger.error(f"对比学习前向传播出错: {str(e)}")
            import traceback
            traceback.print_exc()
            result['loss'] = torch.tensor(0.0, device=device, requires_grad=True)
            result['accuracy'] = 0.0
        
        return result

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """更新动量编码器"""
        for param_q, param_k in zip(self.encoder.parameters(), 
                                  self.momentum_encoder.parameters()):
            param_k.data = param_k.data * self.momentum + \
                          param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 创建新的队列避免原地操作
        new_queue = self.queue.clone()
        
        if ptr + batch_size > self.queue_size:
            new_queue[ptr:] = keys[:self.queue_size - ptr]
            new_queue[:batch_size - (self.queue_size - ptr)] = keys[self.queue_size - ptr:]
            ptr = batch_size - (self.queue_size - ptr)
        else:
            new_queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue_size
            
        # 更新队列
        self.queue.copy_(new_queue)
        self.queue_ptr[0] = ptr 