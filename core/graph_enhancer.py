import torch
import dgl
import logging
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphEnhancer:
    """图结构增强器，用于将简化图结构转换为增强的异构图结构"""
    
    def __init__(self, config: Dict = None):
        """初始化图结构增强器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.default_node_features = {
            'note': {
                'pitch': (1, 60.0),  # 默认中央C
                'duration': (1, 1.0),  # 默认1拍
                'velocity': (1, 64.0),  # 默认中等力度
                'position': (1, 0.0),  # 默认位置0
                'is_rest': (1, 0.0),  # 默认非休止符
                'is_ornament': (1, 0.0),  # 默认非装饰音
                'feat': (128, 0.0)  # 默认特征向量
            },
            'measure': {
                'position': (1, 0.0),  # 小节位置
                'tempo': (1, 80.0),  # 默认速度
                'feat': (64, 0.0)  # 默认特征向量
            },
            'phrase': {
                'position': (1, 0.0),  # 乐句位置
                'length': (1, 4.0),  # 默认长度4小节
                'feat': (64, 0.0)  # 默认特征向量
            }
        }
    
    def enhance_graph(self, graph: dgl.DGLGraph) -> dgl.DGLHeteroGraph:
        """将简化图结构转换为增强的异构图结构"""
        try:
            device = graph.device
            
            # 获取节点数量
            num_notes = graph.num_nodes('note')
            num_measures = max(1, num_notes // 4)  # 每4个音符一个小节
            num_phrases = max(1, num_measures // 4)  # 每4个小节一个乐句
            
            # 获取音符位置信息
            positions = graph.nodes['note'].data.get('position', torch.arange(num_notes, device=device).float())
            
            # 创建图数据字典
            graph_data = {
                ('note', 'temporal', 'note'): ([], []),
                ('note', 'decorate', 'ornament'): ([], []),
                ('note', 'in', 'measure'): ([], []),
                ('measure', 'contains', 'note'): ([], []),
                ('measure', 'in', 'phrase'): ([], []),
                ('phrase', 'contains', 'measure'): ([], [])
            }
            
            # 创建节点数量字典
            num_nodes_dict = {
                'note': num_notes,
                'measure': num_measures,
                'phrase': num_phrases,
                'ornament': 0  # 初始化装饰音节点数为0
            }
            
            # 创建新图，确保在与原图相同的设备上
            new_graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict, device=device)
            
            # 复制音符节点特征
            for feat_name in graph.nodes['note'].data:
                new_graph.nodes['note'].data[feat_name] = graph.nodes['note'].data[feat_name]
            
            # 添加小节节点特征
            new_graph.nodes['measure'].data['position'] = torch.arange(0, num_measures, dtype=torch.float, device=device).unsqueeze(1)
            new_graph.nodes['measure'].data['tempo'] = torch.ones(num_measures, 1, device=device) * 80.0  # 默认速度
            new_graph.nodes['measure'].data['feat'] = torch.zeros(num_measures, 64, device=device)
            
            # 添加乐句节点特征
            new_graph.nodes['phrase'].data['position'] = torch.arange(0, num_phrases, dtype=torch.float, device=device).unsqueeze(1)
            new_graph.nodes['phrase'].data['length'] = torch.ones(num_phrases, 1, device=device) * 4.0  # 默认4小节
            new_graph.nodes['phrase'].data['feat'] = torch.zeros(num_phrases, 64, device=device)
            
            # 添加音符到小节的边
            for i in range(num_notes):
                pos = positions[i].item()
                measure_idx = min(int(pos // 4), num_measures - 1)
                new_graph.add_edges(i, measure_idx, etype=('note', 'in', 'measure'))
                new_graph.add_edges(measure_idx, i, etype=('measure', 'contains', 'note'))
            
            # 添加小节到乐句的边
            for i in range(num_measures):
                phrase_idx = min(i // 4, num_phrases - 1)
                new_graph.add_edges(i, phrase_idx, etype=('measure', 'in', 'phrase'))
                new_graph.add_edges(phrase_idx, i, etype=('phrase', 'contains', 'measure'))
            
            # 转换音符之间的边关系
            if ('note', 'to', 'note') in graph.canonical_etypes:
                src, dst = graph.edges(etype=('note', 'to', 'note'))
                new_graph.add_edges(src, dst, etype=('note', 'temporal', 'note'))
            
            # 添加装饰音关系
            if 'is_ornament' in graph.nodes['note'].data:
                ornament_mask = graph.nodes['note'].data['is_ornament'].squeeze()
                ornament_indices = []
                for i in range(num_notes):
                    if ornament_mask[i] > 0.5:  # 是装饰音
                        # 寻找这个装饰音对应的主音符
                        pos_i = positions[i].item()
                        for j in range(num_notes):
                            if ornament_mask[j] <= 0.5 and abs(positions[j].item() - pos_i) < 0.5:
                                # 创建新的装饰音节点
                                new_graph.add_nodes(1, ntype='ornament')
                                orn_idx = new_graph.num_nodes('ornament') - 1
                                ornament_indices.append(orn_idx)
                                
                                # 复制装饰音特征
                                for feat_name in ['pitch', 'duration', 'velocity', 'position']:
                                    if feat_name in graph.nodes['note'].data:
                                        if 'ornament' not in new_graph.nodes:
                                            new_graph.nodes['ornament'].data[feat_name] = graph.nodes['note'].data[feat_name][i].unsqueeze(0)
                                        else:
                                            new_graph.nodes['ornament'].data[feat_name] = torch.cat([
                                                new_graph.nodes['ornament'].data[feat_name],
                                                graph.nodes['note'].data[feat_name][i].unsqueeze(0)
                                            ])
                                
                                # 添加装饰音边
                                new_graph.add_edges(j, orn_idx, etype=('note', 'decorate', 'ornament'))
                                break
            
            # 确保所有特征都存在
            return self._ensure_all_features(new_graph)
            
        except Exception as e:
            logger.error(f"增强图结构时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return graph  # 出错时返回原图
    
    def _ensure_all_features(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """确保图中的所有节点都有必要的特征
        
        Args:
            graph: DGL异构图
            
        Returns:
            dgl.DGLHeteroGraph: 更新后的图
        """
        # 获取设备
        device = graph.device
        
        for ntype, features in self.default_node_features.items():
            if ntype in graph.ntypes:
                num_nodes = graph.num_nodes(ntype)
                if num_nodes == 0:
                    continue
                
                # 检查并添加缺失的特征
                for feat_name, (dim, default_value) in features.items():
                    if feat_name not in graph.nodes[ntype].data:
                        logger.info(f"为'{ntype}'节点添加缺失的'{feat_name}'特征")
                        graph.nodes[ntype].data[feat_name] = torch.ones(num_nodes, dim, device=device) * default_value
        
        return graph
    
    def batch_enhance_graphs(self, graphs: List[dgl.DGLGraph]) -> List[dgl.DGLHeteroGraph]:
        """批量增强图结构
        
        Args:
            graphs: 图列表
            
        Returns:
            List[dgl.DGLHeteroGraph]: 增强后的图列表
        """
        return [self.enhance_graph(g) for g in graphs]
    
    def log_graph_stats(self, graph: dgl.DGLHeteroGraph, prefix: str = "") -> None:
        """记录图统计信息
        
        Args:
            graph: DGL异构图
            prefix: 日志前缀
        """
        logger.info(f"{prefix} 图统计信息:")
        logger.info(f"  设备: {graph.device}")
        logger.info(f"  节点类型: {graph.ntypes}")
        logger.info(f"  边类型: {graph.canonical_etypes}")
        
        for ntype in graph.ntypes:
            logger.info(f"  {ntype}节点数: {graph.num_nodes(ntype)}")
            logger.info(f"  {ntype}节点特征: {list(graph.nodes[ntype].data.keys())}")
        
        for etype in graph.canonical_etypes:
            logger.info(f"  {etype}边数: {graph.num_edges(etype)}")
            if graph.num_edges(etype) > 0 and len(graph.edges[etype].data) > 0:
                logger.info(f"  {etype}边特征: {list(graph.edges[etype].data.keys())}") 