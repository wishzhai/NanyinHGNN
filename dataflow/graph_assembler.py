import dgl
import torch
from typing import Dict

class NanyinGraphAssembler:
    """南音异构图组装器"""
    
    def __init__(self, config: Dict):
        self.node_capacity = config.get("max_nodes", 1024)
        self.edge_types = [
            ('note', 'temporal', 'note'),
            ('note', 'decorate', 'ornament'),
            ('tech', 'trigger', 'note')
        ]
    
    def assemble(self, raw_graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """组装图结构（build_graph的别名）
        Args:
            raw_graph: 原始图结构
        Returns:
            dgl.DGLHeteroGraph: 处理后的图结构
        """
        features = {
            'pitches': raw_graph.nodes['note'].data.get('pitch', []),
            'length': raw_graph.num_nodes('note'),
            'decorates': self._extract_decorates(raw_graph),
            'tech_triggers': self._extract_tech_triggers(raw_graph)
        }
        return self.build_graph(features)
    
    def _extract_decorates(self, graph: dgl.DGLHeteroGraph) -> list:
        """提取装饰音关系"""
        decorates = []
        if ('note', 'decorate', 'ornament') in graph.canonical_etypes:
            src, dst = graph.edges(etype=('note', 'decorate', 'ornament'))
            for s, d in zip(src.tolist(), dst.tolist()):
                decorates.append({'base_id': s, 'ornament_id': d})
        return decorates
    
    def _extract_tech_triggers(self, graph: dgl.DGLHeteroGraph) -> list:
        """提取技法触发关系"""
        triggers = []
        if ('tech', 'trigger', 'note') in graph.canonical_etypes:
            src, dst = graph.edges(etype=('tech', 'trigger', 'note'))
            for s, d in zip(src.tolist(), dst.tolist()):
                triggers.append({'tech_id': s, 'note_id': d})
        return triggers
    
    def build_graph(self, features: Dict) -> dgl.DGLHeteroGraph:
        """构建初始异构图"""
        graph = dgl.heterograph({
            etype: self._init_edges(features, etype) 
            for etype in self.edge_types
        })
        
        # 添加节点特征
        pitches = torch.as_tensor(features['pitches'], dtype=torch.long)
        graph.nodes['note'].data['pitch'] = pitches.clone().detach()
        graph.nodes['note'].data['position'] = torch.arange(len(features['pitches']), dtype=torch.long)
        return graph
    
    def _init_edges(self, features: Dict, etype: tuple) -> tuple:
        """初始化各类型边"""
        src_type, rel_type, dst_type = etype
        if rel_type == 'temporal':
            return self._temporal_edges(features['length'])
        elif rel_type == 'decorate':
            return self._decorative_edges(features['decorates'])
        elif rel_type == 'trigger':
            return self._tech_edges(features['tech_triggers'])
    
    def _temporal_edges(self, seq_len: int) -> tuple:
        """时序连接边"""
        return (torch.arange(seq_len-1), torch.arange(1, seq_len))
    
    def _decorative_edges(self, decorates: list) -> tuple:
        """装饰关系边"""
        src = [d['base_id'] for d in decorates]
        dst = [d['ornament_id'] for d in decorates]
        return (torch.tensor(src), torch.tensor(dst))
    
    def _tech_edges(self, triggers: list) -> tuple:
        """技法触发边"""
        tech_ids = torch.tensor([t['tech_id'] for t in triggers])
        note_ids = torch.tensor([t['note_id'] for t in triggers])
        return (tech_ids, note_ids)