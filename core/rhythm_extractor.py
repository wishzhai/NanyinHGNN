import torch
import numpy as np
import dgl
import logging
from collections import defaultdict, Counter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RhythmExtractor:
    """节奏结构提取器，从图中提取音乐节奏特征"""
    
    def __init__(self, config=None):
        """初始化
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        logger.info("初始化RhythmExtractor")
    
    def extract_from_batch(self, batch):
        """从批次数据中提取节奏结构
        
        Args:
            batch: 输入批次数据，可以是单个图或图列表
            
        Returns:
            list: 节奏结构列表
        """
        if isinstance(batch, list):
            return [self.extract(g) for g in batch]
        else:
            return [self.extract(batch)]
    
    def extract(self, graph):
        """从单个图中提取节奏结构
        
        Args:
            graph: 输入图
            
        Returns:
            dict: 节奏结构字典
        """
        try:
            # 检查是否为DGLHeteroGraph
            if not isinstance(graph, dgl.DGLHeteroGraph):
                logger.warning("输入不是DGLHeteroGraph")
                return self._empty_rhythm_struct()
            
            # 检查图中是否有'note'节点
            if 'note' not in graph.ntypes:
                logger.warning("图中没有'note'节点类型")
                return self._empty_rhythm_struct()
            
            # 检查'note'节点数量
            num_nodes = graph.num_nodes('note')
            if num_nodes == 0:
                logger.warning("'note'节点数量为0")
                return self._empty_rhythm_struct()
            
            # 检查必要特征
            if 'position' not in graph.nodes['note'].data:
                logger.warning("缺少'position'特征")
                return self._empty_rhythm_struct()
            
            # 获取位置信息
            positions = graph.nodes['note'].data['position']
            if positions.dim() > 1:
                positions = positions.squeeze()
            positions = positions.long().cpu().numpy()
            
            # 获取其他可选特征
            pitches = None
            durations = None
            velocities = None
            
            if 'pitch' in graph.nodes['note'].data:
                pitches = graph.nodes['note'].data['pitch']
                if pitches.dim() > 1:
                    pitches = pitches.squeeze()
                pitches = pitches.cpu().numpy()
            
            if 'duration' in graph.nodes['note'].data:
                durations = graph.nodes['note'].data['duration']
                if durations.dim() > 1:
                    durations = durations.squeeze()
                durations = durations.cpu().numpy()
            
            if 'velocity' in graph.nodes['note'].data:
                velocities = graph.nodes['note'].data['velocity']
                if velocities.dim() > 1:
                    velocities = velocities.squeeze()
                velocities = velocities.cpu().numpy()
            
            # 构建节奏结构
            rhythm_struct = self._build_rhythm_struct(positions, pitches, durations, velocities)
            
            return rhythm_struct
            
        except Exception as e:
            logger.error(f"提取节奏结构时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._empty_rhythm_struct()
    
    def _build_rhythm_struct(self, positions, pitches=None, durations=None, velocities=None):
        """构建节奏结构
        
        Args:
            positions: 位置列表或数组
            pitches: 音高列表或数组（可选）
            durations: 持续时间列表或数组（可选）
            velocities: 力度列表或数组（可选）
            
        Returns:
            dict: 节奏结构字典
        """
        # 确保输入是列表
        positions = positions if isinstance(positions, list) else positions.tolist() if hasattr(positions, 'tolist') else list(positions)
        
        # 对位置进行排序
        sorted_indices = sorted(range(len(positions)), key=lambda k: positions[k])
        sorted_positions = [positions[i] for i in sorted_indices]
        
        # 计算基本统计信息
        note_count = len(positions)
        unique_positions = len(set(sorted_positions))
        
        # 计算时间间隔
        intervals = []
        if len(sorted_positions) >= 2:
            intervals = [sorted_positions[i+1] - sorted_positions[i] for i in range(len(sorted_positions)-1)]
        
        mean_interval = sum(intervals) / len(intervals) if intervals else 0.0
        
        # 构建基本节奏结构
        rhythm_struct = {
            'positions': sorted_positions,
            'note_count': note_count,
            'unique_positions': unique_positions,
            'mean_interval': mean_interval,
            'densities': [1] * len(sorted_positions),  # 简化版本的密度计算
            'pitch_stats': {},
            'duration_stats': {},
            'velocity_stats': {}
        }
        
        # 添加音高统计
        if pitches is not None:
            pitches = pitches if isinstance(pitches, list) else pitches.tolist() if hasattr(pitches, 'tolist') else list(pitches)
            sorted_pitches = [pitches[i] for i in sorted_indices]
            rhythm_struct['pitches'] = sorted_pitches
            rhythm_struct['pitch_stats'] = {
                'mean': sum(sorted_pitches) / len(sorted_pitches),
                'min': min(sorted_pitches),
                'max': max(sorted_pitches)
            }
            
        # 添加持续时间统计
        if durations is not None:
            durations = durations if isinstance(durations, list) else durations.tolist() if hasattr(durations, 'tolist') else list(durations)
            sorted_durations = [durations[i] for i in sorted_indices]
            rhythm_struct['durations'] = sorted_durations
            rhythm_struct['duration_stats'] = {
                'mean': sum(sorted_durations) / len(sorted_durations),
                'min': min(sorted_durations),
                'max': max(sorted_durations)
            }
            
        # 添加力度统计
        if velocities is not None:
            velocities = velocities if isinstance(velocities, list) else velocities.tolist() if hasattr(velocities, 'tolist') else list(velocities)
            sorted_velocities = [velocities[i] for i in sorted_indices]
            rhythm_struct['velocities'] = sorted_velocities
            rhythm_struct['velocity_stats'] = {
                'mean': sum(sorted_velocities) / len(sorted_velocities),
                'min': min(sorted_velocities),
                'max': max(sorted_velocities)
            }
        
        return rhythm_struct
    
    def _empty_rhythm_struct(self):
        """创建空的节奏结构
        
        Returns:
            dict: 空的节奏结构字典
        """
        return {
            'positions': [],
            'note_count': 0,
            'unique_positions': 0,
            'mean_interval': 0.0,
            'densities': [],
            'pitch_stats': {},
            'duration_stats': {},
            'velocity_stats': {}
        }
    
    def calculate_rhythm_loss(self, rhythm1, rhythm2):
        """计算两个节奏结构之间的差异
        
        Args:
            rhythm1: 第一个节奏结构
            rhythm2: 第二个节奏结构
            
        Returns:
            float: 节奏差异分数（0-1之间，0表示完全相同）
        """
        if not rhythm1 or not rhythm2:
            return 1.0  # 最大差异
            
        if rhythm1['note_count'] == 0 or rhythm2['note_count'] == 0:
            return 1.0
            
        # 位置分布差异
        pos1 = set(rhythm1['positions'])
        pos2 = set(rhythm2['positions'])
        
        if not pos1 or not pos2:
            return 1.0
            
        pos_overlap = len(pos1.intersection(pos2))
        pos_union = len(pos1.union(pos2))
        pos_similarity = pos_overlap / max(1, pos_union)
        
        # 基本统计差异
        interval_diff = abs(rhythm1['mean_interval'] - rhythm2['mean_interval']) / (max(rhythm1['mean_interval'], rhythm2['mean_interval']) + 1e-6)
        density_diff = abs(rhythm1['densities'][-1] - rhythm2['densities'][-1]) / 1.0  # 假设densities[-1]是最后一个元素
        
        # 音高统计差异（如果可用）
        pitch_diff = 0.0
        if rhythm1['pitch_stats'] and rhythm2['pitch_stats']:
            pitch_diff = abs(rhythm1['pitch_stats'].get('mean', 0) - rhythm2['pitch_stats'].get('mean', 0)) / 12.0  # 标准化到八度
            pitch_diff = min(1.0, pitch_diff)  # 限制在0-1范围内
        
        # 加权总差异
        total_diff = (
            0.5 * (1 - pos_similarity) +  # 位置差异权重更高
            0.2 * interval_diff +
            0.2 * density_diff +
            0.1 * pitch_diff
        )
        
        return min(1.0, max(0.0, total_diff))  # 确保结果在0-1之间 