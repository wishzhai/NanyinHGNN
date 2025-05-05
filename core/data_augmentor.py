import torch
import random
import numpy as np
from typing import Dict, Any
import dgl

class NanyinDataAugmentor:
    """南音数据增强器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 南音调式音高集合
        self.mode_pitches = {
            'wukong': {
                'upper': [74, 76, 79, 81, 83, 86, 88, 91, 93, 95],  # d1-b2 (G-based)
                'lower': [62, 64, 67, 69, 72, 74, 76, 79, 81]       # d-a1 (C-based)
            },
            'sikong': [53, 55, 57, 60, 62, 65, 67, 69, 72],        # F-based
            'wukong_siyi': [60, 62, 64, 67, 69, 72, 74, 76, 79],   # C-based
            'beisi': [62, 64, 67, 69, 71, 74, 76, 79, 81]          # D-based
        }
        # 特殊音级
        self.special_notes = {'#f', '#c1', '#f1'}
        # 装饰音音程规则
        self.ornament_intervals = [-2, 2]  # 上下大二度
        
    def augment(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """应用数据增强
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLGraph: 增强后的图
        """
        # 1. 调式内移调（保持在同一调式内）
        if random.random() < 0.3:
            graph = self._mode_preserving_transpose(graph)
            
        # 2. 装饰音变体（在保持规则的前提下）
        if random.random() < 0.2:
            graph = self._ornament_variation(graph)
            
        # 3. 节奏微扰（保持基本节奏型）
        if random.random() < 0.15:
            graph = self._rhythm_perturbation(graph)
            
        # 4. 力度变化（符合南音表现习惯）
        if random.random() < 0.25:
            graph = self._dynamics_variation(graph)
            
        return graph
        
    def _mode_preserving_transpose(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """在调式内进行移调
        
        保持在调式音阶内，同时保持特殊音级的关系
        特别处理悟空调的双音域结构
        """
        if 'mode' not in graph.nodes['note'].data:
            return graph
            
        current_mode = graph.nodes['note'].data['mode'][0]
        mode_name = current_mode.lower()
        
        # 获取音高
        pitches = graph.nodes['note'].data['pitch']
        new_pitches = pitches.clone()
        
        if mode_name == 'wukong':
            # 悟空调特殊处理：分别处理上下音域
            for i, pitch in enumerate(pitches):
                pitch_val = pitch.item()
                if pitch_val >= 74:  # d1及以上使用上音域（G宫系统）
                    valid_pitches = self.mode_pitches['wukong']['upper']
                    shift = random.choice([-7, -5, -2, 2, 5, 7])  # 允许更大的移动范围
                else:  # 下音域（C宫系统）
                    valid_pitches = self.mode_pitches['wukong']['lower']
                    shift = random.choice([-5, -2, 2, 5])
                
                if str(pitch_val) in self.special_notes:
                    continue  # 保持特殊音级不变
                
                new_pitch = pitch_val + shift
                # 找到最近的有效音高
                nearest_pitch = min(valid_pitches, key=lambda x: abs(x - new_pitch))
                new_pitches[i] = nearest_pitch
        else:
            # 其他调式的常规处理
            valid_pitches = self.mode_pitches.get(mode_name, self.mode_pitches['sikong'])
            shift = random.choice([-5, -2, 2, 5])  # 纯四度、大二度的移动
            
            for i, pitch in enumerate(pitches):
                if str(pitch.item()) in self.special_notes:
                    continue  # 保持特殊音级不变
                
                new_pitch = pitch.item() + shift
                # 找到最近的调式内音高
                nearest_pitch = min(valid_pitches, key=lambda x: abs(x - new_pitch))
                new_pitches[i] = nearest_pitch
        
        graph.nodes['note'].data['pitch'] = new_pitches
        return graph
        
    def _ornament_variation(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """装饰音变体
        
        在保持装饰音规则的前提下进行变化，考虑调式特点
        """
        if 'ornament' not in graph.ntypes or graph.num_nodes('ornament') == 0:
            return graph
            
        current_mode = graph.nodes['note'].data['mode'][0].lower()
        ornament_nodes = graph.nodes['ornament']
        
        for i in range(graph.num_nodes('ornament')):
            if random.random() < 0.3:
                # 获取主音节点
                main_note = graph.predecessors(i, etype='decorate')[0]
                main_pitch = graph.nodes['note'].data['pitch'][main_note].item()
                
                # 根据调式选择合适的装饰音
                if current_mode == 'wukong':
                    # 悟空调需要考虑双音域
                    if main_pitch >= 74:  # 上音域
                        valid_pitches = self.mode_pitches['wukong']['upper']
                    else:  # 下音域
                        valid_pitches = self.mode_pitches['wukong']['lower']
                else:
                    valid_pitches = self.mode_pitches.get(current_mode, self.mode_pitches['sikong'])
                
                # 在允许的装饰音音程内变化，但确保在调式内
                possible_intervals = []
                for interval in self.ornament_intervals:
                    target_pitch = main_pitch + interval
                    if any(abs(p - target_pitch) <= 2 for p in valid_pitches):
                        possible_intervals.append(interval)
                
                if possible_intervals:
                    new_interval = random.choice(possible_intervals)
                    target_pitch = main_pitch + new_interval
                    # 找到最近的调式内音高
                    nearest_pitch = min(valid_pitches, key=lambda x: abs(x - target_pitch))
                    ornament_nodes.data['pitch'][i] = nearest_pitch
                
        return graph
        
    def _rhythm_perturbation(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """节奏微扰
        
        在保持基本节奏型的前提下添加细微时值变化
        """
        if 'duration' not in graph.nodes['note'].data:
            return graph
            
        durations = graph.nodes['note'].data['duration']
        # 添加小幅度的随机扰动（最大±5%）
        noise = torch.randn_like(durations) * 0.05
        new_durations = durations * (1 + noise)
        # 确保时值保持正数
        new_durations = torch.clamp(new_durations, min=0.1)
        
        graph.nodes['note'].data['duration'] = new_durations
        return graph
        
    def _dynamics_variation(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """力度变化
        
        符合南音表现习惯的力度变化
        """
        if 'velocity' not in graph.nodes['note'].data:
            return graph
            
        velocities = graph.nodes['note'].data['velocity']
        # 生成平滑的力度变化曲线
        positions = graph.nodes['note'].data['position']
        curve = torch.sin(positions * 2 * np.pi) * 0.1  # 生成波动范围在±10%的正弦曲线
        new_velocities = velocities * (1 + curve)
        # 确保力度在有效范围内
        new_velocities = torch.clamp(new_velocities, min=0.1, max=1.0)
        
        graph.nodes['note'].data['velocity'] = new_velocities
        return graph 