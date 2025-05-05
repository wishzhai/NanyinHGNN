import torch
import random
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class SpecialNoteProcessor:
    def __init__(self, special_notes_midi: List[int], target_ratio: float = 0.3):
        """
        初始化特殊音处理器
        
        Args:
            special_notes_midi: 特殊音的MIDI音高列表
            target_ratio: 目标特殊音比例
        """
        self.special_notes_midi = special_notes_midi
        self.target_ratio = target_ratio
        self.midi_to_symbol = {
            50: "d", 52: "e", 53: "f", 55: "g", 57: "a", 59: "b", 60: "c",
            62: "d", 64: "e", 65: "f", 67: "g", 69: "a", 71: "b", 72: "c"
        }
        
    def process(self, graph) -> None:
        """
        处理图中的音符，添加特殊音装饰音
        
        Args:
            graph: DGLGraph对象
        """
        try:
            # 获取所有音符节点
            note_nodes = graph.nodes['note'].data
            if 'pitch' not in note_nodes:
                logger.warning("No pitch data found in note nodes")
                return
                
            # 获取音符特征
            pitches = note_nodes['pitch']
            positions = note_nodes['position']
            durations = note_nodes['duration']
            
            # 计算当前特殊音比例
            current_special_ratio = self._calculate_special_ratio(pitches)
            logger.info(f"Current special note ratio: {current_special_ratio:.4f}")
            
            # 如果当前比例已经达到目标，则不需要添加
            if current_special_ratio >= self.target_ratio:
                logger.info("Target special note ratio already achieved")
                return
                
            # 计算需要添加的特殊音数量
            total_notes = len(pitches)
            target_special_count = int(total_notes * self.target_ratio)
            current_special_count = int(total_notes * current_special_ratio)
            needed_special_count = target_special_count - current_special_count
            
            if needed_special_count <= 0:
                return
                
            # 选择最佳位置添加特殊音
            special_positions = self._select_optimal_positions(
                pitches, positions, needed_special_count
            )
            
            # 添加特殊音
            self._add_special_notes(graph, special_positions)
            
        except Exception as e:
            logger.error(f"Error processing special notes: {str(e)}")
            
    def _calculate_special_ratio(self, pitches: torch.Tensor) -> float:
        """计算当前特殊音比例"""
        special_count = sum(1 for p in pitches if p.item() in self.special_notes_midi)
        return special_count / len(pitches) if len(pitches) > 0 else 0
        
    def _select_optimal_positions(
        self, 
        pitches: torch.Tensor,
        positions: torch.Tensor,
        needed_count: int
    ) -> List[int]:
        """选择最佳位置添加特殊音"""
        # 计算每个位置的得分
        scores = []
        for i in range(len(pitches)):
            score = 0
            # 检查前后音符
            if i > 0:
                prev_pitch = pitches[i-1].item()
                if prev_pitch in self.special_notes_midi:
                    score += 1
            if i < len(pitches) - 1:
                next_pitch = pitches[i+1].item()
                if next_pitch in self.special_notes_midi:
                    score += 1
            # 检查音高间隔
            if i > 0:
                interval = abs(pitches[i].item() - pitches[i-1].item())
                if interval <= 2:  # 小音程
                    score += 1
            if i < len(pitches) - 1:
                interval = abs(pitches[i].item() - pitches[i+1].item())
                if interval <= 2:  # 小音程
                    score += 1
            scores.append(score)
            
        # 选择得分最高的位置
        positions = list(range(len(pitches)))
        positions.sort(key=lambda x: scores[x], reverse=True)
        return positions[:needed_count]
        
    def _add_special_notes(self, graph, positions: List[int]) -> None:
        """在选定位置添加特殊音"""
        try:
            # 获取音符节点数据
            note_nodes = graph.nodes['note'].data
            pitches = note_nodes['pitch']
            positions_data = note_nodes['position']
            durations = note_nodes['duration']
            
            # 创建特殊音节点
            special_nodes = []
            for pos in positions:
                # 获取主音符信息
                main_pitch = pitches[pos].item()
                main_pos = positions_data[pos].item()
                main_dur = durations[pos].item()
                
                # 选择最接近的特殊音
                special_pitch = self._find_nearest_special_note(main_pitch)
                if special_pitch is not None:
                    # 创建特殊音节点
                    special_node = {
                        'pitch': torch.tensor([special_pitch], dtype=torch.float32),
                        'position': torch.tensor([main_pos], dtype=torch.float32),
                        'duration': torch.tensor([main_dur * 0.5], dtype=torch.float32),
                        'velocity': torch.tensor([80], dtype=torch.float32),
                        'is_special': torch.tensor([1], dtype=torch.float32)
                    }
                    special_nodes.append(special_node)
                    
            # 添加特殊音节点到图中
            if special_nodes:
                # 合并所有特殊音节点
                combined_special = {
                    k: torch.cat([node[k] for node in special_nodes])
                    for k in special_nodes[0].keys()
                }
                
                # 更新图的特殊音节点
                for k, v in combined_special.items():
                    graph.nodes['special'].data[k] = v
                    
                logger.info(f"Added {len(special_nodes)} special notes")
                
        except Exception as e:
            logger.error(f"Error adding special notes: {str(e)}")
            
    def _find_nearest_special_note(self, pitch: float) -> int:
        """找到最接近的特殊音"""
        if not self.special_notes_midi:
            return None
            
        # 计算与所有特殊音的距离
        distances = [(abs(pitch - sp), sp) for sp in self.special_notes_midi]
        # 选择距离最小的特殊音
        return min(distances, key=lambda x: x[0])[1] 