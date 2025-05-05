import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import dgl

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NanyinModeConstraints:
    """南音调式约束模块
    
    提供南音调式规则的约束和评估功能，确保生成的内容符合传统南音调式。
    """
    
    # 南音调式定义
    NANYIN_MODES = {
        "WUKONG": {
            "name": "五空管",
            "upper_register": set([62, 64, 67, 69, 71, 74, 76, 79, 81, 83]),  # d1-b2
            "lower_register": set([50, 52, 55, 57, 60, 62, 64, 67, 69])  # d-a1
        },
        "SIKONG": {
            "name": "四空管",
            "scale": set([50, 53, 55, 57, 60, 62, 65, 67, 69, 72, 74, 76, 79, 81])  # d-a2
        },
        "WUKONG_SIYI": {
            "name": "五空四仪管",
            "scale": set([50, 52, 55, 57, 60, 62, 64, 67, 69, 72, 74, 76, 79, 81])  # d-a2
        },
        "BEISI": {
            "name": "倍四管",
            "scale": set([50, 52, 53, 57, 59, 62, 64, 66, 69, 71, 74, 76])  # d-e2
        }
    }
    
    # 南音音高名称映射
    NANYIN_PITCH_NAMES = {
        50: "d",    52: "e",    53: "f",    54: "#f",   55: "g",
        57: "a",    59: "b",    60: "c1",   61: "#c1",  62: "d1",
        64: "e1",   65: "f1",   66: "#f1",  67: "g1",   69: "a1",
        70: "bb1",  71: "b1",   72: "c2",   74: "d2",   76: "e2",
        79: "g2",   81: "a2",   83: "b2"
    }
    
    # 调式特定的重要音位
    MODE_IMPORTANT_POSITIONS = {
        "WUKONG": {
            "first": [50, 62, 74],  # d, d1, d2
            "fifth": [57, 69, 81],  # a, a1, a2
            "ending": [50, 57, 62]  # d, a, d1
        },
        "SIKONG": {
            "first": [53, 65, 76],  # f, f1, e2
            "fifth": [60, 72],      # c1, c2
            "ending": [53, 60, 65]  # f, c1, f1
        },
        "WUKONG_SIYI": {
            "first": [52, 64, 76],  # e, e1, e2
            "fifth": [59, 71],      # b, b1
            "ending": [52, 59, 64]  # e, b, e1
        },
        "BEISI": {
            "first": [53, 65],      # f, f1
            "fifth": [60, 72],      # c1, c2
            "ending": [53, 60, 66]  # f, c1, #f1
        }
    }
    
    def __init__(self, config: Dict):
        """初始化调式约束模块
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.default_mode = config.get('default_mode', 'WUKONG')
        self.compliance_threshold = config.get('compliance_threshold', 0.8)
        
        # 从配置中读取调式权重
        mode_weights = config.get('mode_weights', {})
        self.mode_weights = {
            'scale': mode_weights.get('scale', 1.0),
            'important_positions': mode_weights.get('important_positions', 2.0),
            'ending': mode_weights.get('ending', 3.0)
        }
        
        logger.info(f"初始化NanyinModeConstraints，默认调式: {self.default_mode}")
    
    def get_mode_scale(self, mode_name: str) -> set:
        """获取指定调式的音阶
        
        Args:
            mode_name: 调式名称
            
        Returns:
            set: 音阶集合
        """
        mode_info = self.NANYIN_MODES.get(mode_name)
        if not mode_info:
            logger.warning(f"未知调式: {mode_name}，使用五空管作为默认调式")
            mode_info = self.NANYIN_MODES["WUKONG"]
        
        if "scale" in mode_info:
            return mode_info["scale"]
        else:
            # 对于像WUKONG这样区分高低音区的调式，合并音区
            return mode_info.get("upper_register", set()).union(mode_info.get("lower_register", set()))
    
    def analyze_mode_compliance(self, pitches: List[int], positions: Optional[List[float]] = None) -> Dict[str, float]:
        """分析音符序列对各调式的符合度
        
        Args:
            pitches: 音高序列
            positions: 位置序列
            
        Returns:
            Dict[str, float]: 各调式的符合度得分
        """
        if not pitches:
            return {mode: 0.0 for mode in self.NANYIN_MODES}
        
        # 预处理音高数据，确保是数值而不是列表
        processed_pitches = []
        for p in pitches:
            if isinstance(p, list):
                p = p[0] if p else 0
            processed_pitches.append(int(p))
        
        # 预处理位置数据，确保是数值而不是列表
        processed_positions = None
        if positions:
            processed_positions = []
            for pos in positions:
                if isinstance(pos, list):
                    pos = pos[0] if pos else 0
                processed_positions.append(float(pos))
        
        # 标准化音高 (将所有音高折叠到同一个八度)
        normalized_pitches = [p % 12 for p in processed_pitches]
        
        # 计算各个调式的符合度
        mode_scores = {}
        
        for mode_name in self.NANYIN_MODES:
            # 获取该调式的音阶
            mode_scale = self.get_mode_scale(mode_name)
            
            # 确保音阶中的元素是数值而不是列表
            processed_scale = set()
            for p in mode_scale:
                if isinstance(p, list):
                    p = p[0] if p else 0
                processed_scale.add(int(p))
            
            # 将调式音阶也标准化到同一八度
            normalized_scale = set([p % 12 for p in processed_scale])
            
            # 计算在调式内的音符比例
            in_scale_count = sum(1 for p in processed_pitches if p in processed_scale)
            scale_compliance = in_scale_count / len(processed_pitches)
            
            # 计算重要位置的符合度
            important_pos_compliance = 0.0
            ending_compliance = 0.0
            
            if processed_positions and len(processed_positions) > 0:
                # 获取该调式的重要位置
                important_positions = self.MODE_IMPORTANT_POSITIONS.get(mode_name, {})
                
                # 检查是否有首音或五音在对应位置
                begin_pos = min(processed_positions)
                begin_index = processed_positions.index(begin_pos)
                begin_pitch = processed_pitches[begin_index]
                
                # 检查首音
                if begin_pitch in important_positions.get("first", []):
                    important_pos_compliance += 1.0
                
                # 检查末音
                end_pos = max(processed_positions)
                end_index = processed_positions.index(end_pos)
                end_pitch = processed_pitches[end_index]
                
                if end_pitch in important_positions.get("ending", []):
                    ending_compliance = 1.0
                
                # 检查强拍位置是否符合重要音位
                if len(processed_positions) > 1:
                    strong_beat_indices = []
                    for i, pos in enumerate(processed_positions):
                        # 假设每四个位置单位为一个小节，第一拍是强拍
                        if pos % 4 < 1:
                            strong_beat_indices.append(i)
                    
                    # 如果有强拍，检查这些位置的音高
                    if strong_beat_indices:
                        strong_beat_pitches = [processed_pitches[i] for i in strong_beat_indices]
                        important_pitches = set(important_positions.get("first", []) + important_positions.get("fifth", []))
                        important_count = sum(1 for p in strong_beat_pitches if p in important_pitches)
                        if strong_beat_pitches:
                            important_pos_compliance += important_count / len(strong_beat_pitches)
            
            # 计算加权得分
            weighted_score = (
                scale_compliance * self.mode_weights['scale'] +
                important_pos_compliance * self.mode_weights['important_positions'] +
                ending_compliance * self.mode_weights['ending']
            ) / sum(self.mode_weights.values())
            
            mode_scores[mode_name] = weighted_score
        
        return mode_scores
    
    def find_best_mode(self, pitches: List[int], positions: Optional[List[float]] = None) -> Tuple[str, float]:
        """找出最符合的调式
        
        Args:
            pitches: 音高序列
            positions: 位置序列
            
        Returns:
            Tuple[str, float]: 最符合的调式名称和得分
        """
        # 如果没有音高数据，返回默认调式和0分
        if not pitches:
            logger.warning("没有音高数据，使用默认调式")
            return (self.default_mode, 0.0)
            
        # 预处理音高数据，确保是数值而不是列表
        processed_pitches = []
        for p in pitches:
            if isinstance(p, list):
                p = p[0] if p else 0
            processed_pitches.append(int(p))
            
        mode_scores = self.analyze_mode_compliance(processed_pitches, positions)
        
        # 如果没有得分，返回默认调式和0分
        if not mode_scores:
            logger.warning("无法计算调式得分，使用默认调式")
            return (self.default_mode, 0.0)
            
        best_mode = max(mode_scores.items(), key=lambda x: x[1])
        return best_mode
    
    def correct_pitch_to_mode(self, pitch: int, mode_name: Optional[str] = None) -> int:
        """将给定音高纠正到指定调式中最近的音高
        
        Args:
            pitch: 原始音高
            mode_name: 调式名称（如果为None使用默认调式）
            
        Returns:
            int: 纠正后的音高
        """
        # 确保pitch是数值而不是列表
        if isinstance(pitch, list):
            pitch = pitch[0] if pitch else 0
            
        # 转换为整数
        pitch = int(pitch)
        
        if mode_name is None:
            mode_name = self.default_mode
        
        # 获取调式音阶
        mode_scale = self.get_mode_scale(mode_name)
        
        # 如果音高已经在调式中，无需修改
        if pitch in mode_scale:
            return pitch
        
        # 否则找出距离最近的调式内音高
        scale_list = sorted(list(mode_scale))
        distances = [abs(p - pitch) for p in scale_list]
        nearest_idx = distances.index(min(distances))
        
        corrected_pitch = scale_list[nearest_idx]
        logger.debug(f"将音高 {pitch} 纠正到 {mode_name} 调式中的 {corrected_pitch}")
        
        return corrected_pitch
    
    def apply_mode_constraints(self, graph: dgl.DGLHeteroGraph, mode_name: Optional[str] = None) -> dgl.DGLHeteroGraph:
        """对图应用调式约束
        
        Args:
            graph: 输入图
            mode_name: 调式名称（如果为None则自动检测）
            
        Returns:
            dgl.DGLHeteroGraph: 应用约束后的图
        """
        if 'note' not in graph.ntypes:
            logger.warning("图中没有note节点，无法应用调式约束")
            return graph
        
        # 获取音符特征
        if 'pitch' not in graph.nodes['note'].data:
            logger.warning("图中没有pitch特征，无法应用调式约束")
            return graph
        
        # 复制图，避免修改原图
        new_graph = graph.clone()
        
        # 获取音高和位置
        pitches = graph.nodes['note'].data['pitch'].cpu().numpy().tolist()
        positions = None
        if 'position' in graph.nodes['note'].data:
            positions = graph.nodes['note'].data['position'].cpu().numpy().tolist()
        
        # 如果没有指定调式，自动检测最佳调式
        if mode_name is None:
            mode_name, score = self.find_best_mode(pitches, positions)
            logger.info(f"检测到最佳调式: {mode_name}，得分: {score:.4f}")
            
            # 如果得分太低，使用默认调式
            if score < self.compliance_threshold:
                logger.info(f"调式得分低于阈值 {self.compliance_threshold}，使用默认调式 {self.default_mode}")
                mode_name = self.default_mode
        
        # 获取调式音阶
        mode_scale = self.get_mode_scale(mode_name)
        
        # 修正不在调式内的音符
        corrected_pitches = []
        for pitch in pitches:
            # 确保pitch是数值而不是列表
            if isinstance(pitch, list):
                pitch = pitch[0] if pitch else 0
            
            # 转换为整数
            pitch = int(pitch)
            
            if pitch not in mode_scale:
                corrected_pitch = self.correct_pitch_to_mode(pitch, mode_name)
                corrected_pitches.append(corrected_pitch)
            else:
                corrected_pitches.append(pitch)
        
        # 更新图的音高
        device = graph.device
        new_graph.nodes['note'].data['pitch'] = torch.tensor(corrected_pitches, device=device)
        
        # 添加调式信息作为图的属性
        new_graph.nodes['note'].data['mode'] = torch.full((len(pitches),), 
                                                         list(self.NANYIN_MODES.keys()).index(mode_name), 
                                                         device=device)
        
        return new_graph
    
    def calculate_mode_compliance(self, graph: dgl.DGLHeteroGraph) -> Dict:
        """计算图对各调式的符合度
        
        Args:
            graph: 输入图
            
        Returns:
            Dict: 包含调式分析结果的字典
        """
        if 'note' not in graph.ntypes:
            logger.warning("图中没有note节点，无法计算调式符合度")
            return {"error": "No note nodes in graph", "best_mode": self.default_mode}
        
        # 获取音符特征
        if 'pitch' not in graph.nodes['note'].data:
            logger.warning("图中没有pitch特征，无法计算调式符合度")
            return {"error": "No pitch feature in graph", "best_mode": self.default_mode}
        
        # 获取音高和位置
        pitches = graph.nodes['note'].data['pitch'].cpu().numpy().tolist()
        positions = None
        if 'position' in graph.nodes['note'].data:
            positions = graph.nodes['note'].data['position'].cpu().numpy().tolist()
        
        # 预处理音高数据，确保是数值而不是列表
        processed_pitches = []
        for p in pitches:
            if isinstance(p, list):
                p = p[0] if p else 0
            processed_pitches.append(int(p))
        
        # 分析各调式符合度
        mode_scores = self.analyze_mode_compliance(processed_pitches, positions)
        
        # 找出最符合的调式
        try:
            best_mode, best_score = self.find_best_mode(processed_pitches, positions)
        except Exception as e:
            logger.error(f"查找最佳调式时出错: {e}")
            best_mode = self.default_mode
            best_score = 0.0
        
        # 计算调式内音符比例
        mode_scale = self.get_mode_scale(best_mode)
        
        # 确保音阶中的元素是数值而不是列表
        processed_scale = set()
        for p in mode_scale:
            if isinstance(p, list):
                p = p[0] if p else 0
            processed_scale.add(int(p))
            
        in_scale_count = sum(1 for p in processed_pitches if p in processed_scale)
        in_scale_ratio = in_scale_count / len(processed_pitches) if processed_pitches else 0
        
        result = {
            "best_mode": best_mode,
            "best_mode_name": self.NANYIN_MODES[best_mode]["name"],
            "best_score": best_score,
            "all_mode_scores": mode_scores,
            "in_scale_ratio": in_scale_ratio,
            "total_notes": len(pitches),
            "in_scale_notes": in_scale_count
        }
        
        return result 