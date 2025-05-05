import numpy as np
from typing import Dict

class LiaopaiEngine:
    """南音撩拍节奏生成引擎"""
    
    def __init__(self, template_config: Dict):
        """
        Args:
            template_config: 撩拍模板配置，格式示例：
                {
                    'sections': {
                        'seven_liao': {'pattern': [1,0,0,0,0,0,0,0], 'tempo_range': [50,55], ...},
                        ...
                    }
                }
        """
        self.templates = template_config['sections']
        self.current_position = 0  # 当前生成位置（单位：拍）
    
    def generate_full_structure(self, total_length: float, min_section_length: float = 1.0, 
                             tempo_range: tuple = (60, 120), beat_resolution: float = 0.25) -> Dict:
        """生成完整乐曲结构
        Args:
            total_length: 总拍数
            min_section_length: 最小段落长度（拍）
            tempo_range: 速度范围元组 (最小值, 最大值)
            beat_resolution: 拍子分辨率
        Returns:
            {
                'sections': [
                    {'type': 'seven_liao', 'start':0, 'end':200, 'tempo_curve': [...]},
                    ...
                ],
                'transitions': [
                    {'from': 'seven_liao', 'to': 'three_liao', 'position': 200, 'type': 'gradual'},
                    ...
                ]
            }
        """
        structure = {'sections': [], 'transitions': []}
        remaining_length = float(total_length)
        
        # 按顺序生成四个段落
        section_types = ['seven_liao', 'three_liao', 'one_two_pai', 'diepai']
        for sect_type in section_types:
            sect_config = self.templates[sect_type]
            # 确保段落长度不小于最小长度
            sect_length = max(float(total_length * sect_config['duration_ratio']), min_section_length)
            
            # 生成节拍序列，考虑拍子分辨率
            num_beats = int(sect_length / beat_resolution)
            beat_seq = self._generate_beat_sequence(sect_config['pattern'], num_beats)
            
            # 生成速度曲线，使用传入的速度范围
            min_tempo, max_tempo = tempo_range
            if 'tempo_range' in sect_config:
                min_tempo = sect_config['tempo_range'][0]
                max_tempo = sect_config['tempo_range'][1]
            tempo_curve = np.linspace(min_tempo, max_tempo, num_beats)
            
            # 记录段落信息
            structure['sections'].append({
                'type': sect_type,
                'start': self.current_position,
                'end': self.current_position + sect_length,
                'tempo_curve': tempo_curve.tolist(),
                'beat_sequence': beat_seq
            })
            
            # 处理段落过渡
            if sect_type != 'diepai':
                next_type = section_types[section_types.index(sect_type) + 1]
                transition_type = sect_config.get('transition', {}).get('type', 'gradual')
                
                structure['transitions'].append({
                    'from': sect_type,
                    'to': next_type,
                    'position': self.current_position + sect_length,
                    'type': transition_type
                })
            
            self.current_position += sect_length
            remaining_length -= sect_length
        
        # 处理叠拍结尾撚指
        self._add_final_nianzhi(structure)
        return structure
    
    def _generate_beat_sequence(self, pattern: list, length: int) -> list:
        """扩展基础节拍模式到指定长度"""
        repeated = (pattern * (length // len(pattern) + 1))[:length]
        return repeated
    
    def _add_final_nianzhi(self, structure: Dict):
        """在叠拍结尾添加撚指标记"""
        die_section = structure['sections'][-1]
        ending_config = self.templates['diepai']['ending']['nianzhi_trigger']
        
        if ending_config['position'] == 'last_3_beats':
            nianzhi_start = die_section['end'] - ending_config['duration']
            die_section['special_marks'] = {
                'nianzhi': {
                    'start': nianzhi_start,
                    'duration': ending_config['duration'],
                    'action': 'add_ornament_ending'
                }
            }