import numpy as np
from .dongxiao import DongxiaoGenerator

class ErxianGenerator(DongxiaoGenerator):
    """二弦声部生成规则（继承洞箫逻辑并调整）"""
    
    def __init__(self, base_pitch: int):
        super().__init__(base_pitch)
        self.ornament_shift = +12  # 同洞箫八度
        self.density = 0.7  # 装饰音密度系数
        
    def generate(self, main_notes: list) -> list:
        base_notes = super().generate(main_notes)
        return self._apply_density_filter(base_notes)
    
    def _apply_density_filter(self, notes: list) -> list:
        """应用密度衰减"""
        return [note for i, note in enumerate(notes) 
                if i % 3 != 0 or np.random.rand() < self.density]

    def _create_ornament(self, main_note: dict) -> dict:
        """创建装饰音
        
        Args:
            main_note: 主音符数据，包含pitch, start, duration, velocity
            
        Returns:
            dict: 装饰音数据，包含pitch, start, duration, velocity
        """
        # 获取主音符数据
        pitch = int(main_note['pitch'])
        start = float(main_note['start'])
        duration = float(main_note['duration'])
        velocity = int(main_note['velocity'])
        
        # 计算装饰音参数
        ornament_start = start + duration * 0.1  # 装饰音在主音符开始后10%处
        ornament_duration = duration * 0.3  # 装饰音时值为主音符的30%
        
        # 获取装饰音音高
        ornament_pitch = self._get_ornament_pitch(pitch)
        
        # 获取装饰音力度
        ornament_velocity = self._get_velocity(velocity)
        
        # 返回装饰音数据
        return {
            'pitch': ornament_pitch,
            'start': ornament_start,
            'duration': ornament_duration,
            'velocity': ornament_velocity
        }