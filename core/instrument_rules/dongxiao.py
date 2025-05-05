import numpy as np

class DongxiaoGenerator:
    """洞箫声部生成规则"""
    
    def __init__(self, base_pitch: int):
        """
        Args:
            base_pitch: 主音音高（MIDI音高编号）
        """
        self.base_pitch = base_pitch
        self.ornament_shift = +12  # 升八度
        self.breath_pattern = [0.03, -0.02, 0.01]  # 呼吸扰动模式（秒）
        self.velocity_pattern = [80, 75, 85]  # 力度变化模式
        
    def generate(self, main_notes: list) -> list:
        """生成洞箫声部
        Args:
            main_notes: 主音序列，每个元素包含['pitch','start','duration','velocity']
        Returns:
            [{'pitch': int, 'start': float, 'duration': float, 'velocity': int}, ...]
        """
        dongxiao_notes = []
        breath_counter = 0
        
        for note in main_notes:
            # 跳过撚指触发的休止
            if note.get('tech') == 'nianzhi':
                continue
                
            # 基础装饰音生成
            ornament = self._create_ornament(note)
            
            # 添加呼吸扰动
            ornament['start'] += self._get_breath_shift(breath_counter)
            breath_counter += 1
            
            # 添加力度变化
            ornament['velocity'] = self._get_velocity(breath_counter)
            
            dongxiao_notes.append(ornament)
        
        return dongxiao_notes
    
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
    
    def _get_decorate_shift(self) -> int:
        """装饰音偏移（大二度+特色音优先）"""
        # 特色音强制优先逻辑
        current_pitch = self.base_pitch + self.ornament_shift
        if current_pitch % 12 in {6, 1, 6}:  # #F(6)/#C1(1)/#F1(6)
            return 0  # 已经是特色音，保持原位
        return 2  # 上方大二度
    
    def _get_breath_shift(self, count: int) -> float:
        """获取呼吸扰动时间偏移"""
        return float(self.breath_pattern[count % len(self.breath_pattern)])
        
    def _get_velocity(self, count: int) -> int:
        """获取力度值"""
        return int(self.velocity_pattern[count % len(self.velocity_pattern)])

    def _get_ornament_pitch(self, pitch: int) -> int:
        """获取装饰音音高
        
        Args:
            pitch: 主音音高
            
        Returns:
            int: 装饰音音高
        """
        # 计算装饰音音高（主音 + 八度 + 装饰音偏移）
        return pitch + self.ornament_shift + self._get_decorate_shift()