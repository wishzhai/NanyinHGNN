import logging
import numpy as np
from symusic import Score, Track, Note, Tempo
from core.instrument_rules.dongxiao import DongxiaoGenerator
from core.instrument_rules.erxian import ErxianGenerator
from core.decoder import NanyinDecoder
import random

# MIDI常量
TICKS_PER_BEAT = 480  # 每拍的tick数
DEFAULT_TEMPO = 120   # 默认速度

def seconds_to_ticks(seconds, tempo=DEFAULT_TEMPO):
    """将秒转换为MIDI ticks
    
    Args:
        seconds: 秒数
        tempo: 速度（BPM）
        
    Returns:
        int: tick数
    """
    beats = seconds * (tempo / 60.0)  # 将秒转换为拍数
    return int(beats * TICKS_PER_BEAT)

def create_note(time, duration, pitch, velocity):
    """创建一个音符对象
    
    Args:
        time: 开始时间（秒）
        duration: 持续时间（秒）
        pitch: 音高
        velocity: 力度
        
    Returns:
        Note: 音符对象
    """
    time_ticks = seconds_to_ticks(time)
    duration_ticks = seconds_to_ticks(duration)
    
    return Note(
        time=time_ticks,
        duration=duration_ticks,
        pitch=int(pitch),
        velocity=int(velocity)
    )

def create_instrument_tracks():
    """创建各个乐器的轨道"""
    tracks = {}
    
    # 琵琶轨道（主旋律）
    pipa_track = Track()
    pipa_track.program = 105
    pipa_track.name = "琵琶"
    tracks['pipa'] = pipa_track
    
    # 洞箫轨道
    dongxiao_track = Track()
    dongxiao_track.program = 77
    dongxiao_track.name = "洞箫"
    tracks['dongxiao'] = dongxiao_track
    
    # 二弦轨道
    erxian_track = Track()
    erxian_track.program = 107
    erxian_track.name = "二弦"
    tracks['erxian'] = erxian_track
    
    # 三弦轨道
    sanxian_track = Track()
    sanxian_track.program = 106
    sanxian_track.name = "三弦"
    tracks['sanxian'] = sanxian_track
    
    return tracks

def create_scores(tempo):
    """创建各个乐器的Score对象"""
    scores = {
        'pipa': Score(),
        'dongxiao': Score(),
        'erxian': Score(),
        'sanxian': Score()
    }
    
    # 为每个Score添加速度设置
    tempo_ticks = seconds_to_ticks(1.0, tempo)  # 1秒对应的tick数
    for score in scores.values():
        tempo_event = Tempo(0, tempo)
        score.tempos.append(tempo_event)
        
    return scores

def process_graph_notes(graph):
    """从图中提取音符数据"""
    if not ('note' in graph.ntypes and graph.num_nodes('note') > 0):
        return None, None, None, None, None, None, None
        
    # 提取音符数据
    pitches = graph.nodes['note'].data.get('pitch', None)
    positions = graph.nodes['note'].data.get('position', None)
    durations = graph.nodes['note'].data.get('duration', None)
    velocities = graph.nodes['note'].data.get('velocity', None)
    is_ornament = graph.nodes['note'].data.get('is_ornament', None)
    is_nianzhi = graph.nodes['note'].data.get('is_nianzhi', None)
    
    # 检查数据可用性
    if any(x is None for x in [pitches, positions, durations, velocities]):
        return None, None, None, None, None, None, None
        
    # 转换tensor到numpy
    pitches = pitches.cpu().numpy() if hasattr(pitches, 'cpu') else pitches
    positions = positions.cpu().numpy() if hasattr(positions, 'cpu') else positions
    durations = durations.cpu().numpy() if hasattr(durations, 'cpu') else durations
    velocities = velocities.cpu().numpy() if hasattr(velocities, 'cpu') else velocities
    if is_ornament is not None:
        is_ornament = is_ornament.cpu().numpy() if hasattr(is_ornament, 'cpu') else is_ornament
    if is_nianzhi is not None:
        is_nianzhi = is_nianzhi.cpu().numpy() if hasattr(is_nianzhi, 'cpu') else is_nianzhi
        
    return pitches, positions, durations, velocities, is_ornament, is_nianzhi, len(pitches)

def add_notes_to_track(track, notes):
    """将音符添加到轨道"""
    for note_data in notes:
        note = create_note(
            time=note_data['start'],
            duration=note_data['duration'],
            pitch=note_data['pitch'],
            velocity=note_data['velocity']
        )
        track.notes.append(note)

def create_nianzhi_notes(time, duration, pitch, velocity):
    """创建撚指音符组，模拟真实南音演奏中的撚指特点
    
    Args:
        time: 开始时间（秒）
        duration: 持续时间（秒）
        pitch: 音高
        velocity: 力度
        
    Returns:
        list: 撚指音符列表
    """
    notes = []
    
    # 确定撚指类型（快速、标准或慢速）
    nianzhi_types = ["fast", "standard", "slow"]
    nianzhi_type = random.choice(nianzhi_types)
    
    # 根据撚指类型设置参数
    if nianzhi_type == "fast":
        nianzhi_count = random.randint(4, 6)
        time_decay = 0.75
        velocity_drop_initial = random.randint(3, 6)
        velocity_drop_growth = 1.4
        duration_factor = 0.85
    elif nianzhi_type == "slow":
        nianzhi_count = random.randint(3, 4)
        time_decay = 0.9
        velocity_drop_initial = random.randint(2, 4)
        velocity_drop_growth = 1.1
        duration_factor = 0.92
    else:  # standard
        nianzhi_count = random.randint(3, 5)
        time_decay = 0.85
        velocity_drop_initial = random.randint(3, 5)
        velocity_drop_growth = 1.2
        duration_factor = 0.9
    
    # 计算基本持续时间（以秒为单位）
    base_duration = duration / (nianzhi_count * 1.5)
    if base_duration <= 0:
        base_duration = 0.1
    
    # 应用微小的随机变化
    base_duration *= random.uniform(0.95, 1.05)
    
    # 创建连续的短音符
    current_time = time
    time_factor = 1.0
    
    # 增加第一个音符的力度
    first_velocity = min(int(velocity * random.uniform(1.05, 1.1)), 127)
    
    for j in range(nianzhi_count):
        # 计算当前音符的参数
        current_velocity = first_velocity if j == 0 else max(
            30,  # 最小力度
            int(velocity - (velocity_drop_initial * (j ** velocity_drop_growth)))
        )
        
        # 计算当前音符的持续时间
        current_duration = base_duration * (duration_factor ** j)
        
        # 创建音符并添加到列表
        note_data = {
            'start': current_time,
            'duration': current_duration,
            'pitch': pitch,
            'velocity': current_velocity
        }
        notes.append(note_data)
        
        # 更新下一个音符的时间
        time_interval = base_duration * time_factor
        current_time += time_interval
        time_factor *= time_decay
    
    return notes

def graph_to_enhanced_scores(graph, tempo=120):
    """将图结构转换为多个MIDI文件，每个乐器一个文件"""
    logging.info("开始将图转换为增强版MIDI格式...")
    
    # 创建Score对象
    scores = create_scores(tempo)
    
    # 创建轨道
    tracks = create_instrument_tracks()
    
    try:
        # 处理图中的音符数据
        pitches, positions, durations, velocities, is_ornament, is_nianzhi, note_count = process_graph_notes(graph)
        if note_count is None:
            logging.warning("无法从图中提取有效的音符数据")
            return scores
            
        logging.info(f"发现 {note_count} 个音符节点")
        
        # 构造主旋律音符列表（用于琵琶和生成其他乐器的装饰音）
        main_notes = []
        for i in range(note_count):
            if is_ornament is None or not is_ornament[i]:
                main_notes.append({
                    'pitch': int(pitches[i]),
                    'start': float(positions[i]),
                    'duration': float(durations[i]),
                    'velocity': int(velocities[i]),
                    'is_nianzhi': bool(is_nianzhi[i]) if is_nianzhi is not None else False
                })
        
        if not main_notes:
            logging.warning("没有有效的主旋律音符")
            return scores
            
        # 获取基准音高
        base_pitch = int(np.median([note['pitch'] for note in main_notes]))
        
        # 添加琵琶音符（主旋律）
        for note_data in main_notes:
            if note_data['is_nianzhi']:
                # 处理撚指音符
                nianzhi_notes = create_nianzhi_notes(
                    time=note_data['start'],
                    duration=note_data['duration'],
                    pitch=note_data['pitch'],
                    velocity=note_data['velocity']
                )
                for note in nianzhi_notes:
                    tracks['pipa'].notes.append(note)
            else:
                # 处理普通音符
                note = create_note(
                    time=note_data['start'],
                    duration=note_data['duration'],
                    pitch=note_data['pitch'],
                    velocity=note_data['velocity']
                )
                tracks['pipa'].notes.append(note)
        
        # 生成洞箫装饰音
        dongxiao_gen = DongxiaoGenerator(base_pitch)
        dongxiao_notes = dongxiao_gen.generate(main_notes)
        add_notes_to_track(tracks['dongxiao'], dongxiao_notes)
        
        # 生成二弦装饰音（升八度）
        erxian_gen = ErxianGenerator(base_pitch)
        erxian_notes = erxian_gen.generate(main_notes)
        add_notes_to_track(tracks['erxian'], erxian_notes)
        
        # 生成三弦音符（与琵琶相同，但降八度）
        for note_data in main_notes:
            if note_data['is_nianzhi']:
                # 处理撚指音符
                nianzhi_notes = create_nianzhi_notes(
                    time=note_data['start'],
                    duration=note_data['duration'],
                    pitch=note_data['pitch'] - 12,  # 降八度
                    velocity=note_data['velocity']
                )
                for note in nianzhi_notes:
                    tracks['sanxian'].notes.append(note)
            else:
                # 处理普通音符
                note = create_note(
                    time=note_data['start'],
                    duration=note_data['duration'],
                    pitch=note_data['pitch'] - 12,  # 降八度
                    velocity=note_data['velocity']
                )
                tracks['sanxian'].notes.append(note)
        
        # 将轨道添加到对应的Score
        for instrument, track in tracks.items():
            scores[instrument].tracks.append(track)
            logging.info(f"{instrument}轨道: {len(track.notes)} 个音符")
        
        return scores
        
    except Exception as e:
        logging.error(f"创建MIDI时出错: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return scores 