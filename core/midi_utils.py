import logging
import numpy as np
from symusic import Score, Track, Note, Tempo
from core.instrument_rules.dongxiao import DongxiaoGenerator
from core.instrument_rules.sanxian import SanxianGenerator

def create_note(time, duration, pitch, velocity):
    """创建一个音符对象
    
    Args:
        time: 开始时间
        duration: 持续时间
        pitch: 音高
        velocity: 力度
        
    Returns:
        Note: 音符对象
    """
    return Note(
        time=float(time),
        duration=float(duration),
        pitch=int(pitch),
        velocity=int(velocity)
    )

def add_melody_notes(track, pitches, positions, durations, velocities, is_ornament=None):
    """添加主旋律音符到轨道"""
    try:
        for i in range(len(pitches)):
            if is_ornament is not None and is_ornament[i]:
                continue
                
            note = create_note(
                time=positions[i],
                duration=durations[i],
                pitch=pitches[i],
                velocity=velocities[i]
            )
            track.notes.append(note)
            
    except Exception as e:
        logging.error(f"添加主旋律音符时出错: {str(e)}")

def create_instrument_tracks():
    """创建各个乐器的轨道"""
    tracks = {}
    
    # 主旋律轨道（钢琴）
    main_track = Track()
    main_track.program = 0
    main_track.name = "主旋律"
    tracks['main'] = main_track
    
    # 洞箫轨道
    dongxiao_track = Track()
    dongxiao_track.program = 77
    dongxiao_track.name = "洞箫"
    tracks['dongxiao'] = dongxiao_track
    
    # 三弦轨道
    sanxian_track = Track()
    sanxian_track.program = 106
    sanxian_track.name = "三弦"
    tracks['sanxian'] = sanxian_track
    
    # 二弦轨道
    erxian_track = Track()
    erxian_track.program = 107
    erxian_track.name = "二弦"
    tracks['erxian'] = erxian_track
    
    # 琵琶轨道
    pipa_track = Track()
    pipa_track.program = 105
    pipa_track.name = "琵琶"
    tracks['pipa'] = pipa_track
    
    return tracks

def create_scores(tempo):
    """创建各个乐器的Score对象"""
    scores = {
        'main': Score(),
        'dongxiao': Score(),
        'sanxian': Score(),
        'erxian': Score(),
        'pipa': Score()
    }
    
    # 为每个Score添加速度设置
    for score in scores.values():
        tempo_event = Tempo(0, tempo)
        score.tempos.append(tempo_event)
        
    return scores

def process_graph_notes(graph):
    """从图中提取音符数据"""
    if not ('note' in graph.ntypes and graph.num_nodes('note') > 0):
        return None, None, None, None, None, None
        
    # 提取音符数据
    pitches = graph.nodes['note'].data.get('pitch', None)
    positions = graph.nodes['note'].data.get('position', None)
    durations = graph.nodes['note'].data.get('duration', None)
    velocities = graph.nodes['note'].data.get('velocity', None)
    is_ornament = graph.nodes['note'].data.get('is_ornament', None)
    
    # 检查数据可用性
    if any(x is None for x in [pitches, positions, durations, velocities]):
        return None, None, None, None, None, None
        
    # 转换tensor到numpy
    pitches = pitches.cpu().numpy() if hasattr(pitches, 'cpu') else pitches
    positions = positions.cpu().numpy() if hasattr(positions, 'cpu') else positions
    durations = durations.cpu().numpy() if hasattr(durations, 'cpu') else durations
    velocities = velocities.cpu().numpy() if hasattr(velocities, 'cpu') else velocities
    if is_ornament is not None:
        is_ornament = is_ornament.cpu().numpy() if hasattr(is_ornament, 'cpu') else is_ornament
        
    return pitches, positions, durations, velocities, is_ornament, len(pitches)

def create_instrument_notes(main_notes, base_pitch):
    """为各个乐器生成音符"""
    instrument_notes = {
        'dongxiao': [],
        'sanxian': [],
        'erxian': [],
        'pipa': []
    }
    
    if not main_notes:
        return instrument_notes
        
    # 洞箫装饰音
    dongxiao_gen = DongxiaoGenerator(base_pitch)
    instrument_notes['dongxiao'] = dongxiao_gen.generate(main_notes)
    
    # 三弦装饰音
    sanxian_gen = SanxianGenerator(base_pitch)
    instrument_notes['sanxian'] = sanxian_gen.generate(main_notes)
    
    # 二弦（使用主旋律的变体，降八度）
    instrument_notes['erxian'] = [
        {
            'time': note['time'],
            'duration': note['duration'],
            'pitch': note['pitch'] - 12,
            'velocity': note['velocity']
        }
        for note in main_notes
    ]
    
    # 琵琶（使用主旋律）
    instrument_notes['pipa'] = [
        {
            'time': note['time'],
            'duration': note['duration'],
            'pitch': note['pitch'],
            'velocity': note['velocity']
        }
        for note in main_notes
    ]
    
    return instrument_notes

def graph_to_enhanced_scores(graph, tempo=120):
    """将图结构转换为多个MIDI文件，每个乐器一个文件"""
    logging.info("开始将图转换为增强版MIDI格式...")
    
    # 创建Score对象
    scores = create_scores(tempo)
    
    # 创建轨道
    tracks = create_instrument_tracks()
    
    try:
        # 处理图中的音符数据
        pitches, positions, durations, velocities, is_ornament, note_count = process_graph_notes(graph)
        if note_count is None:
            logging.warning("无法从图中提取有效的音符数据")
            return scores
            
        logging.info(f"发现 {note_count} 个音符节点")
        
        # 添加主旋律音符
        add_melody_notes(tracks['main'], pitches, positions, durations, velocities, is_ornament)
        
        # 构造主旋律音符列表
        main_notes = []
        for i in range(note_count):
            if is_ornament is None or not is_ornament[i]:
                main_notes.append({
                    'pitch': int(pitches[i]),
                    'time': float(positions[i]),
                    'duration': float(durations[i]),
                    'velocity': int(velocities[i])
                })
        
        # 获取基准音高
        base_pitch = int(np.median(pitches)) if len(pitches) > 0 else 60
        
        # 生成各乐器的音符
        instrument_notes = create_instrument_notes(main_notes, base_pitch)
        
        # 将音符添加到各个轨道
        for instrument, notes in instrument_notes.items():
            for note_data in notes:
                note = create_note(
                    time=note_data['time'],
                    duration=note_data['duration'],
                    pitch=note_data['pitch'],
                    velocity=note_data.get('velocity', 64)
                )
                tracks[instrument].notes.append(note)
        
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