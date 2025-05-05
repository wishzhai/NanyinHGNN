import numpy as np
import dgl
import torch
from symusic import Score, Track, Note

class NanyinDecoder:
    """将图结构解码为多轨MIDI"""
    
    def __init__(self, tempo: int = 120):
        self.tempo = tempo
        self.ticks_per_beat = 480  # MIDI默认分辨率
        
    def _tensor_to_int(self, tensor_value) -> int:
        """将张量值安全地转换为Python整数
        
        Args:
            tensor_value: 可能是张量的值
            
        Returns:
            int: Python整数
        """
        try:
            if isinstance(tensor_value, torch.Tensor):
                return int(tensor_value.item())
            elif isinstance(tensor_value, (int, float)):
                return int(tensor_value)
            else:
                return int(float(tensor_value))
        except Exception as e:
            print(f"转换为整数时出错: {str(e)}")
            return 0
        
    def _tensor_to_float(self, tensor_value) -> float:
        """将张量值安全地转换为Python浮点数
        
        Args:
            tensor_value: 可能是张量的值
            
        Returns:
            float: Python浮点数
        """
        try:
            if isinstance(tensor_value, torch.Tensor):
                return float(tensor_value.item())
            elif isinstance(tensor_value, (int, float)):
                return float(tensor_value)
            else:
                return float(tensor_value)
        except Exception as e:
            print(f"转换为浮点数时出错: {str(e)}")
            return 0.0

    def _create_note(self, time_value, duration_value, pitch_value, velocity_value) -> Note:
        """安全地创建Note对象
        
        Args:
            time_value: 时间值
            duration_value: 持续时间值
            pitch_value: 音高值
            velocity_value: 力度值
            
        Returns:
            Note: 创建的音符对象
        """
        try:
            # 先转换为基本Python类型
            time = self._tensor_to_float(time_value)
            duration = self._tensor_to_float(duration_value)
            pitch = self._tensor_to_int(pitch_value)
            velocity = self._tensor_to_int(velocity_value)
            
            # 计算时间相关的值
            time_ticks = int(round(time * self.ticks_per_beat))
            duration_ticks = int(round(duration * self.ticks_per_beat))
            
            # 确保所有值都是有效的
            time_ticks = max(0, time_ticks)
            duration_ticks = max(1, duration_ticks)  # 持续时间至少为1
            pitch = max(0, min(127, pitch))  # MIDI音高范围0-127
            velocity = max(0, min(127, velocity))  # MIDI力度范围0-127
            
            return Note(
                time=time_ticks,
                duration=duration_ticks,
                pitch=pitch,
                velocity=velocity
            )
        except Exception as e:
            print(f"创建音符时出错: {str(e)}")
            return None

    def decode(self, input_data) -> Score:
        """将输入数据解码为多轨MIDI
        
        Args:
            input_data: 可以是图结构或音符列表
            
        Returns:
            Score: 解码后的乐谱
        """
        score = Score()
        score.ticks_per_quarter = self.ticks_per_beat
        
        # 根据输入类型处理
        if isinstance(input_data, dgl.DGLHeteroGraph):
            # 处理异构图
            # 主音轨道（琵琶）
            pipa_track = self._decode_main_notes(input_data)
            if pipa_track.notes:
                score.tracks.append(pipa_track)
            
            # 三弦轨道（低八度）
            sanxian_track = self._create_sanxian_track(pipa_track)
            if sanxian_track.notes:
                score.tracks.append(sanxian_track)
            
            # 洞箫轨道
            dongxiao_track = self._decode_dongxiao(input_data)
            if dongxiao_track.notes:
                score.tracks.append(dongxiao_track)
            
            # 二弦轨道
            erxian_track = self._decode_erxian(input_data)
            if erxian_track.notes:
                score.tracks.append(erxian_track)
            
        elif isinstance(input_data, dgl.DGLGraph):
            # 处理普通图
            track = Track(program=0)
            for nid in range(input_data.num_nodes()):
                node_data = input_data.ndata['note']
                try:
                    # 将拍数转换为ticks并确保是整数
                    time_ticks = int(round(float(node_data['position'][nid].item()) * self.ticks_per_beat))
                    duration_ticks = int(round(float(node_data['duration'][nid].item()) * self.ticks_per_beat))
                    velocity = self._tensor_to_int(node_data['velocity'][nid] * 0.9)
                    pitch = self._tensor_to_int(node_data['pitch'][nid])
                    
                    note = Note(
                        time=time_ticks,
                        duration=duration_ticks,
                        pitch=pitch,
                        velocity=velocity
                    )
                    track.notes.append(note)
                except Exception as e:
                    print(f"处理音符 {nid} 时出错: {str(e)}")
                    continue
            score.tracks.append(track)
            
        elif isinstance(input_data, (list, tuple)):
            # 处理音符列表
            track = Track(program=0)
            for note_data in input_data:
                try:
                    # 将拍数转换为ticks并确保是整数
                    time_ticks = int(round(float(note_data['start']) * self.ticks_per_beat))
                    duration_ticks = int(round(float(note_data['duration']) * self.ticks_per_beat))
                    velocity = self._tensor_to_int(note_data['velocity'] * 0.9)
                    pitch = self._tensor_to_int(note_data['pitch'])
                    
                    note = Note(
                        time=time_ticks,
                        duration=duration_ticks,
                        pitch=pitch,
                        velocity=velocity
                    )
                    track.notes.append(note)
                except Exception as e:
                    print(f"处理音符数据时出错: {str(e)}")
                    continue
            score.tracks.append(track)
            
        else:
            raise ValueError(f"不支持的输入类型: {type(input_data)}")
            
        return score
    
    def _decode_main_notes(self, graph) -> Track:
        track = Track(program=0)
        
        # 获取所有必要的特征数据
        positions = graph.nodes['note'].data.get('position', None)
        durations = graph.nodes['note'].data.get('duration', None)
        pitches = graph.nodes['note'].data.get('pitch', None)
        velocities = graph.nodes['note'].data.get('velocity', None)
        
        if positions is None or durations is None or pitches is None or velocities is None:
            print("警告：缺少必要的节点特征数据")
            return track
            
        for node_id in range(graph.num_nodes('note')):
            try:
                # 先转换为基本Python类型
                position = self._tensor_to_float(positions[node_id])
                duration = self._tensor_to_float(durations[node_id])
                pitch = self._tensor_to_int(pitches[node_id])
                velocity = self._tensor_to_float(velocities[node_id])
                
                # 创建音符
                note = self._create_note(
                    time_value=position,
                    duration_value=duration,
                    pitch_value=pitch,
                    velocity_value=velocity * 0.9  # 调整力度
                )
                
                if note is not None:
                    track.notes.append(note)
                    
            except Exception as e:
                print(f"处理音符 {node_id} 时出错: {str(e)}")
                continue
                
        return track
    
    def _create_sanxian_track(self, pipa_track: Track) -> Track:
        track = Track(program=1)
        for pipa_note in pipa_track.notes:
            try:
                # 创建音符
                note = self._create_note(
                    time_value=pipa_note.time / self.ticks_per_beat,  # 转换回拍数
                    duration_value=pipa_note.duration / self.ticks_per_beat,
                    pitch_value=pipa_note.pitch - 12,  # 降低一个八度
                    velocity_value=pipa_note.velocity * 0.8
                )
                
                if note is not None:
                    track.notes.append(note)
                    
            except Exception as e:
                print(f"创建三弦音符时出错: {str(e)}")
                continue
        return track
    
    def _decode_dongxiao(self, graph) -> Track:
        track = Track(program=3)
        if ('note', 'decorate', 'ornament') not in graph.canonical_etypes:
            return track
            
        src, dst = graph.edges(etype='decorate')
        for s, d in zip(src.tolist(), dst.tolist()):
            try:
                # 先转换为基本Python类型
                position = self._tensor_to_float(graph.nodes['note'].data['position'][s])
                duration = self._tensor_to_float(graph.nodes['note'].data['duration'][s])
                velocity = self._tensor_to_float(graph.nodes['note'].data['velocity'][s])
                pitch = self._tensor_to_int(graph.nodes['ornament'].data['pitch'][d])
                
                # 创建音符
                note = self._create_note(
                    time_value=position,
                    duration_value=duration * 0.8,  # 缩短持续时间
                    pitch_value=pitch + 12,  # 升高一个八度
                    velocity_value=velocity * 0.7  # 降低力度
                )
                
                if note is not None:
                    track.notes.append(note)
                    
            except Exception as e:
                print(f"处理装饰音 {s}->{d} 时出错: {str(e)}")
                continue
        return track
    
    def _decode_erxian(self, graph) -> Track:
        base_track = self._decode_dongxiao(graph)
        track = Track(program=2)
        for i, base_note in enumerate(base_track.notes):
            if i % 3 != 0 or np.random.rand() < 0.7:
                try:
                    # 创建音符
                    note = self._create_note(
                        time_value=base_note.time / self.ticks_per_beat,
                        duration_value=base_note.duration / self.ticks_per_beat,
                        pitch_value=base_note.pitch,
                        velocity_value=base_note.velocity
                    )
                    
                    if note is not None:
                        track.notes.append(note)
                        
                except Exception as e:
                    print(f"创建二弦音符时出错: {str(e)}")
                    continue
        return track