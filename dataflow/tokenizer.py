from miditok import MusicTokenizer, TokenizerConfig   
from symusic import Score, Note, Track
import torch
import os
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional
import dgl
from tqdm import tqdm
import json
import math

class NanyinTokenizerConfig(TokenizerConfig):
    """南音分词器配置"""
    def __init__(self, beat_res, num_velocities, num_microtiming_bins):
        # 调用父类初始化
        super().__init__(
            pitch_range=(21, 109),  # 标准MIDI音高范围
            beat_res=beat_res,
            num_velocities=num_velocities,
            special_tokens=["PAD", "BOS", "EOS", "MASK"],
            encode_ids_split="bar",
            use_velocities=True,
            use_note_duration_programs=list(range(-1, 128)),  # 所有程序都使用持续时间
            use_chords=False,
            use_rests=False,
            use_tempos=True,
            use_time_signatures=True,
            use_sustain_pedals=False,
            use_pitch_bends=False,
            use_pitch_intervals=False,
            use_programs=False,
            use_pitchdrum_tokens=True,
            default_note_duration=0.5,
            beat_res_rest={(0, 1): 8, (1, 2): 4, (2, 12): 2},
            chord_maps=None,
            chord_tokens_with_root_note=False,
            chord_unknown=None,
            num_tempos=32,
            tempo_range=(40, 250),
            log_tempos=False,
            remove_duplicated_notes=False,
            delete_equal_successive_tempo_changes=False,
            time_signature_range={8: [3, 12, 6], 4: [5, 6, 3, 2, 1, 4]},
            sustain_pedal_duration=False,
            pitch_bend_range=(-8192, 8191, 32),
            delete_equal_successive_time_sig_changes=False,
            programs=list(range(-1, 128)),
            one_token_stream_for_programs=True,
            program_changes=False,
            max_pitch_interval=16,
            pitch_intervals_max_time_dist=1,
            drums_pitch_range=(27, 88)
        )
        
        # 保存南音特有的配置
        self.num_microtiming_bins = num_microtiming_bins
        
    @property
    def max_num_pos_per_beat(self) -> int:
        """返回每拍的最大位置数"""
        return max(self.beat_res.values())
        
    def copy(self) -> 'NanyinTokenizerConfig':
        """复制配置对象"""
        return NanyinTokenizerConfig(
            beat_res=self.beat_res.copy(),
            num_velocities=self.num_velocities,
            num_microtiming_bins=self.num_microtiming_bins
        )
        
    def to_dict(self, serialize: bool = False) -> dict:
        """序列化为字典"""
        base_dict = super().to_dict(serialize)
        base_dict.update({
            'num_microtiming_bins': self.num_microtiming_bins
        })
        return base_dict
        
    @classmethod
    def from_dict(cls, input_dict: dict, **kwargs) -> 'NanyinTokenizerConfig':
        """从字典创建配置对象"""
        config = cls(
            beat_res=input_dict.get('beat_res', {(0, 4): 8, (4, 12): 4}),
            num_velocities=input_dict.get('num_velocities', 32),
            num_microtiming_bins=input_dict.get('num_microtiming_bins', 8)
        )
        return config

class NanyinTok(MusicTokenizer):
    # 将NANYIN_PITCHES定义为类变量，正确映射南音音高
    NANYIN_PITCHES = {
        50: "d",   # D3 (小字组D)
        52: "e",   # E3 (小字组E)
        53: "f",   # F3 (小字组F)
        54: "#f",  # F#3 (小字组F#)
        55: "g",   # G3 (小字组G)
        57: "a",   # A3 (小字组A)
        59: "b",   # B3 (小字组B)
        60: "c1",  # C4 (小字一组C)
        61: "#c1", # C#4 (小字一组C#)
        62: "d1",  # D4 (小字一组D)
        64: "e1",  # E4 (小字一组E)
        65: "f1",  # F4 (小字一组F)
        66: "#f1", # F#4 (小字一组F#)
        67: "g1",  # G4 (小字一组G)
        69: "a1",  # A4 (小字一组A)
        70: "bb1", # Bb4 (小字一组Bb)
        71: "b1",  # B4 (小字一组B)
        72: "c2",  # C5 (小字二组C)
        74: "d2",  # D5 (小字二组D)
        76: "e2",  # E5 (小字二组E)
        79: "g2",  # G5 (小字二组G)
        81: "a2",  # A5 (小字二组A)
        83: "b2"   # B5 (小字二组B)
    }
    
    # 将MODES定义为类变量
    MODES = {
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

    # 更新撚指阈值配置
    NIANZHI_THRESHOLDS = {
        'fast': {
            'max_interval': 120,
            'min_velocity_drop': 15,
            'confidence_threshold': 0.65,
            'min_notes': 3,
            'max_notes': 6
        },
        'standard': {
            'max_interval': 180,
            'min_velocity_drop': 12,
            'confidence_threshold': 0.6,
            'min_notes': 3,
            'max_notes': 5
        },
        'slow': {
            'max_interval': 250,
            'min_velocity_drop': 8,
            'confidence_threshold': 0.5,  # 降低慢速撚指阈值，使检测更宽松
            'min_notes': 3,
            'max_notes': 4
        }
    }

    def __init__(self, config: dict):
        """初始化tokenizer
        Args:
            config: 配置字典
        """
        print("初始化tokenizer...")
        
        # 从配置中读取tokenizer参数
        self.tokenizer_config = config.get('tokenizer', {})
        
        # 从配置中读取NANYIN_PITCHES映射，如果没有则使用默认映射
        config_pitches = self.tokenizer_config.get('nanyin_pitches')
        if config_pitches:
            # 将字符串键转换为整数键
            self.NANYIN_PITCHES = {int(k): v for k, v in config_pitches.items()}
            print(f"从配置文件加载NANYIN_PITCHES映射，包含 {len(self.NANYIN_PITCHES)} 个音高")
        else:
            print("配置中未找到nanyin_pitches映射，使用默认映射")
            # 保持使用类变量中定义的默认映射
            self.NANYIN_PITCHES = NanyinTok.NANYIN_PITCHES
            print(f"使用默认映射，包含 {len(self.NANYIN_PITCHES)} 个音高")
        
        # 修正beat_res的格式
        beat_res_config = self.tokenizer_config.get('beat_res', {})
        self.beat_res = {(0, 4): 16}  # 默认值
        if isinstance(beat_res_config, dict):
            for k, v in beat_res_config.items():
                if isinstance(k, str) and k.startswith('(') and k.endswith(')'):
                    try:
                        nums = k.strip('()').split(',')
                        key = (int(nums[0]), int(nums[1]))
                        self.beat_res[key] = v
                    except:
                        print(f"警告：无法解析beat_res键 {k}，使用默认值")
        
        self.ticks_per_quarter = self.tokenizer_config.get('ticks_per_quarter', 480)
        self.num_velocities = self.tokenizer_config.get('num_velocities', 32)
        self.num_microtiming_bins = self.tokenizer_config.get('num_microtiming_bins', 8)
        
        print(f"配置信息:")
        print(f"- ticks_per_quarter: {self.ticks_per_quarter}")
        print(f"- num_velocities: {self.num_velocities}")
        print(f"- beat_res: {self.beat_res}")
        
        # 先创建词汇表
        self._vocab = self._create_base_vocabulary()
        print(f"词汇表初始化完成:")
        print(f"- 总token数量: {len(self._vocab)}")
        print(f"- 包含的特殊token: {[k for k in self._vocab.keys() if k.startswith('Pitch_')][:5]}")
        
        # 创建配置对象
        config_obj = TokenizerConfig(
            pitch_range=(21, 109),  # 标准MIDI音高范围
            beat_res=self.beat_res,
            num_velocities=self.num_velocities,
            special_tokens=["PAD", "BOS", "EOS", "MASK"],
            encode_ids_split="bar",
            use_velocities=True,
            use_tempos=True,
            use_time_signatures=True,
            use_programs=False
        )
        
        # 初始化父类
        try:
            super().__init__(config_obj)
            self.config = config
        except Exception as e:
            print(f"Tokenizer初始化出错: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _create_base_vocabulary(self) -> Dict[str, int]:
        """创建基础词汇表
        Returns:
            Dict[str, int]: 基础词汇表
        """
        vocab = {}
        current_index = 0
        
        # 特殊标记
        special_tokens = ["PAD", "BOS", "EOS", "MASK", "UNK"]
        for token in special_tokens:
            vocab[token] = current_index
            current_index += 1
        
        # 音高词汇表
        pitch_names = ["UNK"] + list(set(self.NANYIN_PITCHES.values()))
        for pitch_name in pitch_names:
            vocab[f"Pitch_{pitch_name}"] = current_index
            current_index += 1
        
        # 时值词汇表
        for i in range(128):
            vocab[f"TimeShift_{i}"] = current_index
            current_index += 1
        
        # 力度词汇表
        for i in range(self.num_velocities):
            vocab[f"Velocity_{i}"] = current_index
            current_index += 1
        
        # 持续时间词汇表
        for i in range(128):
            vocab[f"Duration_{i}"] = current_index
            current_index += 1
        
        # 技法词汇表
        tech_types = ["None", "Nianzhi", "Diantiao"]
        for tech in tech_types:
            vocab[f"Tech_{tech}"] = current_index
            current_index += 1
            
        return vocab

    @property
    def vocab(self):
        """获取词汇表"""
        return self._vocab

    def _create_token_types_graph(self) -> Dict[str, set]:
        """创建token类型转换图
        Returns:
            Dict[str, set]: token类型转换图，使用集合存储可能的下一个类型
        """
        return {
            "Bar": {"TimeShift", "Pitch"},
            "TimeShift": {"Pitch"},
            "Pitch": {"Velocity", "Duration"},
            "Velocity": {"MicroTiming", "Duration"},
            "MicroTiming": {"Duration", "Tech_Nianzhi"},
            "Duration": {"TimeShift", "Pitch", "Tech_Nianzhi"},
            "Tech_Nianzhi": {"TimeShift", "Pitch"}
        }

    def _calculate_interval_trend(self, intervals: List[float]) -> float:
        """计算间隔变化趋势得分"""
        if len(intervals) < 2:
            return 0.5
        
        # 计算相邻间隔的比率，确保不会发生除零错误
        ratios = []
        for i in range(len(intervals)-1):
            if intervals[i] > 0:
                ratios.append(intervals[i+1] / intervals[i])
            else:
                ratios.append(1.0)  # 当间隔为0时，假定比率为1.0
        
        # 计算趋势得分
        trend_score = 0.0
        if ratios:
            # 递减趋势得分高
            decreasing_count = sum(1 for r in ratios if r <= 1.0)
            trend_score = decreasing_count / len(ratios)
            
            # 计算递减的平滑度
            if trend_score > 0.5:
                smoothness = 1.0
                ideal_ratio = 0.85
                for r in ratios:
                    if r > 0:
                        smoothness *= 1.0 - min(abs(r - ideal_ratio) / ideal_ratio, 1.0)
                trend_score = 0.7 * trend_score + 0.3 * smoothness
        
        return trend_score

    def _calculate_velocity_trend(self, velocities: List[int]) -> float:
        """计算力度变化趋势得分"""
        if len(velocities) < 2:
            return 0.5
        
        # 计算力度变化
        changes = [velocities[i+1] - velocities[i] for i in range(len(velocities)-1)]
        
        # 计算趋势得分
        decreasing_count = sum(1 for c in changes if c <= 0)
        trend_score = decreasing_count / len(changes)
        
        # 计算力度变化的平滑度
        if trend_score > 0.5:
            total_drop = velocities[0] - velocities[-1]
            # 避免除零错误
            if len(velocities) > 1 and total_drop != 0:
                ideal_drop_per_note = total_drop / (len(velocities) - 1)
                smoothness = 1.0
                for change in changes:
                    if change < 0 and ideal_drop_per_note != 0:  # 确保除数不为零
                        smoothness *= 1.0 - min(abs(change - (-ideal_drop_per_note)) / ideal_drop_per_note, 1.0)
                trend_score = 0.7 * trend_score + 0.3 * smoothness
        
        return trend_score

    def _calculate_rhythm_regularity(self, intervals: List[float]) -> float:
        """计算节奏规律性得分"""
        if len(intervals) < 2:
            return 0.5
        
        # 计算间隔的变异系数
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 0.0
        
        std_dev = math.sqrt(sum((x - mean_interval) ** 2 for x in intervals) / len(intervals))
        cv = std_dev / mean_interval
        
        # 变异系数越小，规律性越高
        regularity_score = max(0.0, min(1.0, 1.0 - cv))
        
        return regularity_score

    def determine_nianzhi_type(self, mean_interval: float, velocity_drop: float) -> str:
        """根据音符间隔和力度变化确定撚指类型"""
        if mean_interval < 100 and velocity_drop >= 15:
            return 'fast'
        elif mean_interval > 160 and velocity_drop >= 8:
            return 'slow'
        else:
            return 'standard'

    def calculate_nianzhi_confidence(self, feature_scores: Dict[str, float], nianzhi_type: str) -> float:
        """计算撚指置信度"""
        weights = {
            'fast': {
                'interval': 0.5,
                'velocity': 0.4,  # 增加快速撚指的力度权重
                'pitch': 0.15,
                'trend': 0.6,
                'regularity': 0.4
            },
            'slow': {
                'interval': 0.4,
                'velocity': 0.4,
                'pitch': 0.2,
                'trend': 0.45,  # 降低趋势权重
                'regularity': 0.55  # 增加慢速撚指的节奏规律性权重
            },
            'standard': {
                'interval': 0.45,
                'velocity': 0.35,
                'pitch': 0.2,
                'trend': 0.55,
                'regularity': 0.45
            }
        }
        
        w = weights[nianzhi_type]
        base_confidence = (
            feature_scores['interval'] * w['interval'] +
            feature_scores['velocity'] * w['velocity'] +
            feature_scores['pitch'] * w['pitch']
        )
        
        trend_confidence = (
            feature_scores['trend'] * w['trend'] +
            feature_scores['regularity'] * w['regularity']
        )
        
        # 综合评分：基础特征 * 0.7 + 趋势特征 * 0.3
        return base_confidence * 0.7 + trend_confidence * 0.3

    def _detect_nianzhi(self, notes: List[Note]) -> Tuple[bool, float, str]:
        """检测音符序列是否构成撚指"""
        if len(notes) < 3:
            return False, 0.0, ''
        
        # 计算基本特征
        intervals = []
        velocities = []
        pitches = set()
        
        for i in range(len(notes)-1):
            interval = notes[i+1].time - notes[i].time
            intervals.append(interval)
            velocities.append(notes[i].velocity)
            pitches.add(notes[i].pitch)
        velocities.append(notes[-1].velocity)
        pitches.add(notes[-1].pitch)
        
        # 基本特征计算
        mean_interval = sum(intervals) / len(intervals)
        velocity_drop = max(velocities) - min(velocities)
        pitch_consistency = 1.0 if len(pitches) == 1 else 0.0
        
        # 计算趋势特征
        interval_trend = self._calculate_interval_trend(intervals)
        velocity_trend = self._calculate_velocity_trend(velocities)
        rhythm_regularity = self._calculate_rhythm_regularity(intervals)
        
        # 确定撚指类型
        nianzhi_type = self.determine_nianzhi_type(mean_interval, velocity_drop)
        thresholds = self.NIANZHI_THRESHOLDS[nianzhi_type]
        
        # 根据撚指类型调整所需的最小音符数量
        min_notes = thresholds['min_notes']
        if nianzhi_type == 'fast' and len(notes) >= 6:  # 处理快速撚指的特殊情况
            min_notes = 6
        
        # 计算各个特征的分数
        interval_score = max(0, 1 - mean_interval / thresholds['max_interval'])
        velocity_score = min(1.0, velocity_drop / thresholds['min_velocity_drop'])
        trend_score = (interval_trend + velocity_trend) / 2
        
        # 综合特征分数
        feature_scores = {
            'interval': interval_score,
            'velocity': velocity_score,
            'pitch': pitch_consistency,
            'trend': trend_score,
            'regularity': rhythm_regularity
        }
        
        # 计算置信度
        confidence = self.calculate_nianzhi_confidence(feature_scores, nianzhi_type)
        
        # 判断是否满足撚指条件
        is_nianzhi = (confidence >= thresholds['confidence_threshold'] and
                     min_notes <= len(notes) <= thresholds['max_notes'])
        
        return is_nianzhi, confidence, nianzhi_type

    def _add_time_events(self, track: Track) -> List[Dict]:
        """添加时间相关事件
        Args:
            track: 音轨对象
        Returns:
            List[Dict]: 时间事件列表
        """
        events = []
        current_time = 0
        
        for note in sorted(track.notes, key=lambda x: x.time):
            # 计算时值
            time_shift = note.time - current_time
            if time_shift > 0:
                # 将时值转换为拍数，并使用掩码处理超出范围的值
                beats = time_shift / self.ticks_per_quarter
                quantized_beats = min(127, int(beats * self.beat_res[(0, 4)]))
                events.append({"type": "TimeShift", "value": quantized_beats})
                
                # 处理微时值
                micro_shift = time_shift % self.ticks_per_quarter
                if micro_shift > 0:
                    micro_bin = min(self.num_microtiming_bins - 1, 
                                  int(micro_shift * self.num_microtiming_bins / self.ticks_per_quarter))
                    events.append({"type": "MicroTiming", "value": micro_bin})
            
            current_time = note.time
            
        return events

    def _tokens_to_score(self, tokens: List[int]) -> Score:
        """将tokens转换回Score对象"""
        score = Score(self.ticks_per_quarter)
        track = Track()
        
        current_time = 0
        current_pitch = None
        current_velocity = None
        is_nianzhi = False
        nianzhi_notes = []
        
        for token in tokens:
            token_str = self.vocab[token]
            
            if token_str == "Tech_Nianzhi":
                is_nianzhi = True
                continue
                
            if token_str.startswith("Pitch_"):
                current_pitch = int(token_str.split("_")[1])
            elif token_str.startswith("Velocity_"):
                current_velocity = int(token_str.split("_")[1])
            elif token_str.startswith("Duration_"):
                duration = int(token_str.split("_")[1]) * self.ticks_per_quarter
                if current_pitch is not None and current_velocity is not None:
                    note = Note(
                        time=current_time,
                        duration=duration,
                        pitch=current_pitch,
                        velocity=current_velocity
                    )
                    
                    if is_nianzhi:
                        nianzhi_notes.append(note)
                        if len(nianzhi_notes) >= 3:  # 修改为至少3个音符
                            for n in nianzhi_notes:
                                track.notes.append(n)
                            nianzhi_notes = []
                            is_nianzhi = False
                    else:
                        track.notes.append(note)
                        
            elif token_str.startswith("TimeShift_"):
                current_time += int(token_str.split("_")[1]) * self.ticks_per_quarter
                
        score.tracks.append(track)
        return score

    def _score_to_tokens(self, score: Score) -> List[int]:
        """将Score对象转换为tokens，使用掩码处理超出范围的值"""
        tokens = []
        notes_window = []
        is_nianzhi = False
        current_time = 0
        
        try:
            for track in score.tracks:
                if not track.notes:
                    continue
                
                for note in sorted(track.notes, key=lambda x: x.time):
                    # 添加时间事件
                    time_shift = note.time - current_time
                    if time_shift > 0:
                        beats = time_shift / self.ticks_per_quarter
                        quantized_beats = min(127, max(0, int(beats * self.beat_res[(0, 4)])))
                        time_token = self.vocab.get(f"TimeShift_{quantized_beats}", self.vocab["TimeShift_0"])
                        tokens.append(time_token)
                    
                    # 处理音高 - 使用掩码处理超出范围的音高
                    pitch_name = self.NANYIN_PITCHES.get(note.pitch, "UNK")
                    pitch_token = self.vocab.get(f"Pitch_{pitch_name}", self.vocab["Pitch_UNK"])
                    tokens.append(pitch_token)
                    mode_masks.append(True)
                    
                    # 处理力度 - 使用掩码处理超出范围的力度
                    velocity_bin = min(self.num_velocities - 1, 
                                     int(note.velocity * self.num_velocities / 128))
                    tokens.append(self.vocab[f"Velocity_{velocity_bin}"])
                    mode_masks.append(False)
                    
                    # 处理时长 - 使用掩码处理超出范围的时长
                    duration_beats = note.duration / self.ticks_per_quarter
                    quantized_duration = min(127, int(duration_beats * self.beat_res[(0, 4)]))
                    tokens.append(self.vocab[f"Duration_{quantized_duration}"])
                    mode_masks.append(False)
                    
                    current_time = note.time
                    
                    # 更新滑动窗口
                    notes_window.append(note)
                    if len(notes_window) > 15:
                        notes_window.pop(0)
                    
                    # 检测撚指
                    if len(notes_window) >= 3 and self._detect_nianzhi(notes_window):
                        if not is_nianzhi:
                            is_nianzhi = True
                            tokens.append(self.vocab["Tech_Nianzhi"])
                    else:
                        is_nianzhi = False
            
            return tokens
            
        except Exception as e:
            print(f"转换tokens时出错: {str(e)}")
            # 返回最小有效的token序列
            return [
                self.vocab["Pitch_UNK"],
                self.vocab["Velocity_0"],
                self.vocab["Duration_0"]
            ]

    def _get_mode_mask(self, notes: List[Note], mode: str) -> List[bool]:
        """
        Args:
            notes: 音符列表
            mode: 调式名称
            
        Returns:
            List[bool]: 模式掩码，True表示音符在调式内，False表示在调式外
        """
        if mode not in self.MODES:
            return [False] * len(notes)

        mode_scale = self.MODES[mode]["scale"]
        return [note.pitch in mode_scale for note in notes]

    def _tokenize_with_mode_mask(self, notes: List[Note], mode: str) -> Tuple[List[int], List[bool]]:
        """带有模式掩码的tokenization
        
        Args:
            notes: 音符列表
            mode: 调式名称
            
        Returns:
            Tuple[List[int], List[bool]]: (tokens, mode_mask)
        """
        tokens = []
        mode_mask = self._create_mode_masks(notes, mode)
        
        for note, in_mode in zip(notes, mode_mask):
            # 基本的音符tokens
            note_tokens = [
                self.vocab[f"Pitch_{self.NANYIN_PITCHES[note.pitch]}"],
                self.vocab[f"Velocity_{note.velocity}"],
                self.vocab[f"Duration_{note.duration // self.ticks_per_quarter}"]
            ]
            
            # 如果是撚指，添加撚指标记
            if self._detect_nianzhi([note]):  # 这里需要根据实际情况修改检测逻辑
                note_tokens.append(self.vocab["Tech_Nianzhi"])
            
            tokens.extend(note_tokens)
            
        return tokens, mode_mask

    def __call__(self, score: Score, mode: str = "G") -> Tuple[List[int], List[List[bool]]]:
        """处理完整的乐谱，返回tokens和模式掩码
        
        Args:
            score: 乐谱对象
            mode: 默认调式
            
        Returns:
            Tuple[List[int], List[List[bool]]]: (tokens, mode_masks)
        """
        all_tokens = []
        all_mode_masks = []
        
        for track in score.tracks:
            if not track.notes:
                continue
                
            track_tokens, track_mask = self._tokenize_with_mode_mask(track.notes, mode)
            all_tokens.extend(track_tokens)
            all_mode_masks.append(track_mask)
            
        return all_tokens, all_mode_masks

    def generate_with_mode_constraint(self, tokens: List[int], mode_masks: List[bool], temperature: float = 1.0):
        """使用模式掩码约束的生成"""
        generated = []  # 确保这行缩进是 4 个空格
        is_nianzhi = False
        nianzhi_tokens = []

        for token, in_mode in zip(tokens, mode_masks):
            token_str = self.vocab[token]

            if token_str == "Tech_Nianzhi":
                is_nianzhi = True
                generated.append(token)
            else:
                if is_nianzhi:
                    nianzhi_tokens.append(token)
                    if token_str.startswith("Duration_"):
                        is_nianzhi = False
                        # 处理撚指音符
                        generated.extend(nianzhi_tokens)
                        nianzhi_tokens = []
                else:
                    generated.append(token)

        return generated  # 确保 return 语句在方法内部并正确缩进

    def tokenize(self, score: Score) -> Dict:
        """将Score对象转换为tokens"""
        tokens = []
        mode_masks = []
        
        try:
            print("开始tokenization过程...")
            print(f"词汇表大小: {len(self.vocab)}")
            
            # 预处理音乐文件
            self._preprocess_score(score)
            
            # 处理每个轨道
            for track_idx, track in enumerate(score.tracks):
                print(f"\n处理轨道 {track_idx}:")
                print(f"音符数量: {len(track.notes)}")
                if track.notes:
                    print(f"第一个音符信息: 音高={track.notes[0].pitch}, 力度={track.notes[0].velocity}")
                
                if not track.notes:
                    print(f"警告：轨道 {track_idx} 没有音符")
                    continue
                    
                track_tokens, track_masks = self._tokenize_track(track)
                print(f"轨道 {track_idx} tokenization结果:")
                print(f"生成的token数量: {len(track_tokens)}")
                
                if track_tokens and track_masks:  # 只添加非空的结果
                    tokens.extend(track_tokens)
                    mode_masks.extend(track_masks)
            
            # 验证token序列
            if not tokens:
                print("警告：没有生成任何token，返回空结果")
                return self._create_empty_result()
            
            # 构建结果字典
            result = {
                'token_sequence': tokens,
                'mode_mask': mode_masks,
                'tech_positions': self._get_tech_positions(tokens),
                'tech_types': self._get_tech_types(tokens)
            }
            
            print("\nTokenization完成")
            print(f"技法位置数量: {len(result['tech_positions'])}")
            
            return result
            
        except Exception as e:
            print(f"Tokenize出错: {str(e)}")
            print("错误详细信息:", e.__class__.__name__)
            import traceback
            traceback.print_exc()
            return self._create_empty_result()
            
    def _create_empty_result(self) -> Dict:
        """创建空的结果字典"""
        return {
            'token_sequence': [],
            'mode_mask': [],
            'tech_positions': [],
            'tech_types': []
        }
        
    def _detect_nianzhi_pattern(self, tokens: List[int]) -> bool:
        """检测是否为撚指模式"""
        try:
            if len(tokens) != 4:
                return False
                
            # 获取token对应的字符串
            token_strs = [self.vocab.get(t, "") for t in tokens]
            
            # 检查模式：音高 + 力度 + 短时值 + 技法标记
            is_valid_pitch = token_strs[0].startswith("Pitch_")
            is_valid_velocity = token_strs[1].startswith("Velocity_")
            is_short_duration = (
                token_strs[2].startswith("Duration_") and 
                int(token_strs[2].split("_")[1]) < self.ticks_per_quarter // 2
            )
            is_tech_token = token_strs[3] == "Tech_Nianzhi"
            
            return all([is_valid_pitch, is_valid_velocity, is_short_duration, is_tech_token])
            
        except Exception as e:
            print(f"检测撚指模式时出错: {str(e)}")
            return False

    def _tokenize_track(self, track: Track) -> Tuple[List[int], List[bool]]:
        """将音轨转换为token序列，增强撚指检测
        
        Args:
            track: 音轨对象
            
        Returns:
            Tuple[List[int], List[bool]]: (token列表, 模式掩码列表)
        """
        tokens = []
        mode_masks = []
        notes_window = []
        current_time = 0
        is_nianzhi = False
        
        try:
            # 过滤并排序有效音符，添加详细调试信息
            print(f"音符总数: {len(track.notes)}")
            valid_notes = []
            invalid_notes_count = 0
            
            for i, note in enumerate(sorted(track.notes, key=lambda x: x.time)):
                try:
                    # 直接访问属性并检查值
                    pitch = note.pitch
                    velocity = note.velocity
                    duration = note.duration
                    time = note.time
                    
                    # 输出前5个音符的详细信息用于调试
                    if i < 5:
                        print(f"音符 {i} 详情: 音高={pitch}, 力度={velocity}, 持续时间={duration}, 开始时间={time}")
                    
                    # 基于值检查音符有效性，允许持续时间为0
                    if duration >= 0:  # 放宽条件，允许持续时间为0
                        valid_notes.append(note)
                    else:
                        invalid_notes_count += 1
                        if i < 5:
                            print(f"音符 {i} 无效: 持续时间为 {duration}")
                except Exception as e:
                    invalid_notes_count += 1
                    print(f"处理音符 {i} 时出错: {str(e)}")
            
            print(f"有效音符数: {len(valid_notes)}, 无效音符数: {invalid_notes_count}")
            
            if not valid_notes:
                print("警告：轨道中没有有效音符")
                return [], []
            
            # 预处理：检测所有可能的撚指段落
            nianzhi_segments = self._detect_nianzhi_in_sequence(valid_notes)
            print(f"检测到的撚指段落数量: {len(nianzhi_segments)}")
            
            # 创建撚指标记映射
            nianzhi_map = set()
            for start_idx, end_idx, conf in nianzhi_segments:
                for i in range(start_idx, end_idx + 1):
                    nianzhi_map.add(i)
            
            # 处理每个音符
            for i, note in enumerate(valid_notes):
                # 检查是否是撚指开始
                is_nianzhi_start = False
                for start_idx, end_idx, conf in nianzhi_segments:
                    if i == start_idx:
                        is_nianzhi_start = True
                        print(f"检测到撚指开始, 位置: {i}, 置信度: {conf:.3f}")
                        break
                
                # 如果是撚指开始，添加撚指标记
                if is_nianzhi_start:
                    tech_token = self.vocab["Tech_Nianzhi"]
                    tokens.append(tech_token)
                    mode_masks.append(False)
                
                # 1. 处理时间偏移
                time_shift = note.time - current_time
                if time_shift > 0:
                    beats = time_shift / self.ticks_per_quarter
                    quantized_beats = min(127, max(0, int(beats * self.beat_res[(0, 4)])))
                    time_token = self.vocab.get(f"TimeShift_{quantized_beats}", self.vocab["TimeShift_0"])
                    tokens.append(time_token)
                    mode_masks.append(False)
                current_time = note.time
                
                # 2. 处理音高
                pitch_name = self.NANYIN_PITCHES.get(note.pitch, "UNK")
                pitch_token = self.vocab.get(f"Pitch_{pitch_name}", self.vocab["Pitch_UNK"])
                tokens.append(pitch_token)
                mode_masks.append(True)
                
                # 3. 处理力度
                velocity_bin = min(self.num_velocities - 1, 
                                max(0, int(note.velocity * self.num_velocities / 128)))
                velocity_token = self.vocab.get(f"Velocity_{velocity_bin}", self.vocab["Velocity_0"])
                tokens.append(velocity_token)
                mode_masks.append(False)
                
                # 4. 处理持续时间
                duration_beats = note.duration / self.ticks_per_quarter
                quantized_duration = min(127, max(1, int(duration_beats * self.beat_res[(0, 4)])))
                duration_token = self.vocab.get(f"Duration_{quantized_duration}", self.vocab["Duration_1"])
                tokens.append(duration_token)
                mode_masks.append(False)
            
            print(f"生成token数量: {len(tokens)}")
            return tokens, mode_masks
            
        except Exception as e:
            print(f"_tokenize_track出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], []

    def _get_tech_positions(self, tokens: List[int]) -> List[int]:
        """获取技法位置"""
        tech_positions = []
        try:
            # 创建反向映射
            reverse_vocab = {v: k for k, v in self._vocab.items()}
            
            for i, token in enumerate(tokens):
                if token in reverse_vocab and reverse_vocab[token] == "Tech_Nianzhi":
                    tech_positions.append(i)
        except Exception as e:
            print(f"获取技法位置时出错: {str(e)}")
        return tech_positions

    def _get_tech_types(self, tokens: List[int]) -> List[str]:
        """获取技法类型"""
        tech_types = []
        try:
            # 创建反向映射
            reverse_vocab = {v: k for k, v in self._vocab.items()}
            
            for token in tokens:
                if token in reverse_vocab and reverse_vocab[token] == "Tech_Nianzhi":
                    tech_types.append("nianzhi")
        except Exception as e:
            print(f"获取技法类型时出错: {str(e)}")
        return tech_types


    def _save_processed(self, processed: Dict, output_path: str) -> None:
        """保存处理后的数据
        Args:
            processed: 处理后的数据
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 将数据转换为可序列化的格式
        serializable_data = {
            'token_sequence': processed['token_sequence'],
            'mode_mask': processed['mode_mask'],
            'tech_positions': processed['tech_positions'],
            'tech_types': processed['tech_types']
        }
        
        # 保存为JSON文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    def _preprocess_score(self, score: Score) -> None:
        """预处理Score对象
        Args:
            score: Score对象
        """
        # 1. 时间量化
        self._quantize_time(score)
        
        # 2. 量化力度和速度
        self._quantize_velocities_and_tempos(score)

    def _quantize_time(self, score: Score) -> None:
        """量化时间
        Args:
            score: Score对象
        """
        ticks_per_beat = score.ticks_per_quarter
        for track in score.tracks:
            for note in track.notes:
                # 量化开始时间
                note.time = int(round(note.time / ticks_per_beat) * ticks_per_beat)
                # 量化持续时间
                note.duration = int(round(note.duration / ticks_per_beat) * ticks_per_beat)


    def _quantize_velocities_and_tempos(self, score: Score) -> None:
        """量化力度和速度
        Args:
            score: Score对象
        """
        # 量化力度
        for track in score.tracks:
            for note in track.notes:
                note.velocity = int(note.velocity * self.num_velocities / 128)
        
        # 量化速度
        if hasattr(score, 'tempos'):
            for tempo in score.tempos:
                tempo.tempo = int(round(tempo.tempo / 5) * 5)  # 量化到最接近的5

    

    def process_file(self, input_path: str, output_path: str) -> None:
        """处理单个MIDI文件"""
        try:
            # 检查文件是否存在
            if not os.path.isfile(input_path):
                print(f"文件不存在: {input_path}")
                return
                
            # 读取MIDI文件
            score = Score(input_path)
            
            # 预处理音乐文件
            self._preprocess_score(score)
            
            # 处理并获取token数据
            processed = self.tokenize(score)
            
            # 验证数据格式
            if not isinstance(processed, dict):
                raise ValueError(f"处理结果必须是字典格式，而不是 {type(processed)}")
            
            if not isinstance(processed.get('token_sequence', None), list):
                raise ValueError("token_sequence 必须是列表格式")
            
            # 保存处理后的数据
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(processed, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"处理文件 {input_path} 时出错: {str(e)}")
            # 保存最小有效数据
            minimal_data = {
                'token_sequence': [],
                'mode_mask': [],
                'tech_positions': [],
                'tech_types': []
            }
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(minimal_data, f, ensure_ascii=False, indent=2)

    def process_directory(self, input_dir: str, output_dir: str) -> None:
        """处理目录中的所有MIDI文件
        Args:
            input_dir: 输入目录路径
            output_dir: 输出目录路径
        """
        import os
        from pathlib import Path
        from tqdm import tqdm
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"开始处理MIDI文件，输入目录：{input_dir}，输出目录：{output_dir}")
        
        # 获取所有MIDI文件（只处理当前目录下的文件，不递归）
        input_path = Path(input_dir)
        midi_files = list(input_path.glob("*.mid")) + list(input_path.glob("*.midi"))
        print(f"找到 {len(midi_files)} 个MIDI文件")
        
        processed_count = 0
        error_count = 0
        
        for midi_path in tqdm(midi_files, desc="处理MIDI文件"):
            try:
                # 生成输出文件路径
                output_path = Path(output_dir) / midi_path.name.replace('.mid', '.tok').replace('.midi', '.tok')
                
                # 处理文件
                self.process_file(str(midi_path), str(output_path))
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"处理文件 {midi_path} 时出错: {str(e)}")
                continue
        
        print(f"\n处理完成：")
        print(f"成功处理：{processed_count} 个文件")
        print(f"处理失败：{error_count} 个文件")
        print(f"处理后的文件保存在：{output_dir}")

    def _detect_nianzhi_in_sequence(self, notes: List[Note], window_size=15):
        """在音符序列中检测撚指模式
        
        使用滑动窗口和分段分析，更准确地检测撚指模式。
        
        Args:
            notes: 音符列表
            window_size: 滑动窗口大小
            
        Returns:
            List[Tuple[int, int, float]]: 检测到的撚指段落 [(起始索引, 结束索引, 置信度)]
        """
        if len(notes) < 3:
            return []
        
        print(f"开始撚指序列检测，音符数量: {len(notes)}")
        
        # 存储检测到的撚指段落
        nianzhi_segments = []
        
        # 尝试1：使用滑动窗口检测连续的撚指模式
        for i in range(len(notes) - 2):  # 至少需要3个音符
            for window_len in range(3, min(8, len(notes) - i + 1)):
                window = notes[i:i+window_len]
                
                # 应用增强型撚指检测，降低检测阈值
                is_nianzhi, confidence, nianzhi_type = self._detect_nianzhi(window)
                
                # 对于测试目的，降低置信度阈值至0.4
                if is_nianzhi or confidence > 0.4:
                    print(f"检测到潜在撚指: 开始={i}, 结束={i+window_len-1}, 置信度={confidence:.3f}, 类型={nianzhi_type}")
                    nianzhi_segments.append((i, i+window_len-1, confidence))
                    
                    # 跳过已检测的窗口，减少重叠
                    break
        
        # 尝试2：按照音高分组音符序列（原有方法的优化版本）
        pitch_groups = {}
        for i, note in enumerate(notes):
            if note.pitch not in pitch_groups:
                pitch_groups[note.pitch] = []
            pitch_groups[note.pitch].append((i, note))
        
        # 对每个音高组分析可能的撚指
        for pitch, note_indices in pitch_groups.items():
            # 如果同音高的音符太少，跳过
            if len(note_indices) < 3:
                continue
                
            # 提取同音高的音符
            indices, pitch_notes = zip(*note_indices)
            
            # 检查是否有时间上连续的音符组
            i = 0
            while i < len(pitch_notes) - 2:  # 至少需要3个音符
                # 尝试不同的窗口大小
                for window_len in range(3, min(8, len(pitch_notes) - i + 1)):
                    window = list(pitch_notes[i:i+window_len])
                    
                    # 检查时间连续性，放宽阈值
                    is_continuous = True
                    for j in range(len(window) - 1):
                        interval = window[j+1].time - window[j].time
                        if interval > self.ticks_per_quarter * 1.2:  # 更宽松的阈值
                            is_continuous = False
                            break
                    
                    if not is_continuous:
                        continue
                        
                    # 应用增强型撚指检测
                    is_nianzhi, confidence, nianzhi_type = self._detect_nianzhi(window)
                    
                    # 对于测试目的，降低置信度阈值至0.4
                    if is_nianzhi or confidence > 0.4:
                        start_idx = indices[i]
                        end_idx = indices[i+window_len-1]
                        print(f"按音高分组方法检测到潜在撚指: 开始={start_idx}, 结束={end_idx}, 置信度={confidence:.3f}, 类型={nianzhi_type}")
                        nianzhi_segments.append((start_idx, end_idx, confidence))
                        
                        # 跳过已检测的窗口
                        i += window_len
                        break
                else:
                    i += 1  # 如果没有检测到撚指，向前移动一步
        
        # 合并重叠的撚指段落
        if nianzhi_segments:
            nianzhi_segments.sort(key=lambda x: x[0])  # 按起始位置排序
            merged_segments = [nianzhi_segments[0]]
            
            for start, end, conf in nianzhi_segments[1:]:
                prev_start, prev_end, prev_conf = merged_segments[-1]
                
                # 如果当前段落与前一个重叠
                if start <= prev_end:
                    # 取并集，并使用更高的置信度
                    new_end = max(end, prev_end)
                    new_conf = max(conf, prev_conf)
                    merged_segments[-1] = (prev_start, new_end, new_conf)
                else:
                    merged_segments.append((start, end, conf))
            
            print(f"合并后的撚指段落数量: {len(merged_segments)}")
            return merged_segments
        
        print("未检测到任何撚指段落")
        return []

    def extract_nianzhi_features(self, notes: List[Note]) -> Optional[Dict]:
        """提取撚指特征"""
        if len(notes) < 3:
            return None
        
        is_nianzhi, confidence, nianzhi_type = self._detect_nianzhi(notes)
        if not is_nianzhi:
            return None
        
        # 提取基本特征
        intervals = [notes[i+1].time - notes[i].time for i in range(len(notes)-1)]
        velocities = [note.velocity for note in notes]
        durations = [note.duration for note in notes]
        
        # 计算趋势特征
        interval_trend = self._calculate_interval_trend(intervals)
        velocity_trend = self._calculate_velocity_trend(velocities)
        rhythm_regularity = self._calculate_rhythm_regularity(intervals)
        
        # 计算平均间隔
        mean_interval = sum(intervals) / len(intervals) if intervals else 0
        
        # 修复：避免除零错误
        max_interval = max(intervals) if intervals and max(intervals) > 0 else 1
        interval_profile = [i/max_interval for i in intervals] if intervals else []
        
        # 力度特征
        initial_velocity = velocities[0] if velocities else 0
        velocity_drop = (velocities[0] - velocities[-1]) if len(velocities) > 1 else 0
        
        # 修复：避免除零错误
        max_velocity_diff = max(abs(v - velocities[0]) for v in velocities[1:]) if len(velocities) > 1 else 1
        velocity_profile = [(velocities[0] - v)/max_velocity_diff if max_velocity_diff > 0 else 0 for v in velocities[1:]] if len(velocities) > 1 else []
        
        # 持续时间特征
        mean_duration = sum(durations) / len(durations) if durations else 0
        
        # 修复：避免除零错误
        max_duration = max(durations) if durations and max(durations) > 0 else 1
        duration_profile = [d/max_duration for d in durations] if durations else []
        
        # 确定撚指类型
        nianzhi_category = "unknown"
        if mean_interval < 50:  # 快速撚指
            if velocity_drop > 30:
                nianzhi_category = "fast_intense"
            else:
                nianzhi_category = "fast_soft"
        else:  # 慢速撚指
            if velocity_drop > 15:
                nianzhi_category = "slow_intense"
            else:
                nianzhi_category = "slow_soft"
                
        return {
            'mean_interval': mean_interval,
            'interval_std': interval_trend,
            'interval_profile': interval_profile,
            'initial_velocity': initial_velocity,
            'velocity_drop': velocity_drop,
            'velocity_profile': velocity_profile,
            'mean_duration': mean_duration,
            'duration_profile': duration_profile,
            'nianzhi_type': nianzhi_type,
            'confidence': confidence,
            'category': nianzhi_category,
            'note_count': len(notes)
        }
