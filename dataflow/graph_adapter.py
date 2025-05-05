import dgl
import torch
import numpy as np
import os
from collections import defaultdict
from typing import List, Dict, Tuple

class NanyinGraphAdapter:
    """将NanyinTok的结构化输出转换为南音异构图"""
    
    def __init__(self, config: dict):
        """初始化图适配器
        Args:
            config: 配置字典
        """
        # 从tokenizer配置中获取nanyin_pitches和MODES映射
        tokenizer_config = config.get('tokenizer', {})
        
        # 正确获取nanyin_pitches映射
        config_pitches = tokenizer_config.get('nanyin_pitches')
        if config_pitches:
            # 将字符串键转换为整数键
            self.pitch_map = {int(k): v for k, v in config_pitches.items()}
            print(f"从配置文件加载nanyin_pitches映射，包含 {len(self.pitch_map)} 个音高")
        else:
            print("警告：配置中未找到nanyin_pitches映射，使用默认映射")
            # 使用默认映射
            self.pitch_map = {
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
            
        # 临时使用硬编码的MODES，后续需要从配置读取
        self.modes = {
            "WUKONG": {
                "name": "五空管",
                "upper_register": set([62, 64, 67, 69, 71, 74, 76, 79, 81, 83]),  # d1-b2
                "lower_register": set([50, 52, 55, 57, 60, 62, 64, 67, 69])  # d-a1
            },
            "SIKONG": {
                "name": "四空管",
                "scale": set([50, 53, 55, 57, 60, 62, 65, 67, 69, 72, 74, 76, 79, 81])  # d-a2
            }
        }
        
        # 获取其他配置
        self.config = config
        model_config = config.get('model', {})
        
        # 设置默认值
        self.base_pitch = model_config.get('base_pitch', 60)  # 默认中央C
        self.default_tempo = model_config.get('default_tempo', 80)
        
        # 初始化其他成员变量
        self.current_tempo = self.default_tempo
        self.current_time = 0
        self.nodes = []
        self.edges = []
        
        # 创建反向映射
        self.reverse_pitch_map = {v: k for k, v in self.pitch_map.items()}
        
        # 特色音符号
        self.special_notes_symbols = ['#f', '#c1', '#f1']
        
        # 特色音MIDI音高映射
        self.special_note_map = {
            '#f': 54,   # F#3音（小字组F#，MIDI音高54）
            '#c1': 61,  # C#4音（小字一组C#，MIDI音高61）
            '#f1': 66   # F#4音（小字一组F#，MIDI音高66）
        }
        
        # 特色音MIDI音高集合（用于快速检查）
        self.special_notes_midi = set(self.special_note_map.values())
        
        # 为了与rule_injector.py保持一致，使用MIDI音高值
        self.special_notes = self.special_notes_midi
        
        self.tech_types = ['nianzhi', 'diantiao']
        self.processed_graphs = []
        
    def get_pitch_name(self, midi_pitch: int) -> str:
        """获取MIDI音高对应的名称
        Args:
            midi_pitch: MIDI音高值
        Returns:
            str: 音高名称
        """
        return self.pitch_map.get(midi_pitch, 'UNK')
        
    def get_midi_pitch(self, pitch_name: str) -> int:
        """获取音高名称对应的MIDI值
        Args:
            pitch_name: 音高名称
        Returns:
            int: MIDI音高值
        """
        return self.reverse_pitch_map.get(pitch_name, -1)
        
    def is_valid_pitch(self, midi_pitch: int) -> bool:
        """检查是否为有效的音高
        Args:
            midi_pitch: MIDI音高值
        Returns:
            bool: 是否有效
        """
        return midi_pitch in self.pitch_map
        
    def get_mode_scale(self, mode_name: str) -> set:
        """获取调式的音阶
        Args:
            mode_name: 调式名称
        Returns:
            set: 音阶音高集合
        """
        mode = self.modes.get(mode_name, {})
        if 'scale' in mode:
            return mode['scale']
        elif 'upper_register' in mode and 'lower_register' in mode:
            return mode['upper_register'] | mode['lower_register']
        return set()
        
    def is_in_mode(self, midi_pitch: int, mode_name: str) -> bool:
        """检查音高是否在调式中
        Args:
            midi_pitch: MIDI音高值
            mode_name: 调式名称
        Returns:
            bool: 是否在调式中
        """
        scale = self.get_mode_scale(mode_name)
        return midi_pitch in scale
    
    def convert_to_graphs(self, processed_dir: str):
        """处理目录中的所有token化数据并转换为图
        
        Args:
            processed_dir: 包含token化数据的目录路径
        """
        for filename in os.listdir(processed_dir):
            if filename.endswith('.tok'):
                filepath = os.path.join(processed_dir, filename)
                with open(filepath, 'r') as f:
                    tokenized_data = eval(f.read())  # 读取Python字典格式的数据
                graph = self.convert(tokenized_data)
                self.processed_graphs.append(graph)
                
                # 保存转换后的图
                graph_path = os.path.join(processed_dir, f"{os.path.splitext(filename)[0]}.dgl")
                dgl.save_graphs(graph_path, [graph])
    
    def get_processed_graphs(self) -> List[dgl.DGLHeteroGraph]:
        """获取所有处理后的图
        
        Returns:
            List[dgl.DGLHeteroGraph]: 处理后的图列表
        """
        return self.processed_graphs

    def convert(self, tokenized_data: dict) -> dgl.DGLHeteroGraph:
        """执行核心转换逻辑，添加错误处理和数据验证"""
        try:
            # 验证输入数据
            if not isinstance(tokenized_data, dict):
                print(f"输入数据类型错误: {type(tokenized_data)}")
                return self._create_minimal_graph()
            
            # 获取token序列
            token_seq = tokenized_data.get('token_sequence')
            if token_seq is None:
                print("token_sequence不存在")
                return self._create_minimal_graph()
            
            if not isinstance(token_seq, list):
                print(f"token_sequence类型错误: {type(token_seq)}")
                return self._create_minimal_graph()
            
            if not token_seq:
                print("token_sequence为空")
                return self._create_minimal_graph()
            
            # 打印调试信息
            print(f"处理token序列:")
            print(f"  - 长度: {len(token_seq)}")
            print(f"  - 前几个token: {token_seq[:8]}")
            
            # 确保所有token都是整数
            try:
                token_seq = [int(t) for t in token_seq]
            except (ValueError, TypeError) as e:
                print(f"转换token为整数时出错: {str(e)}")
                return self._create_minimal_graph()
            
            # 计算音符数量
            note_count = len(token_seq) // 4
            
            try:
                # 构建节点特征
                # 确保token序列长度足够
                if len(token_seq) < 4:
                    print("token序列太短，无法构建有效的音符")
                    return self._create_minimal_graph()
                
                # 计算实际的音符数量（确保不会越界）
                note_count = len(token_seq) // 4
                
                # 安全地提取特征
                pitches = []
                velocities = []
                durations = []
                
                for i in range(0, len(token_seq) - 3, 4):
                    try:
                        pitches.append(token_seq[i])
                        velocities.append(token_seq[i + 1])
                        durations.append(token_seq[i + 2])
                    except IndexError:
                        break
                
                # 转换为张量
                pitch = torch.tensor(pitches, dtype=torch.long)
                velocity = torch.tensor(velocities, dtype=torch.long)
                duration = torch.tensor(durations, dtype=torch.long)
                
                # 创建图结构
                graph_data = {
                    ('note', 'temporal', 'note'): ([], []),
                    ('note', 'decorate', 'ornament'): ([], []),
                    ('tech', 'trigger', 'note'): ([], [])
                }
                
                # 创建图
                graph = dgl.heterograph(graph_data)
                
                # 添加节点
                actual_note_count = len(pitches)
                graph.add_nodes(actual_note_count, ntype='note')
                
                # 添加基本节点特征
                graph.nodes['note'].data['pitch'] = pitch
                graph.nodes['note'].data['velocity'] = velocity
                graph.nodes['note'].data['duration'] = duration
                graph.nodes['note'].data['position'] = torch.arange(actual_note_count)
                
                # 添加时序边
                if actual_note_count > 1:
                    src = torch.arange(actual_note_count - 1)
                    dst = torch.arange(1, actual_note_count)
                    graph.add_edges(src, dst, etype='temporal')
                
                # 处理技法
                tech_positions = tokenized_data.get('tech_positions', [])
                tech_types = tokenized_data.get('tech_types', [])
                if tech_positions and tech_types:
                    # 确保技法位置在有效范围内
                    valid_tech_positions = [pos for pos in tech_positions if pos//4 < actual_note_count]
                    valid_tech_types = tech_types[:len(valid_tech_positions)]
                    
                    if valid_tech_positions:
                        tech_ids = torch.tensor([pos//4 for pos in valid_tech_positions], dtype=torch.long)
                        type_indices = [self.tech_types.index(t) if t in self.tech_types else 0 
                                      for t in valid_tech_types]
                        tech_type_tensor = torch.tensor(type_indices, dtype=torch.long)
                        
                        # 添加技法节点
                        graph.add_nodes(len(tech_ids), ntype='tech')
                        graph.nodes['tech'].data['type'] = tech_type_tensor
                        
                        # 添加技法触发边
                        graph.add_edges(
                            torch.arange(len(tech_ids)), 
                            tech_ids, 
                            etype=('tech', 'trigger', 'note')
                        )
                
                # 添加撚指特征
                graph = self.to_dgl_graph(graph)
                
                return graph
                
            except Exception as e:
                print(f"构建图结构时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return self._create_minimal_graph()
                
        except Exception as e:
            print(f"转换过程出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_minimal_graph()
        
    def _create_minimal_graph(self) -> dgl.DGLHeteroGraph:
        """创建一个最小的有效图，用于错误处理时返回
        
        Returns:
            dgl.DGLHeteroGraph: 最小的有效异构图
        """
        # 创建最小图数据结构
        minimal_data = {
            ('note', 'temporal', 'note'): ([0], [0]),  # 自循环
            ('note', 'decorate', 'ornament'): ([0], [0]),  # 基本装饰
            ('tech', 'trigger', 'note'): ([0], [0])  # 基本技法
        }
        
        # 创建最小图
        graph = dgl.heterograph(minimal_data)
        
        # 添加基本特征
        graph.nodes['note'].data['pitch'] = torch.zeros(1, dtype=torch.long)
        graph.nodes['note'].data['duration'] = torch.zeros(1, dtype=torch.long)
        graph.nodes['note'].data['velocity'] = torch.zeros(1, dtype=torch.long)
        graph.nodes['note'].data['position'] = torch.zeros(1, dtype=torch.long)
        graph.nodes['ornament'].data['type'] = torch.zeros(1, dtype=torch.long)
        graph.nodes['tech'].data['type'] = torch.zeros(1, dtype=torch.long)
        
        # 添加撚指特征
        is_nianzhi = torch.zeros(1, dtype=torch.float)
        nianzhi_speed = torch.zeros(1, dtype=torch.float)
        nianzhi_intensity = torch.zeros(1, dtype=torch.float)
        
        graph.nodes['note'].data['is_nianzhi'] = is_nianzhi
        graph.nodes['note'].data['nianzhi_speed'] = nianzhi_speed
        graph.nodes['note'].data['nianzhi_intensity'] = nianzhi_intensity
        
        # 组合为nianzhi特征向量 [是否撚指, 速度变化, 强度模式]
        nianzhi_feature = torch.stack([is_nianzhi, nianzhi_speed, nianzhi_intensity], dim=1)
        graph.nodes['note'].data['nianzhi'] = nianzhi_feature
        
        return graph
        
    def _build_nodes(self, graph_data: dict, data: dict):
        """构建三类核心节点"""
        try:
            # 主音节点（每个音符一个）
            note_count = len(data['token_sequence']) // 4  # 每个音符4个token (pitch, velocity, duration, tech)
            graph_data[('note', 'temporal', 'note')] = ([], [])
            graph_data[('note', 'decorate', 'ornament')] = ([], [])
            graph_data[('tech', 'trigger', 'note')] = ([], [])
            
            # 节点特征
            self.node_features = {
                'note': {
                    'pitch': torch.zeros(note_count, dtype=torch.long),
                    'position': torch.arange(note_count),
                    'is_special': torch.zeros(note_count, dtype=torch.bool),
                    'section': torch.zeros(note_count, dtype=torch.long)
                }
            }
            
            # 技法节点（预定义类型）
            self.node_features['tech'] = {
                'type': torch.tensor([self.tech_types.index(t) for t in self.tech_types])
            }
            
            # 装饰音节点（动态生成候选）
            self._precompute_ornaments(graph_data, data)
            
        except Exception as e:
            print(f"构建节点时出错: {str(e)}")
            # 创建最小的有效节点集
            self._build_minimal_nodes(graph_data)
            
    def _build_minimal_nodes(self, graph_data: dict):
        """构建最小的有效节点集"""
        graph_data[('note', 'temporal', 'note')] = ([], [])
        graph_data[('note', 'decorate', 'ornament')] = ([], [])
        graph_data[('tech', 'trigger', 'note')] = ([], [])
        
        self.node_features = {
            'note': {
                'pitch': torch.zeros(0, dtype=torch.long),
                'position': torch.zeros(0, dtype=torch.long),
                'is_special': torch.zeros(0, dtype=torch.bool),
                'section': torch.zeros(0, dtype=torch.long)
            },
            'tech': {
                'type': torch.tensor([0])  # 只保留一个默认技法类型
            }
        }
    
    def _precompute_ornaments(self, graph_data: dict, data: dict):
        """预生成装饰音候选节点，基于特色音和调式规则"""
        ornaments = []
        mode_mask = data.get('mode_mask', [])  # 从tokenizer获取调式内/外信息
        
        for note_id in range(len(self.node_features['note']['position'])):
            base_pitch = data['token_sequence'][note_id*3]  # 基础音高
            
            # 检查上下大二度是否为特色音
            upper_second = base_pitch + 2
            lower_second = base_pitch - 2
            
            # 边界检查：确保装饰音音高不会为负值
            if lower_second < 0:
                logger.warning(f"下方大二度装饰音音高为负值 ({lower_second})，将跳过该装饰音")
                lower_second = None  # 标记为无效
            
            upper_special = self._is_special_note(upper_second)
            lower_special = self._is_special_note(lower_second) if lower_second is not None else False
            
            # 如果存在特色音装饰音，优先选择
            if upper_special or lower_special:
                if upper_special:
                    ornaments.append({
                        'base_note': note_id,
                        'pitch': upper_second,
                        'weight': 1.0,
                        'type': 'upper_special'
                    })
                if lower_special and lower_second is not None:
                    ornaments.append({
                        'base_note': note_id,
                        'pitch': lower_second,
                        'weight': 1.0,
                        'type': 'lower_special'
                    })
            else:
                # 没有特色音，根据调式内外规则处理
                is_in_mode = mode_mask[note_id] if note_id < len(mode_mask) else False
                
                if is_in_mode:
                    # 调式内音符：五声正音级进优先原则
                    ornaments.append({
                        'base_note': note_id,
                        'pitch': upper_second,
                        'weight': 0.85,  # 上方级进音权重
                        'type': 'upper_modal'
                    })
                    if lower_second is not None:
                        ornaments.append({
                            'base_note': note_id,
                            'pitch': lower_second,
                            'weight': 0.80,  # 下方级进音权重
                            'type': 'lower_modal'
                        })
                else:
                    # 调式外音符：偏音处理策略
                    ornaments.append({
                        'base_note': note_id,
                        'pitch': upper_second,
                        'weight': 0.45,  # 普通上方大二度权重
                        'type': 'upper_non_modal'
                    })
                    if lower_second is not None:
                        ornaments.append({
                            'base_note': note_id,
                            'pitch': lower_second,
                            'weight': 0.40,  # 普通下方大二度权重
                            'type': 'lower_non_modal'
                        })
        
        # 保存装饰音特征
        self.node_features['ornament'] = {
            'pitch': torch.tensor([o['pitch'] for o in ornaments]),
            'weight': torch.tensor([o['weight'] for o in ornaments], dtype=torch.float),
            'type': torch.tensor([self._encode_ornament_type(o['type']) for o in ornaments]),
            'base_note': torch.tensor([o['base_note'] for o in ornaments])
        }
        
        # 更新装饰关系边
        for i, orn in enumerate(ornaments):
            src = orn['base_note']
            graph_data[('note', 'decorate', 'ornament')][0].append(src)
            graph_data[('note', 'decorate', 'ornament')][1].append(i)
    
    def _connect_temporal(self, graph_data: dict):
        """连接时序关系边（note->note）"""
        note_count = len(self.node_features['note']['position'])
        src = list(range(note_count-1))
        dst = list(range(1, note_count))
        graph_data[('note', 'temporal', 'note')] = (src, dst)
    
    def _find_note_at_position(self, graph_data: dict, position: int) -> int:
        """根据token位置找到对应的音符索引
        
        Args:
            graph_data: 图数据
            position: token位置
            
        Returns:
            int: 对应的音符索引，如果没找到则返回None
        """
        # 每个音符占3个token位置 (Pitch, Velocity, Duration)
        note_idx = position // 3
        
        # 确保索引有效
        if 'note' in graph_data['nodes'] and note_idx < len(graph_data['nodes']['note']['ids']):
            return note_idx
        return None
    
    def _connect_techniques(self, graph_data: dict, data: dict):
        """连接技术标记与音符
        
        Args:
            graph_data: 图数据
            data: tokenized 数据
        """
        tech_positions = data.get('tech_positions', [])
        tech_types = data.get('tech_types', [])
        
        # 如果没有技术标记，则跳过
        if not tech_positions or not tech_types:
            return
        
        # 遍历所有技术标记
        for pos, tech_type in zip(tech_positions, tech_types):
            # 确保技术类型是已知的
            if tech_type.lower() not in self.tech_types:
                continue
            
            # 特别处理撚指技术
            if tech_type.lower() == 'nianzhi':
                # 找到对应的音符节点
                note_idx = self._find_note_at_position(graph_data, pos)
                if note_idx is not None:
                    # 添加撚指标记为音符特征，而不是添加新的音符
                    # 这样模型可以学习到撚指是同音高的快速连奏
                    if 'is_nianzhi' not in graph_data['nodes']['note']['features']:
                        note_count = len(graph_data['nodes']['note']['ids'])
                        graph_data['nodes']['note']['features']['is_nianzhi'] = [0.0] * note_count
                    
                    # 设置撚指标记为1.0
                    graph_data['nodes']['note']['features']['is_nianzhi'][note_idx] = 1.0
                    
                    # 添加撚指详细特征 - 用于自监督学习
                    # 1. 增加撚指速度变化特征 - 表示由慢到快的程度
                    if 'nianzhi_speed' not in graph_data['nodes']['note']['features']:
                        graph_data['nodes']['note']['features']['nianzhi_speed'] = [0.0] * note_count
                    graph_data['nodes']['note']['features']['nianzhi_speed'][note_idx] = 0.8  # 0.8表示较明显的加速
                    
                    # 2. 增加撚指强度模式特征 - 表示由强到弱的程度
                    if 'nianzhi_intensity' not in graph_data['nodes']['note']['features']:
                        graph_data['nodes']['note']['features']['nianzhi_intensity'] = [0.0] * note_count
                    graph_data['nodes']['note']['features']['nianzhi_intensity'][note_idx] = 0.7  # 0.7表示明显的强度递减
                    
                    # 3. 记录上下文信息 - 帮助自监督模块学习撚指的上下文特征
                    if 'nianzhi_context' not in graph_data['nodes']['note']['features']:
                        graph_data['nodes']['note']['features']['nianzhi_context'] = [0.0] * note_count
                    
                    # 标记前后几个音符为撚指上下文（但强度较低）
                    context_range = 2  # 前后各2个音符
                    for i in range(max(0, note_idx-context_range), min(note_count, note_idx+context_range+1)):
                        if i != note_idx:  # 不是撚指中心点
                            # 值越小表示离撚指越远
                            distance = abs(i - note_idx)
                            context_value = max(0.0, 0.5 - 0.2 * distance)
                            graph_data['nodes']['note']['features']['nianzhi_context'][i] = context_value
    
    def _mark_special_notes(self, graph: dgl.DGLHeteroGraph, data: dict):
        """标记特色音节点"""
        special_ids = [pos//3 for pos in data.get('special_positions', [])]
        is_special = graph.nodes['note'].data['is_special']
        is_special[special_ids] = True
        graph.nodes['note'].data['is_special'] = is_special
    
    def _mark_liaopai_sections(self, graph: dgl.DGLHeteroGraph, data: dict):
        """标记撩拍段落"""
        note_count = graph.num_nodes('note')
        section_markers = torch.zeros(note_count, dtype=torch.long)
        
        for sect_type, (start, end) in data.get('liaopai_sections', {}).items():
            start_note = start // 3
            end_note = min(end // 3, note_count)
            section_markers[start_note:end_note] = self._section_type_to_id(sect_type)
            
        graph.nodes['note'].data['section'] = section_markers
    
    def _is_special_note(self, pitch: int) -> bool:
        """检查音高是否为特色音"""
        if pitch is None:
            return False
        # 直接检查MIDI音高值是否在特色音集合中
        return pitch in self.special_notes_midi
    
    def _encode_ornament_type(self, type_str: str) -> int:
        """将装饰音类型编码为整数"""
        type_map = {
            'upper_special': 0,
            'lower_special': 1,
            'upper_modal': 2,
            'lower_modal': 3,
            'upper_non_modal': 4,
            'lower_non_modal': 5
        }
        return type_map.get(type_str, 0)
    
    def to_dgl_graph(self, graph: dgl.DGLHeteroGraph) -> dgl.DGLHeteroGraph:
        """将现有DGL图转换成包含组合撚指特征的图
        
        Args:
            graph: 原始DGL图
            
        Returns:
            dgl.DGLHeteroGraph: 包含组合撚指特征的图
        """
        device = graph.device
        if 'note' not in graph.ntypes:
            return graph  # 没有音符节点，直接返回原图
        
        actual_note_count = graph.num_nodes('note')
        if actual_note_count == 0:
            return graph  # 没有音符，直接返回原图
        
        # 初始化撚指特征
        is_nianzhi = torch.zeros(actual_note_count, device=device)
        nianzhi_speed = torch.zeros(actual_note_count, device=device)
        nianzhi_intensity = torch.zeros(actual_note_count, device=device)
        
        # 检查是否有撚指技法
        if 'tech' in graph.ntypes and ('tech', 'trigger', 'note') in graph.canonical_etypes:
            # 获取技法类型和触发的音符
            tech_types = graph.nodes['tech'].data.get('type', None)
            if tech_types is not None:
                # 获取触发边
                tech_src, tech_dst = graph.edges(etype=('tech', 'trigger', 'note'))
                
                # 查找撚指技法
                for i, tech_type in enumerate(tech_types):
                    if tech_type == self.tech_types.index('nianzhi'):
                        # 获取撚指音符索引
                        note_idx = tech_dst[i].item()
                        
                        # 设置撚指标记为1.0
                        is_nianzhi[note_idx] = 1.0
                        
                        # 设置撚指速度和强度特征
                        nianzhi_speed[note_idx] = 0.8  # 0.8表示较明显的加速
                        nianzhi_intensity[note_idx] = 0.7  # 0.7表示明显的强度递减
                        
                        # 标记前后几个音符为撚指上下文
                        context_range = 2
                        for j in range(max(0, note_idx-context_range), min(actual_note_count, note_idx+context_range+1)):
                            if j != note_idx:  # 不是撚指中心点
                                # 值越小表示离撚指越远
                                distance = abs(j - note_idx)
                                context_value = max(0.0, 0.5 - 0.2 * distance)
                                # 上下文特征先不添加，模型目前没有使用
        
        # 添加单独的撚指特征
        graph.nodes['note'].data['is_nianzhi'] = is_nianzhi
        graph.nodes['note'].data['nianzhi_speed'] = nianzhi_speed
        graph.nodes['note'].data['nianzhi_intensity'] = nianzhi_intensity
        
        # 组合为nianzhi特征向量 [是否撚指, 速度变化, 强度模式]
        nianzhi_feature = torch.stack([is_nianzhi, nianzhi_speed, nianzhi_intensity], dim=1)
        graph.nodes['note'].data['nianzhi'] = nianzhi_feature
        
        # 记录统计信息
        nianzhi_count = (is_nianzhi > 0.5).sum().item()
        if nianzhi_count > 0:
            print(f"成功添加 {nianzhi_count} 个撚指特征向量到图中")
        
        return graph