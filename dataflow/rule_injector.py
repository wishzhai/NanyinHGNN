import torch
import numpy as np
import dgl
from typing import Dict, Any
import logging
import random
import traceback

logger = logging.getLogger(__name__)

class RuleInjector:
    def __init__(self, config: Dict[str, Any]):
        # 使用配置中的值
        rule_injection = config.get('rule_injection', {})
        self.pentatonic_boost = rule_injection.get('pentatonic_boost', 2.0)
        self.base_decorate_weight = rule_injection.get('base_decorate_weight', 0.8)
        self.upper_ornament_prob = 0.8  # 上方装饰音概率
        
        # 添加装饰音密度参数
        self.ornament_density = rule_injection.get('ornament_density', 0.6)
        self.enable_ornaments = rule_injection.get('enable_ornaments', True)
        
        logger.info(f"规则注入器初始化 - 五声音阶提升: {self.pentatonic_boost}")
        logger.info(f"规则注入器初始化 - 装饰音密度: {self.ornament_density}")
        logger.info(f"规则注入器初始化 - 装饰音生成: {'启用' if self.enable_ornaments else '禁用'}")
        
    def apply(self, graph: dgl.DGLGraph) -> dgl.DGLGraph:
        """注入南音规则到图结构中"""
        logger.info("开始应用规则注入...")
        logger.info(f"规则注入前 - 图节点类型: {graph.ntypes}")
        logger.info(f"规则注入前 - 图边类型: {graph.etypes}")
        
        # 检查并确保所有必要的节点类型和边类型存在
        if not self._ensure_graph_structure(graph):
            logger.warning("图结构不完整，规则注入可能无法完全应用")
        
        # 五声音阶级进增强
        self._enhance_pentatonic(graph)
        
        # 为每个主音符添加装饰音
        if self.enable_ornaments:
            # 计算目标装饰音数量
            num_notes = graph.num_nodes('note')
            target_ornaments = int(num_notes * self.ornament_density)
            logger.info(f"目标装饰音数量: {target_ornaments} (密度: {self.ornament_density})")
            
            # 添加装饰音节点
            self._add_ornaments_to_all_notes(graph, target_ornaments)
        else:
            logger.info("装饰音生成已禁用，跳过添加装饰音步骤")
        
        # 技法处理规则
        self._apply_tech_rules(graph)
        
        logger.info(f"规则注入后 - 图节点类型: {graph.ntypes}")
        logger.info(f"规则注入后 - 图边类型: {graph.etypes}")
        if 'ornament' in graph.ntypes:
            logger.info(f"规则注入后 - 装饰音节点数量: {graph.num_nodes('ornament')}")
        if 'decorate' in graph.etypes:
            logger.info(f"规则注入后 - 装饰音边数量: {graph.num_edges('decorate')}")
        
        return graph
    
    def _ensure_graph_structure(self, graph: dgl.DGLGraph) -> bool:
        """确保图具有所有必要的节点类型和边类型
        
        Args:
            graph: 要检查的图
            
        Returns:
            bool: 是否成功确保图结构完整
        """
        try:
            # 检查必要的节点类型
            required_ntypes = ['note', 'ornament', 'tech']
            for ntype in required_ntypes:
                if ntype not in graph.ntypes:
                    logger.warning(f"图中缺少 {ntype} 节点类型，尝试添加...")
                    graph.add_nodes(0, ntype=ntype)
                    logger.info(f"已添加 {ntype} 节点类型")
            
            # 检查必要的边类型
            required_etypes = [
                ('note', 'temporal', 'note'),
                ('note', 'decorate', 'ornament'),
                ('tech', 'trigger', 'note')
            ]
            
            for src_type, etype_name, dst_type in required_etypes:
                # 检查是否存在此边类型
                if etype_name not in graph.etypes:
                    logger.warning(f"图中缺少 {etype_name} 边类型，尝试添加...")
                    # 为缺少的边类型添加空边
                    graph.add_edges([], [], etype=(src_type, etype_name, dst_type))
                    logger.info(f"已添加 {etype_name} 边类型")
            
            # 确保必要的特征存在于note节点
            if graph.num_nodes('note') > 0:
                required_features = ['pitch', 'duration', 'velocity', 'position']
                for feature in required_features:
                    if feature not in graph.nodes['note'].data:
                        logger.warning(f"音符节点缺少 {feature} 特征")
                        
                        # 添加默认特征值
                        if feature == 'position':
                            default_val = torch.arange(graph.num_nodes('note'), dtype=torch.float) * 4.0
                        elif feature == 'duration':
                            default_val = torch.ones(graph.num_nodes('note'), dtype=torch.float) * 4.0
                        elif feature == 'velocity':
                            default_val = torch.ones(graph.num_nodes('note'), dtype=torch.long) * 80
                        else:  # pitch
                            default_val = torch.ones(graph.num_nodes('note'), dtype=torch.long) * 60
                            
                        graph.nodes['note'].data[feature] = default_val
                        logger.info(f"已为音符节点添加默认 {feature} 特征")
            
            return True
        except Exception as e:
            logger.error(f"确保图结构时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _enhance_pentatonic(self, graph):
        """五声音阶级进增强"""
        # 检查是否存在装饰音边
        if 'decorate' not in graph.etypes or graph.num_edges('decorate') == 0:
            return
            
        # 获取装饰音边的索引
        decorate_edges = graph.edges(etype='decorate')
        
        # 获取边权重
        if 'weight' not in graph.edges['decorate'].data:
            # 确保在与图相同的设备上创建张量
            device = graph.device
            num_edges = graph.num_edges('decorate')
            graph.edges['decorate'].data['weight'] = torch.ones(num_edges, device=device)
            
        # 对于装饰音边，增强其权重
        graph.edges['decorate'].data['weight'] *= self.pentatonic_boost
        
    def _apply_tech_rules(self, graph):
        """应用技法相关规则"""
        # 撚指休止处理
        self._handle_nianzhi(graph)
        
        # 点挑甲组合处理
        self._handle_diantiao(graph)
    
    def _handle_nianzhi(self, graph):
        """处理撚指休止规则"""
        # 检查是否存在trigger边类型或边数量为0
        if 'trigger' not in graph.etypes or graph.num_edges('trigger') == 0:
            return
            
        nianzhi_edges = graph.edges(etype='trigger')
        for src, dst in zip(*nianzhi_edges):
            # 初始化掩码（如果不存在）
            if 'xiao_mask' not in graph.nodes['note'].data:
                # 确保在与图相同的设备上创建张量
                device = graph.device
                num_notes = graph.num_nodes('note')
                graph.nodes['note'].data['xiao_mask'] = torch.ones(num_notes, device=device)
            if 'erxian_mask' not in graph.nodes['note'].data:
                device = graph.device
                num_notes = graph.num_nodes('note')
                graph.nodes['note'].data['erxian_mask'] = torch.ones(num_notes, device=device)
            
            # 抑制洞箫/二弦声部
            if dst < graph.num_nodes('note'):  # 确保目标节点有效
                graph.nodes['note'].data['xiao_mask'][dst] = 0.0
                graph.nodes['note'].data['erxian_mask'][dst] = 0.0
                
                # 添加装饰音结尾 - 不再考虑位置限制
                self._add_final_ornament(graph, dst)
    
    def _handle_diantiao(self, graph):
        """处理点挑甲组合规则"""
        # 检查是否存在tech边类型或边数量为0
        if 'tech' not in graph.etypes or graph.num_edges('tech') == 0:
            return
            
        diantiao_edges = graph.edges(etype='tech')
        for tech_src, note_dst in zip(*diantiao_edges):
            if note_dst < 2 or note_dst >= graph.num_nodes('note'):  # 确保有足够的前导音符且目标节点有效
                continue
                
            # 前两音同高处理
            prev_indices = [note_dst-2, note_dst-1]
            # 确保索引有效
            if min(prev_indices) < 0 or max(prev_indices) >= graph.num_nodes('note'):
                continue
                
            prev_pitches = graph.nodes['note'].data['pitch'][prev_indices]
            if prev_pitches[0] == prev_pitches[1]:
                # 贯式处理：添加前导装饰音
                self._add_prefix_ornament(graph, note_dst)
                
                # 接式处理：添加后倚音
                if not self._has_nianzhi_after(graph, note_dst):
                    self._add_post_ornament(graph, note_dst)
    
    def _add_final_ornament(self, graph, node_id):
        """添加结尾装饰音"""
        try:
            device = graph.device
            
            # 获取主音符的属性
            main_note_pitch = graph.nodes['note'].data['pitch'][node_id].item()
            main_note_pos = graph.nodes['note'].data['position'][node_id].item()
            main_note_duration = graph.nodes['note'].data['duration'][node_id].item()
            main_note_velocity = graph.nodes['note'].data['velocity'][node_id].item()
            
            # 生成装饰音属性
            ornament_pitch = self._generate_ornament_pitch(main_note_pitch)
            ornament_pos = main_note_pos + 0.1  # 结尾装饰音放在主音符后面
            ornament_duration = max(0.1, main_note_duration * 0.2)  # 使用主音符时值的20%
            ornament_velocity = int(main_note_velocity * 0.8)  # 装饰音力度略小
            
            # 准备装饰音数据
            new_data = {
                'pitch': torch.tensor([ornament_pitch], device=device, dtype=torch.long),
                'position': torch.tensor([ornament_pos], device=device, dtype=torch.float),
                'duration': torch.tensor([ornament_duration], device=device, dtype=torch.float),
                'velocity': torch.tensor([ornament_velocity], device=device, dtype=torch.long),
                'type': torch.tensor([1], device=device, dtype=torch.long)  # 1表示后倚音
            }
            
            # 验证数据完整性
            if not all(k in new_data for k in ['pitch', 'position', 'duration', 'velocity', 'type']):
                logger.error(f"装饰音数据不完整: {new_data.keys()}")
                return False
            
            # 添加装饰音节点
            try:
                graph.add_nodes(1, new_data, ntype='ornament')
                new_orn_id = graph.num_nodes('ornament') - 1
                
                # 添加装饰关系边
                graph.add_edges([node_id], [new_orn_id], etype=('note', 'decorate', 'ornament'))
                
                # 添加边权重
                if 'weight' not in graph.edges[('note', 'decorate', 'ornament')].data:
                    num_edges = graph.num_edges(('note', 'decorate', 'ornament'))
                    graph.edges[('note', 'decorate', 'ornament')].data['weight'] = torch.ones(num_edges, device=device)
                
                # 设置微时差扰动
                if 'microshift' not in graph.nodes['ornament'].data:
                    graph.nodes['ornament'].data['microshift'] = torch.zeros(graph.num_nodes('ornament'), device=device)
                graph.nodes['ornament'].data['microshift'][new_orn_id] = float(np.random.choice([0.03, -0.02]))
                
                logger.info(f"成功添加结尾装饰音 - 主音位置: {node_id}, 装饰音ID: {new_orn_id}")
                return True
                
            except Exception as e:
                logger.error(f"添加装饰音节点失败: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"添加结尾装饰音失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _add_prefix_ornament(self, graph, node_id):
        """添加前导装饰音"""
        try:
            device = graph.device
            
            # 获取主音符的属性
            main_note_pitch = graph.nodes['note'].data['pitch'][node_id].item()
            main_note_pos = graph.nodes['note'].data['position'][node_id].item()
            main_note_duration = graph.nodes['note'].data['duration'][node_id].item()
            main_note_velocity = graph.nodes['note'].data['velocity'][node_id].item()
            
            # 生成装饰音属性
            ornament_pitch = self._generate_ornament_pitch(main_note_pitch)
            ornament_pos = max(0, main_note_pos - 0.1)  # 前导装饰音放在主音符前面
            ornament_duration = max(0.1, main_note_duration * 0.3)  # 使用主音符时值的30%
            ornament_velocity = int(main_note_velocity * 0.8)  # 装饰音力度略小
            
            # 准备装饰音数据
            new_data = {
                'pitch': torch.tensor([ornament_pitch], device=device, dtype=torch.long),
                'position': torch.tensor([ornament_pos], device=device, dtype=torch.float),
                'duration': torch.tensor([ornament_duration], device=device, dtype=torch.float),
                'velocity': torch.tensor([ornament_velocity], device=device, dtype=torch.long),
                'type': torch.tensor([0], device=device, dtype=torch.long)  # 0表示前倚音
            }
            
            # 验证数据完整性
            if not all(k in new_data for k in ['pitch', 'position', 'duration', 'velocity', 'type']):
                logger.error(f"装饰音数据不完整: {new_data.keys()}")
                return False
            
            # 添加装饰音节点
            try:
                graph.add_nodes(1, new_data, ntype='ornament')
                new_orn_id = graph.num_nodes('ornament') - 1
                
                # 添加装饰关系边
                graph.add_edges([node_id], [new_orn_id], etype=('note', 'decorate', 'ornament'))
                
                # 添加边权重
                if 'weight' not in graph.edges[('note', 'decorate', 'ornament')].data:
                    num_edges = graph.num_edges(('note', 'decorate', 'ornament'))
                    graph.edges[('note', 'decorate', 'ornament')].data['weight'] = torch.ones(num_edges, device=device)
                
                logger.info(f"成功添加前导装饰音 - 主音位置: {node_id}, 装饰音ID: {new_orn_id}")
                return True
                
            except Exception as e:
                logger.error(f"添加装饰音节点失败: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"添加前导装饰音失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _add_post_ornament(self, graph, node_id):
        """添加后倚音"""
        try:
            # 验证节点ID
            if node_id < 0 or node_id >= graph.num_nodes('note'):
                logger.error(f"无效的节点ID: {node_id}")
                return False
            
            # 验证必要的特征
            required_features = ['pitch', 'position', 'duration', 'velocity']
            for feature in required_features:
                if feature not in graph.nodes['note'].data:
                    logger.error(f"音符节点缺少{feature}特征")
                    return False
            
            # 获取主音符属性
            main_pitch = graph.nodes['note'].data['pitch'][node_id].item()
            main_pos = graph.nodes['note'].data['position'][node_id].item()
            main_duration = graph.nodes['note'].data['duration'][node_id].item()
            main_velocity = graph.nodes['note'].data['velocity'][node_id].item()
            
            # 计算装饰音属性
            ornament_pitch = main_pitch + 2  # 上方大二度
            ornament_pos = main_pos + 0.2
            ornament_duration = max(0.1, main_duration * 0.2)  # 后倚音稍短
            ornament_velocity = int(main_velocity * 0.8)
            
            # 准备装饰音数据
            device = graph.device
            new_data = {
                'pitch': torch.tensor([ornament_pitch], device=device, dtype=torch.long),
                'position': torch.tensor([ornament_pos], device=device, dtype=torch.float),
                'duration': torch.tensor([ornament_duration], device=device, dtype=torch.float),
                'velocity': torch.tensor([ornament_velocity], device=device, dtype=torch.long),
                'type': torch.tensor([2], device=device, dtype=torch.long)  # 2表示后倚音
            }
            
            # 验证数据完整性
            if not all(k in new_data for k in ['pitch', 'position', 'duration', 'velocity', 'type']):
                logger.error(f"装饰音数据不完整: {new_data.keys()}")
                return False
            
            # 添加装饰音节点
            try:
                graph.add_nodes(1, new_data, ntype='ornament')
                new_orn_id = graph.num_nodes('ornament') - 1
                
                # 添加装饰关系边
                graph.add_edges([node_id], [new_orn_id], etype=('note', 'decorate', 'ornament'))
                
                # 添加边权重
                if 'weight' not in graph.edges[('note', 'decorate', 'ornament')].data:
                    num_edges = graph.num_edges(('note', 'decorate', 'ornament'))
                    graph.edges[('note', 'decorate', 'ornament')].data['weight'] = torch.ones(num_edges, device=device)
                
                logger.info(f"成功添加后倚音 - 主音位置: {node_id}, 装饰音ID: {new_orn_id}")
                return True
                
            except Exception as e:
                logger.error(f"添加装饰音节点失败: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"添加后倚音失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _has_nianzhi_after(self, graph, node_id, lookahead=3):
        """检测后续是否存在撚指"""
        # 检查节点ID是否有效
        if node_id < 0 or node_id >= graph.num_nodes('note'):
            return False
            
        # 计算有效的结束索引
        end = min(node_id + lookahead + 1, graph.num_nodes('note'))
        
        # 检查技法特征是否存在
        if 'tech' not in graph.nodes['note'].data:
            return False
            
        # 获取节点范围
        start = node_id + 1
        if start >= end:
            return False
            
        # 检查技法特征中是否包含'nianzhi'
        tech_features = graph.nodes['note'].data['tech'][start:end]
        if isinstance(tech_features[0], str):
            return any('nianzhi' in t for t in tech_features)
        return False

    def _generate_ornament_pitch(self, main_pitch):
        """生成装饰音的音高
        
        Args:
            main_pitch (int): 主音音高
            
        Returns:
            int: 装饰音音高
        """
        try:
            # 上方大二度优先原则
            intervals = [-3, -2, 2, 3]
            weights = [0.05, 0.40, 0.50, 0.05]  # 上方级进音权重更高
            
            # 使用权重随机选择音程
            interval = random.choices(intervals, weights=weights)[0]
            ornament_pitch = main_pitch + interval
            
            # 确保装饰音在合理范围内 (MIDI音高范围：21-108)
            ornament_pitch = max(21, min(108, ornament_pitch))
            
            return ornament_pitch
        
        except Exception as e:
            logger.error(f"生成装饰音音高失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 如果出错，返回主音音高
            return main_pitch

    def _add_ornaments_to_all_notes(self, graph: dgl.DGLGraph, target_ornaments: int) -> int:
        """为所有音符添加装饰音"""
        try:
            device = graph.device
            added_ornaments = 0
            
            # 获取音符总数
            num_notes = graph.num_nodes('note')
            if num_notes == 0:
                logger.warning("图中没有音符节点")
                return 0
            
            # 获取音符特征
            pitches = graph.nodes['note'].data.get('pitch')
            positions = graph.nodes['note'].data.get('position')
            durations = graph.nodes['note'].data.get('duration')
            velocities = graph.nodes['note'].data.get('velocity')
            
            if any(x is None for x in [pitches, positions, durations, velocities]):
                logger.error("缺少必要的音符特征")
                return 0
            
            # 计算每个音符的装饰音概率
            probs = torch.ones(num_notes, device=device)
            
            # 根据音符位置调整概率
            # 1. 乐句开始和结束的音符增加概率
            phrase_len = 16  # 假设每个乐句16拍
            for i in range(num_notes):
                pos = positions[i].item()
                phrase_pos = pos % phrase_len
                if phrase_pos < 2 or phrase_pos > phrase_len - 2:
                    probs[i] *= 1.2
            
            # 2. 相邻音符的音高差大的位置增加概率
            if num_notes > 1:
                pitch_diffs = torch.abs(pitches[1:] - pitches[:-1])
                for i in range(num_notes - 1):
                    if pitch_diffs[i] > 2:  # 大于大二度
                        probs[i] *= 1.3
                        probs[i + 1] *= 1.3
            
            # 3. 时值较长的音符增加概率
            mean_duration = torch.mean(durations)
            for i in range(num_notes):
                if durations[i] > mean_duration:
                    probs[i] *= 1.2
            
            # 归一化概率
            probs = probs / torch.sum(probs)
            
            # 随机选择音符添加装饰音
            selected_indices = torch.multinomial(
                probs, 
                min(target_ornaments, num_notes),
                replacement=False
            )
            
            # 为每个选定的音符创建装饰音
            for note_idx in selected_indices:
                try:
                    # 获取主音符的属性
                    main_note_pitch = graph.nodes['note'].data['pitch'][note_idx].item()
                    main_note_pos = graph.nodes['note'].data['position'][note_idx].item()
                    main_note_duration = graph.nodes['note'].data['duration'][note_idx].item()
                    main_note_velocity = graph.nodes['note'].data['velocity'][note_idx].item()
                    
                    # 生成装饰音属性
                    ornament_pitch = self._generate_ornament_pitch(main_note_pitch)
                    ornament_type = random.randint(0, 1)  # 0=前倚音, 1=后倚音
                    
                    # 计算装饰音位置和持续时间
                    if ornament_type == 0:  # 前倚音
                        ornament_pos = max(0, main_note_pos - 0.1)
                        duration_factor = 0.3
                    else:  # 后倚音
                        ornament_pos = main_note_pos + 0.1
                        duration_factor = 0.2
                    
                    ornament_duration = max(0.1, main_note_duration * duration_factor)
                    ornament_velocity = int(main_note_velocity * 0.8)  # 装饰音力度略小
                    
                    # 准备装饰音数据
                    new_data = {
                        'pitch': torch.tensor([ornament_pitch], device=device, dtype=torch.long),
                        'position': torch.tensor([ornament_pos], device=device, dtype=torch.float),
                        'duration': torch.tensor([ornament_duration], device=device, dtype=torch.float),
                        'velocity': torch.tensor([ornament_velocity], device=device, dtype=torch.long),
                        'type': torch.tensor([ornament_type], device=device, dtype=torch.long)
                    }
                    
                    # 验证数据完整性
                    if not all(k in new_data for k in ['pitch', 'position', 'duration', 'velocity', 'type']):
                        logger.error(f"装饰音数据不完整: {new_data.keys()}")
                        continue
                    
                    # 添加装饰音节点
                    try:
                        graph.add_nodes(1, new_data, ntype='ornament')
                        new_orn_id = graph.num_nodes('ornament') - 1
                        
                        # 添加装饰关系边
                        graph.add_edges([note_idx], [new_orn_id], etype=('note', 'decorate', 'ornament'))
                        
                        # 添加边权重
                        if 'weight' not in graph.edges[('note', 'decorate', 'ornament')].data:
                            num_edges = graph.num_edges(('note', 'decorate', 'ornament'))
                            graph.edges[('note', 'decorate', 'ornament')].data['weight'] = torch.ones(num_edges, device=device)
                        
                        added_ornaments += 1
                        logger.info(f"成功添加装饰音 - 主音位置: {note_idx}, 装饰音ID: {new_orn_id}")
                        
                    except Exception as e:
                        logger.error(f"添加装饰音节点失败: {str(e)}")
                        continue
                    
                except Exception as e:
                    logger.error(f"处理音符 {note_idx} 时出错: {str(e)}")
                    continue
            
            return added_ornaments
            
        except Exception as e:
            logger.error(f"添加装饰音过程失败: {str(e)}")
            logger.error(traceback.format_exc())
            return 0

    def build_graph(self, notes):
        """构建异构图
        
        Args:
            notes: 音符列表
            
        Returns:
            dgl.DGLGraph: 异构图
        """
        # 准备节点特征
        pitches = []
        durations = []
        velocities = []
        positions = []
        is_rests = []
        is_ornaments = []
        
        # 准备边的源节点和目标节点
        temporal_src = []
        temporal_dst = []
        decorate_src = []
        decorate_dst = []
        trigger_src = []
        trigger_dst = []
        
        # 处理每个音符
        for i, note in enumerate(notes):
            # 添加节点特征
            pitches.append(note['pitch'])
            durations.append(note['duration'])
            velocities.append(note.get('velocity', 80))
            positions.append(note['position'])
            is_rests.append(1 if note.get('is_rest', False) else 0)
            is_ornaments.append(1 if note.get('is_ornament', False) else 0)
            
            # 添加时间关系边（与下一个音符）
            if i < len(notes) - 1:
                temporal_src.append(i)
                temporal_dst.append(i + 1)
            
            # 添加装饰音关系边
            if note.get('is_ornament', False):
                main_note_idx = note.get('main_note_idx')
                if main_note_idx is not None:
                    decorate_src.append(main_note_idx)
                    decorate_dst.append(i)
            
            # 添加触发关系边（撚指等技巧）
            if note.get('tech') == 'nianzhi':
                next_note_idx = i + 1
                if next_note_idx < len(notes):
                    trigger_src.append(i)
                    trigger_dst.append(next_note_idx)
        
        # 创建异构图
        graph_data = {
            ('note', 'temporal', 'note'): (torch.tensor(temporal_src), torch.tensor(temporal_dst)),
            ('note', 'decorate', 'note'): (torch.tensor(decorate_src), torch.tensor(decorate_dst)),
            ('note', 'trigger', 'note'): (torch.tensor(trigger_src), torch.tensor(trigger_dst))
        }
        
        graph = dgl.heterograph(graph_data)
        
        # 添加节点特征
        graph.nodes['note'].data['pitch'] = torch.tensor(pitches)
        graph.nodes['note'].data['duration'] = torch.tensor(durations)
        graph.nodes['note'].data['velocity'] = torch.tensor(velocities)
        graph.nodes['note'].data['position'] = torch.tensor(positions)
        graph.nodes['note'].data['is_rest'] = torch.tensor(is_rests)
        graph.nodes['note'].data['is_ornament'] = torch.tensor(is_ornaments)
        
        return graph