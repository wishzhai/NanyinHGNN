import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .simple_gatv2 import SimpleGATv2Model
from .feature_enhancer import MusicFeatureEnhancer
from .rhythm_extractor import RhythmExtractor
from .simple_contrastive import SimpleContrastiveLearning
from .mode_constraints import NanyinModeConstraints
from .simple_rhythm import SimpleRhythmProcessor
import dgl
import logging
import torch.nn as nn
import random
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleNanyinModel(pl.LightningModule):
    """简化版南音模型"""
    
    def __init__(self, config):
        """初始化
        
        Args:
            config: 配置字典
        """
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 设置batch_size属性
        train_config = config.get('train', {})
        self.batch_size = train_config.get('batch_size', 16)
        
        # 特征增强器
        self.feature_enhancer = MusicFeatureEnhancer(config)
        
        # GAT模型
        model_config = config.get('model', {})
        self.gat_model = SimpleGATv2Model(
            in_feats=model_config.get('hidden_dim', 128),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_heads=model_config.get('num_heads', 4),
            num_layers=model_config.get('gatv2_layers', 3),
            dropout=model_config.get('dropout', {}).get('feat', 0.2)
        )
        
        # 节奏提取器
        self.rhythm_extractor = RhythmExtractor(config)
        
        # 简化版节奏处理器
        self.rhythm_processor = SimpleRhythmProcessor(config)
        logger.info("初始化简化版节奏处理器")
        
        # 调式约束模块
        self.mode_constraints = NanyinModeConstraints(config)
        logger.info("初始化调式约束模块")
        
        # 撩拍引擎
        if 'liaopai_templates' in config:
            try:
                from core.liaopai_engine import LiaopaiEngine
                self.liaopai_engine = LiaopaiEngine(config['liaopai_templates'])
                logger.info("成功初始化撩拍引擎")
            except Exception as e:
                logger.error(f"初始化撩拍引擎失败: {str(e)}")
                self.liaopai_engine = None
        else:
            self.liaopai_engine = None
            logger.info("配置中没有liaopai_templates，撩拍引擎未初始化")
        
        # 乐器规则系统
        try:
            from core.instrument_rules.dongxiao import DongxiaoGenerator
            from core.instrument_rules.erxian import ErxianGenerator
            
            self.dongxiao_generator = DongxiaoGenerator(config.get('instrument_rules', {}).get('dongxiao', {}))
            self.erxian_generator = ErxianGenerator(config.get('instrument_rules', {}).get('erxian', {}))
            logger.info("成功初始化乐器规则系统")
        except Exception as e:
            logger.error(f"初始化乐器规则系统失败: {str(e)}")
            self.dongxiao_generator = None
            self.erxian_generator = None
        
        # 对比学习模块
        self.use_contrastive = config.get('contrastive', {}).get('enabled', True)
        if self.use_contrastive:
            self.contrastive = SimpleContrastiveLearning(config)
            logger.info("使用简化版对比学习模块")
            
        # 自回归解码器
        self._init_decoder()
        
        # 损失权重
        self.loss_weights = config.get('train', {}).get('loss_weights', {
            'reconstruction': 1.0,
            'rhythm': 0.1,
            'contrastive': 0.1,
            'mode': 0.2
        })
        
        # 记录训练状态
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        logger.info("初始化SimpleNanyinModel完成")
    
    def _init_decoder(self):
        """初始化自回归解码器"""
        # 设置默认值，无论是否使用解码器
        self.max_decode_length = 512  # 默认最大生成长度
        self.teacher_forcing_ratio = 0.5  # 默认教师强制率
        self.sampling_temperature = 1.0  # 默认采样温度
        
        # 获取解码器配置
        decoder_config = self.config.get('decoder', {})
        model_config = self.config.get('model', {})
        
        # 检查是否使用解码器
        use_decoder = decoder_config.get('use_decoder', True)
        if not use_decoder:
            logger.info("根据配置跳过解码器初始化")
            # 确保必要的属性已设置为默认值（已在方法开始设置）
            self.decoder = None
            self.instrument_rule_processor = None
            return
        
        # 尝试初始化解码器
        try:
            # 获取相关配置
            self.max_decode_length = decoder_config.get('max_length', 512)
            self.teacher_forcing_ratio = decoder_config.get('teacher_forcing', 0.5)
            self.sampling_temperature = decoder_config.get('sampling_temp', 1.0)
            
            # 创建自回归解码器
            self.decoder = NanyinAutoRegressiveDecoder(
                input_dim=model_config.get('hidden_dim', 256),
                feature_dim=5,  # [pitch, duration, velocity, position, is_ornament]
            )
            
            # 初始化乐器规则处理器
            self.instrument_rule_processor = None
            # 根据需要在此处初始化乐器规则处理器
            # 例如: self.instrument_rule_processor = DongxiaoGenerator()
            
            logger.info("解码器初始化成功")
        except Exception as e:
            logger.error(f"解码器初始化失败: {str(e)}")
            # 确保必要的属性有默认值（已在方法开始设置）
            self.decoder = None
            self.instrument_rule_processor = None
    
    def generate(self, seed_graph=None, max_notes=32, temperature=1.0, apply_rhythm=True, apply_mode=True, add_liaopai=True):
        """生成音乐内容
        
        Args:
            seed_graph: 种子图，若为None则创建空图
            max_notes: 最大生成音符数
            temperature: 生成温度，控制随机性
            apply_rhythm: 是否应用节奏模板
            apply_mode: 是否应用调式约束
            add_liaopai: 是否添加撩拍
            
        Returns:
            dgl.DGLHeteroGraph: 生成的图
        """
        device = self.device
        
        try:
            # 创建种子图或使用提供的种子图
            if seed_graph is None:
                logger.info("创建新的空图作为种子...")
                # 创建空图
                seed_graph = dgl.heterograph({('note', 'precedes', 'note'): ([], [])})
                
                # 添加初始音符（使用南音常见的起始音）
                default_pitch = 50  # d1
                default_pos = 0.0
                default_dur = 4.0
                default_vel = 80
                
                # 添加一个节点
                seed_graph.add_nodes(1, ntype='note')
                
                # 设置节点特征
                seed_graph.nodes['note'].data['pitch'] = torch.tensor([default_pitch], device=device)
                seed_graph.nodes['note'].data['position'] = torch.tensor([default_pos], device=device)
                seed_graph.nodes['note'].data['duration'] = torch.tensor([default_dur], device=device)
                seed_graph.nodes['note'].data['velocity'] = torch.tensor([default_vel], device=device)
                
                logger.info(f"创建了初始音符: 音高={default_pitch}, 位置={default_pos}")
            
            # 确保图在正确的设备上
            seed_graph = seed_graph.to(device)
            
            # 随机选择一个节奏模板（如果需要应用节奏模板）
            rhythm_template = "基本四拍"
            if apply_rhythm:
                try:
                    # 根据生成配置选择适当的速度
                    tempo = self.config.get('generation', {}).get('tempo', 'medium')
                    template_name, _ = self.rhythm_processor.get_random_template(tempo)
                    rhythm_template = template_name
                    logger.info(f"选择节奏模板: {rhythm_template}")
                except Exception as e:
                    logger.warning(f"选择节奏模板出错: {str(e)}")
            
            # 迭代生成
            current_graph = seed_graph
            
            for i in range(max_notes):
                logger.info(f"生成第 {i+1}/{max_notes} 个音符...")
                
                # 使用简单扩展方法
                current_graph = self._simple_extend_graph(current_graph, temperature)
                
                # 检查是否成功添加了新音符
                if current_graph.num_nodes('note') <= i + 1:  # +1 是因为种子图已有1个音符
                    logger.warning(f"无法添加新音符，停止生成")
                    break
            
            # 应用节奏模板（如果需要）
            if apply_rhythm:
                try:
                    logger.info(f"应用节奏模板: {rhythm_template}")
                    current_graph = self.rhythm_processor.apply_rhythm_template(current_graph, rhythm_template)
                except Exception as e:
                    logger.error(f"应用节奏模板出错: {str(e)}")
            
            # 应用调式约束（如果需要）
            if apply_mode:
                try:
                    logger.info("应用调式约束...")
                    current_graph = self.mode_constraints.apply_mode_constraints(current_graph)
                except Exception as e:
                    logger.error(f"应用调式约束出错: {str(e)}")
            
            # 添加撩拍（如果需要）
            if add_liaopai:
                try:
                    liaopai_prob = self.config.get('generation', {}).get('liaopai_probability', 0.3)
                    logger.info(f"添加撩拍，概率: {liaopai_prob}")
                    current_graph = self.rhythm_processor.add_liaopai(current_graph, probability=liaopai_prob)
                except Exception as e:
                    logger.error(f"添加撩拍出错: {str(e)}")
            
            # 最终完善
            current_graph = self._finalize_graph(current_graph)
            
            return current_graph
            
        except Exception as e:
            logger.error(f"生成过程中出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回种子图或空图
            return seed_graph if seed_graph is not None else dgl.heterograph({('note', 'precedes', 'note'): ([], [])})
    
    def _finalize_graph(self, graph):
        """完善图，添加时序边和其他必要处理
        
        Args:
            graph: 输入图
            
        Returns:
            dgl.DGLHeteroGraph: 完善后的图
        """
        if 'note' not in graph.ntypes:
            return graph
        
        try:
            # 复制图，避免修改原图
            new_graph = graph.clone()
            num_notes = graph.num_nodes('note')
            
            if num_notes < 2:
                return new_graph
            
            # 获取位置并排序
            if 'position' in graph.nodes['note'].data:
                positions = graph.nodes['note'].data['position'].cpu().numpy()
                sorted_indices = np.argsort(positions).tolist()
                
                # 添加时序边（按位置顺序连接音符）
                src_nodes = sorted_indices[:-1]
                dst_nodes = sorted_indices[1:]
                
                # 创建新边
                new_graph.add_edges(src_nodes, dst_nodes, etype='precedes')
                logger.info(f"添加了 {len(src_nodes)} 条时序边")
            
            return new_graph
            
        except Exception as e:
            logger.error(f"完善图时出错: {str(e)}")
            return graph
    
    def _simple_extend_graph(self, graph, temperature=1.0):
        """简化版扩展图方法，无需解码器，使用简单规则从现有节点生成新节点
        
        Args:
            graph: 当前图
            temperature: 温度参数控制随机性
            
        Returns:
            dgl.DGLHeteroGraph: 扩展后的图
        """
        try:
            logger.info("使用简单规则扩展图...")
            logger.info(f"简化扩展使用设备: {graph.device}")
            
            # 获取设备
            device = graph.device
            
            # 检查图是否为空
            if graph.num_nodes('note') == 0:
                logger.warning("图中没有音符节点，无法扩展")
                return graph
            
            # 获取最后一个音符信息
            n_existing_notes = graph.num_nodes('note')
            
            # 从配置获取生成参数
            gen_config = self.config.get('generation', {})
            min_interval = gen_config.get('min_interval', -5)
            max_interval = gen_config.get('max_interval', 5)
            temperature = gen_config.get('temperature', temperature)
            
            # 获取当前的音符特征
            pitches = graph.nodes['note'].data['pitch']
            positions = graph.nodes['note'].data['position']
            durations = graph.nodes['note'].data['duration']
            velocities = graph.nodes['note'].data['velocity']
            
            # 获取最后一个音符的值
            last_pitch = pitches[-1].item()
            last_pos = positions[-1].item()
            last_dur = durations[-1].item()
            last_vel = velocities[-1].item()
            
            # 检测当前节奏模式
            try:
                rhythm_features = self.rhythm_processor.extract_rhythm_features(graph)
                rhythm_pattern = rhythm_features.get("pattern", "基本四拍")
                rhythm_group = rhythm_features.get("group", "中板")
                logger.info(f"检测到当前节奏模式: {rhythm_pattern}, 组别: {rhythm_group}")
            except Exception as e:
                logger.warning(f"检测节奏模式出错: {str(e)}")
                rhythm_pattern = "基本四拍"
                rhythm_group = "中板"
            
            # 检测当前音符序列的调式
            try:
                mode_results = self.mode_constraints.calculate_mode_compliance(graph)
                current_mode = mode_results.get("best_mode")
                logger.info(f"检测到当前音符序列的调式: {current_mode}, 符合度: {mode_results.get('best_score', 0.0):.4f}")
            except Exception as e:
                logger.warning(f"检测调式出错: {str(e)}")
                current_mode = None
            
            # 生成新的音符属性
            # 1. 音高：在最后一个音符基础上，在一定范围内随机变化
            # 使用温度控制随机性，温度越高变化越大
            intervals = list(range(min_interval, max_interval + 1))
            probabilities = [torch.exp(torch.tensor(-abs(i) / temperature)) for i in intervals]
            probabilities_sum = sum(probabilities)
            probabilities = [p / probabilities_sum for p in probabilities]
            
            # 使用概率分布来随机选择音程
            interval = random.choices(intervals, weights=probabilities, k=1)[0]
            
            # 计算新音高
            new_pitch = last_pitch + interval
            
            # 如果启用调式约束，确保新音高在调式内
            if current_mode:
                try:
                    # 检查新音高是否在调式内，如果不在则修正
                    if new_pitch not in self.mode_constraints.get_mode_scale(current_mode):
                        old_pitch = new_pitch
                        new_pitch = self.mode_constraints.correct_pitch_to_mode(new_pitch, current_mode)
                        logger.info(f"应用调式约束修正音高: {old_pitch} -> {new_pitch} (调式: {current_mode})")
                except Exception as e:
                    logger.warning(f"应用调式约束出错: {str(e)}")
            
            # 2. 位置：基于节奏模式或上一个音符结束后的位置
            # 先计算最后一个音符结束的位置
            last_end = last_pos + last_dur
            
            # 获取当前全局小节位置
            try:
                # 基于当前节奏模式，决定新音符的位置
                bar_unit = self.rhythm_processor.section_length  # 一小节的单位
                beat_unit = self.rhythm_processor.beat_unit  # 一拍的单位
                
                # 计算当前小节位置
                current_bar = int(last_pos // bar_unit)
                position_in_bar = last_pos % bar_unit
                
                # 获取节奏模板
                template = self.rhythm_processor.get_template(rhythm_pattern)
                
                # 找出模板中下一个位置
                next_template_pos = None
                for pos in template:
                    if pos > position_in_bar:
                        next_template_pos = pos
                        break
                
                # 如果找不到下一个位置，则进入下一小节
                if next_template_pos is None:
                    next_template_pos = template[0]
                    current_bar += 1
                
                # 计算新位置
                new_pos = current_bar * bar_unit + next_template_pos
                
                # 确保新位置在上一个音符结束后
                if new_pos < last_end:
                    new_pos = last_end + beat_unit * 0.25
                
                logger.info(f"使用节奏模板决定新音符位置: {new_pos} (模板: {rhythm_pattern})")
            except Exception as e:
                logger.warning(f"应用节奏模板出错: {str(e)}, 使用默认方式设置位置")
                # 默认位置设置：在上一个音符结束后加1/4拍
                new_pos = last_end + beat_unit * 0.25
            
            # 3. 持续时间：在常见时值中随机选择
            common_durations = [120, 240, 360, 480, 720]  # 十六分音符、八分音符、点八分音符、四分音符、点四分音符
            duration_probs = torch.ones(len(common_durations)) / len(common_durations)
            dur_idx = torch.multinomial(duration_probs, 1).item()
            new_dur = common_durations[dur_idx]
            
            # 4. 力度：在上一个音符基础上小幅度变化
            vel_change = torch.randint(-10, 11, (1,)).item()
            new_vel = max(30, min(120, last_vel + vel_change))
            
            # 创建新的节点特征
            new_pitches = torch.cat([pitches, torch.tensor([new_pitch], device=device)])
            new_positions = torch.cat([positions, torch.tensor([new_pos], device=device)])
            new_durations = torch.cat([durations, torch.tensor([new_dur], device=device)])
            new_velocities = torch.cat([velocities, torch.tensor([new_vel], device=device)])
            
            # 特征字典
            new_features = {
                'pitch': new_pitches,
                'position': new_positions,
                'duration': new_durations,
                'velocity': new_velocities
            }
            
            # 复制feat特征（如果存在）
            if 'feat' in graph.nodes['note'].data:
                # 获取原特征和最后一个节点的特征
                original_feats = graph.nodes['note'].data['feat']
                last_feat = original_feats[-1]
                
                # 创建新音符的特征，复制最后一个音符的特征
                new_feat = last_feat.clone()
                
                # 如果特征维度足够大，可以修改前几个维度反映新的音符属性
                if new_feat.shape[0] >= 4:
                    # 归一化后设置到特征向量中
                    new_feat[0] = new_pitch / 127.0      # 音高归一化
                    new_feat[1] = new_pos / 10000.0      # 位置归一化（假设最大位置为10000）
                    new_feat[2] = new_dur / 1000.0       # 持续时间归一化
                    new_feat[3] = new_vel / 127.0        # 力度归一化
                
                # 添加到特征字典
                new_features['feat'] = torch.cat([original_feats, new_feat.unsqueeze(0)])
            
            # 构建新图的节点字典
            n_nodes_dict = {ntype: graph.num_nodes(ntype) for ntype in graph.ntypes}
            n_nodes_dict['note'] += 1  # 增加一个音符节点
            
            # 创建边字典
            edge_dict = {}
            for etype in graph.canonical_etypes:
                src_type, rel_type, dst_type = etype
                
                if src_type == 'note' and dst_type == 'note' and rel_type == 'temporal':
                    # 时序连接：添加从最后一个音符到新音符的边
                    if n_existing_notes > 1:
                        # 保留原有边
                        original_src, original_dst = graph.edges(etype=etype)
                        src = torch.cat([original_src, torch.tensor([n_existing_notes-1], device=device)])
                        dst = torch.cat([original_dst, torch.tensor([n_existing_notes], device=device)])
                    else:
                        # 只有一个原始节点，添加一条从0到1的边
                        src = torch.tensor([0], device=device)
                        dst = torch.tensor([1], device=device)
                    
                    edge_dict[etype] = (src, dst)
                else:
                    # 复制其他类型的边
                    src, dst = graph.edges(etype=etype)
                    edge_dict[etype] = (src, dst)
            
            # 构建新图
            new_graph = dgl.heterograph(edge_dict, n_nodes_dict)
            
            # 添加节点特征
            for key, tensor in new_features.items():
                new_graph.nodes['note'].data[key] = tensor
            
            # 复制其他节点类型的特征
            for ntype in graph.ntypes:
                if ntype != 'note':
                    for key in graph.nodes[ntype].data.keys():
                        new_graph.nodes[ntype].data[key] = graph.nodes[ntype].data[key].clone()
            
            # 验证新图大小
            old_size = graph.num_nodes('note')
            new_size = new_graph.num_nodes('note')
            logger.info(f"简化图扩展完成: 添加新音符，音高={new_pitch}，位置={new_pos}，时值={new_dur}")
            logger.info(f"节点数变化: {old_size} -> {new_size}")
            
            if new_size <= old_size:
                logger.warning("没有成功添加新节点！")
            
            # 检查节奏特征
            try:
                # 临时添加新音符进行节奏特征分析
                positions_with_new = positions.cpu().numpy().tolist() + [new_pos]
                new_rhythm_features = self.rhythm_processor.detect_rhythm_pattern(positions_with_new)
                logger.info(f"添加新音符后的节奏模式: {new_rhythm_features.get('pattern')}, 相似度: {new_rhythm_features.get('similarity', 0.0):.4f}")
            except Exception as e:
                logger.warning(f"检查节奏特征出错: {str(e)}")
            
            # 检查添加节点后的调式符合度
            try:
                with torch.no_grad():
                    # 创建临时图进行测试
                    test_graph = new_graph.clone()
                    mode_results = self.mode_constraints.calculate_mode_compliance(test_graph)
                    logger.info(f"扩展后的图调式符合度: {mode_results.get('best_score', 0.0):.4f} (调式: {mode_results.get('best_mode')})")
            except Exception as e:
                logger.warning(f"检查扩展后调式符合度出错: {str(e)}")
            
            return new_graph
            
        except Exception as e:
            logger.error(f"扩展图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return graph
    
    def forward(self, batch):
        """前向传播
        
        Args:
            batch: 输入批次
            
        Returns:
            tuple: (处理后的图, 节奏结构)
        """
        try:
            # 将输入批次转换为图列表
            if isinstance(batch, dgl.DGLHeteroGraph):
                graphs = [batch]
            elif isinstance(batch, list):
                graphs = batch
            else:
                raise ValueError(f"不支持的输入类型: {type(batch)}")
            
            # 特征增强
            enhanced_graphs = self.feature_enhancer(graphs)
            
            # GAT处理
            processed_graphs = []
            for g in enhanced_graphs:
                # 确保图中有tempo特征
                if 'note' in g.ntypes and 'tempo' not in g.nodes['note'].data:
                    g.nodes['note'].data['tempo'] = torch.ones(g.num_nodes('note'), device=g.device) * 80.0
                
                # GAT处理
                processed_g = self.gat_model(g)
                
                # 应用装饰音生成
                processed_g = self._add_ornaments(processed_g)
                
                # 应用撩拍模板（如果启用）
                if self.liaopai_engine is not None:
                    try:
                        total_length = float(g.nodes['note'].data['position'].max())
                        rhythm_struct = self.liaopai_engine.generate_full_structure(
                            total_length=total_length,
                            min_section_length=4.0,  # 最小4拍
                            tempo_range=(60, 120),
                            beat_resolution=0.25
                        )
                        processed_g = self._apply_rhythm_structure(processed_g, rhythm_struct)
                    except Exception as e:
                        logger.warning(f"应用撩拍模板时出错: {str(e)}")
                
                processed_graphs.append(processed_g)
            
            # 提取节奏结构
            rhythm_structs = self.rhythm_extractor.extract_from_batch(processed_graphs)
            
            return processed_graphs, rhythm_structs
            
        except Exception as e:
            logger.error(f"前向传播时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _add_ornaments(self, graph):
        """添加装饰音到图中
        
        Args:
            graph: 输入异构图
            
        Returns:
            dgl.DGLHeteroGraph: 添加装饰音后的图
        """
        try:
            # 暂时禁用装饰音功能，直接返回原图
            logger.info("装饰音功能已暂时禁用，返回原图")
            return graph
            
        except Exception as e:
            logger.error(f"添加装饰音时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return graph
    
    def _add_nianzhi_techniques(self, graph):
        """添加撚指技巧到图结构中
        
        撚指是一种由强到弱、由慢到快连续点挑拨弦的技术，应保持同音高。
        
        Args:
            graph: 原始图结构
        Returns:
            dgl.DGLHeteroGraph: 添加撚指后的图结构
        """
        try:
            logger.info("开始添加撚指技巧...")
            
            # 如果图中节点太少则返回原图
            if graph.num_nodes('note') < 5:
                logger.info("图中音符数量不足，无法添加撚指")
                return graph
            
            # 获取音符特征
            if 'pitch' not in graph.nodes['note'].data:
                logger.warning("图中没有pitch特征，无法添加撚指")
                return graph
            
            device = graph.device
            pitches = graph.nodes['note'].data['pitch']
            positions = graph.nodes['note'].data['position']
            durations = graph.nodes['note'].data['duration']
            velocities = graph.nodes['note'].data['velocity']
            
            # 新的音符特征列表
            new_pitches = []
            new_positions = []
            new_durations = []
            new_velocities = []
            
            # 创建是否为撚指音符的标记
            is_nianzhi = []
            
            # 撚指音区范围 (一般在中高音区的二弦/洞箫上使用)
            min_pitch, max_pitch = 55, 85  # 假设这是适合撚指的音高范围
            
            # 撚指音符的时值和力度调整系数
            nianzhi_duration_ratio = 0.3  # 撚指音符持续时间更短
            nianzhi_velocity_boost = 10   # 撚指音符力度提高10
            
            # 获取配置中的撚指概率
            nianzhi_probability = self.config.get('generation', {}).get('nianzhi_probability', 0.15)
            
            # 逐个检查音符，按一定概率添加撚指
            added_nianzhi_count = 0
            
            # 首先复制所有原始音符
            for i in range(graph.num_nodes('note')):
                pitch = float(pitches[i])
                orig_pos = float(positions[i])
                orig_dur = float(durations[i])
                orig_vel = float(velocities[i])
                
                new_pitches.append(pitch)
                new_positions.append(orig_pos)
                new_durations.append(orig_dur)
                new_velocities.append(orig_vel)
                is_nianzhi.append(0.0)  # 原始音符不是撚指
                
                # 检查此音符是否适合添加撚指
                if min_pitch <= pitch <= max_pitch and orig_dur > 0.5:
                    # 随机决定是否添加撚指
                    if torch.rand(1).item() < nianzhi_probability:
                        # 撚指是相同音高的快速连奏，添加3个相同音高的撚指音符
                        for j in range(3):
                            # 撚指音符的位置由强到弱、由慢到快
                            nianzhi_pos = orig_pos + (orig_dur * 0.1) + (j * orig_dur * 0.15)
                            nianzhi_dur = orig_dur * nianzhi_duration_ratio * (1.0 - j * 0.15)  # 逐渐减少持续时间
                            nianzhi_vel = min(orig_vel + nianzhi_velocity_boost - (j * 5), 127)  # 逐渐减少力度
                            
                            # 添加撚指音符（相同音高）
                            new_pitches.append(float(pitch))
                            new_positions.append(float(nianzhi_pos))
                            new_durations.append(float(nianzhi_dur))
                            new_velocities.append(float(nianzhi_vel))
                            is_nianzhi.append(1.0)  # 标记为撚指音符
                        
                        added_nianzhi_count += 1
                        logger.info(f"添加撚指: 原音高={pitch}, 位置={orig_pos}, 添加3个同音高撚指音符")
            
            # 如果没有添加任何撚指，直接返回原图
            if added_nianzhi_count == 0:
                logger.info("没有添加任何撚指，返回原图")
                return graph
            
            # 创建新的节点数据
            new_pitches = torch.tensor(new_pitches, device=device)
            new_positions = torch.tensor(new_positions, device=device)
            new_durations = torch.tensor(new_durations, device=device)
            new_velocities = torch.tensor(new_velocities, device=device)
            is_nianzhi = torch.tensor(is_nianzhi, device=device)
            
            # 按位置排序
            sorted_indices = torch.argsort(new_positions)
            new_pitches = new_pitches[sorted_indices]
            new_positions = new_positions[sorted_indices] 
            new_durations = new_durations[sorted_indices]
            new_velocities = new_velocities[sorted_indices]
            is_nianzhi = is_nianzhi[sorted_indices]
            
            # 创建新图
            num_nodes = len(new_pitches)
            num_nodes_dict = {'note': num_nodes}
            
            # 创建边
            src_nodes = list(range(num_nodes - 1))
            dst_nodes = list(range(1, num_nodes))
            
            # 创建自循环边（必要的）
            self_src = torch.arange(num_nodes, device=device)
            self_dst = torch.arange(num_nodes, device=device)
            
            # 定义边字典
            edge_dict = {
                ('note', 'self', 'note'): (self_src, self_dst)
            }
            
            # 如果有足够的节点，添加时序边
            if num_nodes > 1:
                edge_dict[('note', 'temporal', 'note')] = (
                    torch.tensor(src_nodes, device=device),
                    torch.tensor(dst_nodes, device=device)
                )
            
            # 创建新图
            new_graph = dgl.heterograph(edge_dict, num_nodes_dict=num_nodes_dict, device=device)
            
            # 添加节点特征
            new_graph.nodes['note'].data['pitch'] = new_pitches
            new_graph.nodes['note'].data['position'] = new_positions
            new_graph.nodes['note'].data['duration'] = new_durations
            new_graph.nodes['note'].data['velocity'] = new_velocities
            new_graph.nodes['note'].data['is_nianzhi'] = is_nianzhi  # 添加撚指标记为特征
            
            # 如果原图有feat特征，也需要复制和扩展
            if 'feat' in graph.nodes['note'].data:
                # 处理基础特征
                orig_feats = graph.nodes['note'].data['feat']
                feat_dim = orig_feats.shape[1]
                
                # 创建新的特征张量
                new_feats = torch.zeros((num_nodes, feat_dim), device=device)
                
                # 复制原始特征并处理撚指特征
                orig_idx = 0
                for i in range(num_nodes):
                    if is_nianzhi[i] < 0.5:  # 不是撚指音符
                        new_feats[i] = orig_feats[orig_idx]
                        orig_idx += 1
                    else:  # 是撚指音符
                        # 为撚指音符设置特殊特征
                        # 找到对应的原始音符并基于其特征调整
                        prev_idx = max(0, i-1)
                        while prev_idx > 0 and is_nianzhi[prev_idx] > 0.5:
                            prev_idx -= 1
                        new_feats[i] = new_feats[prev_idx].clone()
                        # 在特征中标记这是撚指音符
                        if feat_dim > 4:
                            new_feats[i, 4] = 1.0  # 第5个位置标记为撚指
                
                # 将特征添加到新图
                new_graph.nodes['note'].data['feat'] = new_feats
            
            logger.info(f"撚指处理完成: 从 {graph.num_nodes('note')} 个音符扩展到 {new_graph.num_nodes('note')} 个音符")
            return new_graph
            
        except Exception as e:
            logger.error(f"添加撚指技巧时出错: {str(e)}")
            return graph
    
    def _apply_rhythm_structure(self, graph, rhythm_struct):
        """应用节奏结构到图中
        
        Args:
            graph: 输入图
            rhythm_struct: 节奏结构信息
            
        Returns:
            dgl.DGLHeteroGraph: 应用节奏结构后的图
        """
        try:
            # 暂时实现一个简单的版本，仅记录日志并返回原图
            logger.info(f"应用节奏结构: {rhythm_struct.get('tempo', 80)} BPM")
            
            # 如果图中有note节点，更新tempo属性
            if 'note' in graph.ntypes and graph.num_nodes('note') > 0:
                tempo = rhythm_struct.get('tempo', 80)
                graph.nodes['note'].data['tempo'] = torch.ones(graph.num_nodes('note'), 
                                                              device=graph.device) * tempo
            
            return graph
            
        except Exception as e:
            logger.warning(f"应用节奏结构时出错: {str(e)}")
            return graph
    
    def training_step(self, batch, batch_idx):
        """训练步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            torch.Tensor: 损失值
        """
        try:
            batch_g = batch
            loss_dict = {}
            
            # 使用特征增强器
            enhanced_g = self.feature_enhancer(batch_g)
            
            # 计算重建损失
            pred = self.gat_model(enhanced_g)
            loss = F.mse_loss(pred, enhanced_g.nodes['note'].data['feat'])
            loss_dict['reconstruction'] = loss
            
            # 计算节奏损失
            try:
                rhythm_targets = self.rhythm_extractor.extract_from_batch(batch_g)
                rhythm_preds = self.rhythm_extractor.extract_from_batch(enhanced_g)
                
                # 确保有有效的节奏特征
                if rhythm_targets and rhythm_preds:
                    rhythm_loss = self.rhythm_extractor.calculate_rhythm_loss(rhythm_targets, rhythm_preds)
                    # 如果节奏损失是一个有效的tensor并且可以求导，则使用它
                    if isinstance(rhythm_loss, torch.Tensor) and rhythm_loss.requires_grad:
                        loss_dict['rhythm'] = rhythm_loss * self.loss_weights['rhythm']
                    else:
                        # 否则创建一个零损失但可以求导
                        loss_dict['rhythm'] = torch.tensor(0.0, device=loss.device, requires_grad=True)
                else:
                    loss_dict['rhythm'] = torch.tensor(0.0, device=loss.device, requires_grad=True)
            except Exception as e:
                logger.warning(f"计算节奏损失时出错: {str(e)}")
                loss_dict['rhythm'] = torch.tensor(0.0, device=loss.device, requires_grad=True)
            
            # 计算调式符合度损失
            try:
                mode_results = self.mode_constraints.calculate_mode_compliance(batch_g)
                if "best_score" in mode_results:
                    # 调式符合度越高，损失越低
                    mode_loss = 1.0 - mode_results["best_score"]
                    mode_loss = torch.tensor(mode_loss, device=loss.device, requires_grad=True)
                    loss_dict['mode'] = mode_loss * self.loss_weights['mode']
                    
                    # 记录调式符合度
                    self.log('train/mode_compliance', mode_results["best_score"], prog_bar=True)
            except Exception as e:
                logger.warning(f"计算调式符合度损失时出错: {str(e)}")
                loss_dict['mode'] = torch.tensor(0.0, device=loss.device, requires_grad=True)
            
            # 计算对比损失
            if self.use_contrastive:
                try:
                    contrastive_out = self.contrastive(batch_g)
                    contrastive_loss = contrastive_out['loss']
                    
                    # 确保对比损失是一个有效的tensor并且可以求导
                    if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.requires_grad:
                        loss_dict['contrastive'] = contrastive_loss * self.loss_weights['contrastive']
                    else:
                        # 否则创建一个零损失但可以求导
                        loss_dict['contrastive'] = torch.tensor(0.0, device=loss.device, requires_grad=True)
                        
                    # 记录对比学习的准确率
                    self.log('train/contrastive_acc', contrastive_out.get('accuracy', 0.0), prog_bar=True)
                except Exception as e:
                    logger.warning(f"计算对比损失时出错: {str(e)}")
                    loss_dict['contrastive'] = torch.tensor(0.0, device=loss.device, requires_grad=True)
            
            # 计算总损失
            total_loss = loss_dict['reconstruction'] * self.loss_weights['reconstruction']
            
            if 'rhythm' in loss_dict:
                total_loss = total_loss + loss_dict['rhythm']
                
            if 'mode' in loss_dict:
                total_loss = total_loss + loss_dict['mode']
                
            if 'contrastive' in loss_dict:
                total_loss = total_loss + loss_dict['contrastive']
            
            # 记录各项损失
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    self.log(f'train/{k}_loss', v.item(), prog_bar=True)
            
            # 记录总损失
            self.log('train/loss', total_loss.item(), prog_bar=True)
            
            # 保存输出以便在训练结束时计算
            self.training_step_outputs.append(total_loss.item())
            
            return total_loss
            
        except Exception as e:
            logger.error(f"训练步骤执行出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回一个零损失，避免训练中断
            return torch.tensor(0.0, requires_grad=True, device=self.device)
    
    def validation_step(self, batch, batch_idx):
        """验证步骤
        
        Args:
            batch: 输入批次
            batch_idx: 批次索引
            
        Returns:
            dict: 包含验证指标的字典
        """
        try:
            # 前向传播
            processed_graphs, rhythm_structs = self.forward(batch)
            
            # 计算验证损失
            base_loss = torch.tensor(0.0, device=self.device)
            
            # 基本特征重建损失 - 修复: 确保batch和processed_graphs是正确的类型
            if batch is not None and processed_graphs:
                if isinstance(batch, dgl.DGLHeteroGraph) and isinstance(processed_graphs[0], dgl.DGLHeteroGraph):
                    if 'note' in batch.ntypes and 'note' in processed_graphs[0].ntypes:
                        orig_pitch = batch.nodes['note'].data.get('pitch')
                        pred_pitch = processed_graphs[0].nodes['note'].data.get('pitch')
                        
                        if orig_pitch is not None and pred_pitch is not None:
                            pitch_loss = F.mse_loss(pred_pitch.float(), orig_pitch.float())
                            base_loss = base_loss + pitch_loss
                            self.log('val/pitch_loss', pitch_loss.detach(), batch_size=self.batch_size)
            
            # 确保验证损失是有效的可微张量
            if base_loss == 0:
                dummy_param = nn.Parameter(torch.tensor(0.001, device=self.device))
                val_loss = dummy_param * dummy_param  # 确保有计算图
            else:
                val_loss = base_loss
            
            # 记录总验证损失，明确指定batch_size
            self.log('val_loss', val_loss.detach(), prog_bar=True, batch_size=self.batch_size)
            
            # 存储验证输出
            self.validation_step_outputs.append({'val_loss': val_loss.detach()})
            
            return {'val_loss': val_loss}
            
        except Exception as e:
            logger.error(f"验证步骤出错: {str(e)}")
            import traceback
            traceback.print_exc()
            dummy_param = nn.Parameter(torch.tensor(0.1, device=self.device))
            dummy_loss = dummy_param * dummy_param
            self.log('val_loss', dummy_loss.detach(), prog_bar=True, batch_size=self.batch_size)
            return {'val_loss': dummy_loss}
    
    def on_train_epoch_end(self):
        """训练周期结束处理"""
        # 重置保存的输出
        self.training_step_outputs = []
    
    def on_validation_epoch_end(self):
        """验证周期结束处理"""
        # 重置保存的输出
        self.validation_step_outputs = []
    
    def configure_optimizers(self):
        """配置优化器
        
        Returns:
            dict: 包含优化器和学习率调度器的字典
        """
        try:
            # 获取配置
            train_config = self.config.get('train', {})
            lr = train_config.get('learning_rate', 0.001)
            weight_decay = train_config.get('weight_decay', 1e-5)
            
            # 创建优化器
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=lr, 
                weight_decay=weight_decay
            )
            
            # 创建学习率调度器
            scheduler_config = train_config.get('scheduler', {})
            if scheduler_config.get('type') == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=scheduler_config.get('T_max', 100),
                    eta_min=scheduler_config.get('eta_min', 1e-6)
                )
            else:
                # 默认使用余弦退火调度器
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=100,
                    eta_min=1e-6
                )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'monitor': 'val_loss'
                }
            }
            
        except Exception as e:
            logger.error(f"配置优化器时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 使用默认优化器作为回退
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            return optimizer 