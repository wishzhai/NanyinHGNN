import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from .gatv2 import NanyinGATv2Conv
from .liaopai_engine import LiaopaiEngine
from .instrument_rules.dongxiao import DongxiaoGenerator
from .instrument_rules.erxian import ErxianGenerator
from .decoder import NanyinDecoder
from .contrastive_learning import NanyinContrastiveLearning  # 导入对比学习模块
import dgl
import dgl.function as fn
import torch.nn as nn
import copy

class NanyinLightningModel(pl.LightningModule):
    def __init__(self, config):
        """初始化模型
        
        Args:
            config: 配置字典
        """
        super().__init__()
        # 保存配置
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        
        # 设置默认设备
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化基本参数
        self.hidden_dim = self.model_config.get('hidden_dim', 128)
        self.num_heads = self.model_config.get('num_heads', 4)
        self.gatv2_layers = self.model_config.get('gatv2_layers', 3)
        self.base_pitch = self.model_config.get('base_pitch', 60)
        self.default_tempo = self.model_config.get('default_tempo', 80)
        
        # 初始化dropout参数
        dropout_config = self.model_config.get('dropout', {})
        self.feat_dropout = nn.Dropout(dropout_config.get('feat', 0.2))
        self.attn_dropout = nn.Dropout(dropout_config.get('attn', 0.1))
        self.mode_dropout = nn.Dropout(dropout_config.get('mode', 0.2))
        
        # 初始化训练阶段配置
        self.progressive_config = self.training_config.get('progressive', {})
        self.current_stage = None
        self.enabled_losses = []
        self.enabled_modules = []
        
        # 初始化撩拍引擎(如果配置存在)
        if 'liaopai_templates' in config:
            try:
                from core.liaopai_engine import LiaopaiEngine
        self.liaopai_engine = LiaopaiEngine(config['liaopai_templates'])
                print("成功初始化撩拍引擎")
            except Exception as e:
                print(f"初始化撩拍引擎失败: {str(e)}")
                self.liaopai_engine = None
        else:
            self.liaopai_engine = None
            print("配置中没有liaopai_templates，撩拍引擎未初始化")
        
        # 初始化编码器层
        self._init_encoder_layers()
        
        # 初始化解码器
        self._init_decoders()
        
        # 初始化对比学习模块(如果启用)
        if self.training_config.get('use_contrastive', False):
            self._init_contrastive_module()
            
        # 初始化监控工具
        self._setup_value_logger()
        
        # 注册保存超参数
        self.save_hyperparameters()
        
        # 初始化测试时使用的度量
        self.test_metrics = nn.ModuleDict({
            'mode_accuracy': nn.Linear(self.hidden_dim, 7),  # 7种南音调式
            'ornament_quality': nn.Linear(self.hidden_dim, 1)
        })
    
    def _init_encoder_layers(self):
        """初始化编码器层"""
        self.encoder_layers = nn.ModuleList()
        for _ in range(self.gatv2_layers):
            self.encoder_layers.append(
                NanyinGATv2Conv(
                    in_feats=self.hidden_dim,
                    out_feats=self.hidden_dim,
                    num_heads=self.num_heads,
            etypes=['temporal', 'decorate', 'trigger'],
                    feat_drop=self.feat_dropout,
                    attn_drop=self.attn_dropout
                )
            )
    
    def _init_decoders(self):
        """初始化解码器模块"""
        # 定义各种解码器
        self.main_decoder = nn.ModuleDict({
            'pitch': nn.LSTM(1, self.hidden_dim, num_layers=2, batch_first=True, bidirectional=False),
            'duration': nn.LSTM(1, self.hidden_dim, num_layers=2, batch_first=True, bidirectional=False),
            'velocity': nn.LSTM(1, self.hidden_dim, num_layers=2, batch_first=True, bidirectional=False)
        })
        
        # 解码器输出投影
        self.main_output_projection = nn.ModuleDict({
            'pitch': nn.Linear(self.hidden_dim, 128),  # 音高范围0-127
            'duration': nn.Linear(self.hidden_dim, 1),  # 单值输出
            'velocity': nn.Linear(self.hidden_dim, 1)   # 单值输出
        })
        
        # 装饰音生成器
        self.ornament_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            self.feat_dropout,  # 直接使用已创建的dropout实例
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # 装饰音解码器
        self.ornament_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            self.feat_dropout,
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # 装饰音输出层
        self.ornament_output = nn.ModuleDict({
            'pitch': nn.Linear(self.hidden_dim, 128),
            'duration': nn.Linear(self.hidden_dim, 1),
            'velocity': nn.Linear(self.hidden_dim, 1),
            'position': nn.Linear(self.hidden_dim, 1),
            'style': nn.Linear(self.hidden_dim, len(self.config.get('ornament_styles', [])))
        })
        
        # 节奏结构生成器
        self.rhythm_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            self.feat_dropout,  # 直接使用已创建的dropout实例
            nn.Linear(self.hidden_dim // 2, 8)  # 节奏模式分类
        )
    
    def _init_contrastive_module(self):
        """初始化对比学习模块"""
        contrastive_config = self.config.get('contrastive', {})
        queue_size = contrastive_config.get('queue_size', 8192)
        momentum = contrastive_config.get('momentum', 0.999)
        temperature = contrastive_config.get('temperature', 0.1)
        
        # 创建动量编码器（复制当前编码器）
        momentum_encoder = nn.ModuleList()
        for layer in self.encoder_layers:
            momentum_encoder.append(copy.deepcopy(layer))
        
        # 创建投影头
        projection_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # 初始化队列
        self._init_contrastive_queue(self.hidden_dim, queue_size)
        
        return {
            'momentum_encoder': momentum_encoder,
            'projection_head': projection_head,
            'momentum': momentum,
            'temperature': temperature
        }

    def forward(self, batch):
        """前向传播方法
        
        Args:
            batch: 输入批次(单个图或图列表)
            
        Returns:
            tuple: (处理后的图列表, 节奏结构列表)
        """
        device = self._device
        
        # 将输入批次转换为图列表
        try:
            if isinstance(batch, dgl.DGLHeteroGraph):
                # 如果输入是单个图，转换为列表
                graphs = [batch]
            elif isinstance(batch, list):
                # 如果输入已经是列表，直接使用
                graphs = batch
            else:
                # 尝试解包批次为图列表
                try:
                    graphs = dgl.unbatch(batch)
                except:
                    print(f"警告: 无法将输入批次转换为图列表，输入类型: {type(batch)}")
                    return None, None
            
            # 验证批次中的图是否有效
            if not graphs or len(graphs) == 0:
                print("警告: 空的图列表")
                return None, None
            
            # 初始化结果列表
            processed_graphs = []
            rhythm_structs = []
            
            # 处理每个图
            for idx, graph in enumerate(graphs):
                # 检查图是否有效
                if not isinstance(graph, dgl.DGLHeteroGraph):
                    print(f"警告: 图 {idx} 不是dgl.DGLHeteroGraph类型")
                    processed_graphs.append(None)
                    rhythm_structs.append(None)
                    continue
                
                # 确保图有'note'节点类型
                if 'note' not in graph.ntypes:
                    print(f"警告: 图 {idx} 没有'note'节点类型")
                    processed_graphs.append(None)
                    rhythm_structs.append(None)
                    continue
                
                # 检查节点数量，确保有音符数据
                num_notes = graph.num_nodes('note')
                if num_notes == 0:
                    print(f"警告: 图 {idx} 没有音符节点")
                    processed_graphs.append(None)
                    rhythm_structs.append(None)
                    continue
                
                # 将图移动到正确的设备
                graph = graph.to(device)
                
                # 确保'note'节点有所有必要的特征
                self._ensure_node_features(graph, 'note', {
                    'feat': (4, torch.zeros(num_notes, 4, device=device)),
                    'position': (1, torch.arange(num_notes, device=device)),
                    'pitch': (1, torch.ones(num_notes, device=device) * 60),  # 默认C4
                    'duration': (1, torch.ones(num_notes, device=device) * 480),  # 默认一拍
                    'velocity': (1, torch.ones(num_notes, device=device) * 64)  # 默认中等力度
                })
                
                # 确保其他节点类型也有基本特征
                for ntype in graph.ntypes:
                    if ntype != 'note':
                        num_nodes = graph.num_nodes(ntype)
                        if num_nodes > 0:
                            self._ensure_node_features(graph, ntype, {
                                'feat': (4, torch.zeros(num_nodes, 4, device=device))
                            })
                
                # 处理图，通过编码器层
                processed_graph = graph
                for layer in self.encoder_layers:
                    processed_graph = layer(processed_graph)
                
                # 生成节奏结构
                try:
                    rhythm_struct = self._generate_rhythm_structure(processed_graph)
                except Exception as e:
                    print(f"生成节奏结构时出错: {str(e)}")
                    rhythm_struct = {'positions': graph.nodes['note'].data['position']}
                
                # 添加到结果列表
                processed_graphs.append(processed_graph)
                rhythm_structs.append(rhythm_struct)
        
        return processed_graphs, rhythm_structs

        except Exception as e:
            print(f"前向传播时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _ensure_node_features(self, graph, ntype, feature_specs):
        """确保图的节点具有所有必需的特征
        
        Args:
            graph: 需要检查的图
            ntype: 节点类型
            feature_specs: 特征规范字典，格式为 {特征名: (维度, 默认值)}
        """
        if ntype not in graph.ntypes:
            return
            
        num_nodes = graph.num_nodes(ntype)
        if num_nodes == 0:
            return
            
        # 检查每个指定的特征
        missing_features = []
        for feat_name, (dim, default_value) in feature_specs.items():
            if feat_name not in graph.nodes[ntype].data:
                graph.nodes[ntype].data[feat_name] = default_value
                missing_features.append(feat_name)
            elif graph.nodes[ntype].data[feat_name].dim() == 1 and dim > 1:
                # 如果特征是一维的，但需要多维，则进行扩展
                orig_feat = graph.nodes[ntype].data[feat_name]
                expanded_feat = orig_feat.unsqueeze(1).expand(num_nodes, dim)
                graph.nodes[ntype].data[feat_name] = expanded_feat
        
        if missing_features:
            print(f"警告: '{ntype}'节点缺少必要特征: {missing_features}")
            print("已初始化缺失的特征")

    def _extract_main_notes(self, graph):
        """从图中提取主要音符特征
        
        Args:
            graph: 输入图
            
        Returns:
            node_features: 主要音符的特征向量
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return None
            
        try:
            device = next(self.parameters()).device
            
            # 1. 首先获取所有特征
            all_pitches = []
            all_positions = []
            all_durations = []
            all_velocities = []
            
            # 2. 从图中收集特征
            note_features = graph.nodes['note'].data
            
            if 'pitch' in note_features:
                all_pitches = note_features['pitch'].tolist()
            if 'position' in note_features:
                all_positions = note_features['position'].tolist()
            if 'duration' in note_features:
                all_durations = note_features['duration'].tolist()
            if 'velocity' in note_features:
                all_velocities = note_features['velocity'].tolist()
            
            # 3. 转换为张量
            if all_pitches:  # 确保有数据
                pitches = torch.tensor(all_pitches, dtype=torch.long, device=device)
                positions = torch.tensor(all_positions, dtype=torch.float, device=device)
                durations = torch.tensor(all_durations, dtype=torch.float, device=device)
                velocities = torch.tensor(all_velocities, dtype=torch.float, device=device)
                
                # 4. 返回组合特征
                return {
                    'pitch': pitches,
                    'position': positions,
                    'duration': durations,
                    'velocity': velocities
                }
            else:
                return None
                
        except Exception as e:
            print(f"提取主要音符特征时出错: {str(e)}")
            return None
            
    def _add_ornaments_to_graph(self, graph, ornament_data):
        """向图中添加装饰音
        
        Args:
            graph: 输入图
            ornament_data: 装饰音数据
        """
        try:
            device = next(self.parameters()).device
            
            # 从装饰音数据中获取特征
            all_pitches = ornament_data.get('pitches', [])
            all_positions = ornament_data.get('positions', [])
            all_durations = ornament_data.get('durations', [])
            all_velocities = ornament_data.get('velocities', [])
            all_base_notes = ornament_data.get('base_notes', [])
            all_instruments = ornament_data.get('instruments', [])
            
            # 转换为张量并添加节点
            if all_pitches:  # 只在有数据时创建节点
                # 移除现有节点
                if 'ornament' in graph.ntypes and graph.num_nodes('ornament') > 0:
                    graph.remove_nodes(torch.arange(graph.num_nodes('ornament'), device=device), ntype='ornament')
                
                # 创建装饰音节点
                num_ornaments = len(all_pitches)
                graph.add_nodes(num_ornaments, ntype='ornament')
                
                # 添加装饰音特征
                graph.nodes['ornament'].data['pitch'] = torch.tensor(all_pitches, dtype=torch.long, device=device)
                graph.nodes['ornament'].data['position'] = torch.tensor(all_positions, dtype=torch.float, device=device)
                graph.nodes['ornament'].data['duration'] = torch.tensor(all_durations, dtype=torch.float, device=device)
                graph.nodes['ornament'].data['velocity'] = torch.tensor(all_velocities, dtype=torch.float, device=device)
                
                # 添加装饰音到主音符的边
                if all_base_notes:
                    # 创建从主音符到装饰音的边
                    src_nodes = torch.tensor(all_base_notes, dtype=torch.long, device=device)
                    dst_nodes = torch.arange(num_ornaments, device=device)
                    graph.add_edges(src_nodes, dst_nodes, etype=('note', 'decorate', 'ornament'))
                
                # 添加乐器信息（如果有）
                if all_instruments:
                    graph.nodes['ornament'].data['instrument'] = torch.tensor(all_instruments, dtype=torch.long, device=device)
                
        except Exception as e:
            print(f"添加装饰音时出错: {str(e)}")
            # 出错时重置装饰音节点
            try:
                if 'ornament' in graph.ntypes:
                    graph.remove_nodes(torch.arange(graph.num_nodes('ornament'), device=device), ntype='ornament')
            except:
                pass

    def _generate_ornaments(self, main_features, main_to_ornament_idx, ornament_count):
        """
        非自回归生成装饰音，支持两种演奏风格
        Args:
            main_features: 主音特征 [num_main_nodes, hidden_dim]
            main_to_ornament_idx: 主音到装饰音的映射索引
            ornament_count: 装饰音数量
        Returns:
            ornament_features: 生成的装饰音特征字典
        """
        # 获取对应的主音特征
        selected_main_features = main_features[main_to_ornament_idx]
        
        # 通过装饰音解码器生成装饰音特征
        ornament_hidden = self.ornament_decoder(selected_main_features)
        
        # 投影到输出空间
        pitch_logits = self.ornament_output['pitch'](ornament_hidden)
        duration_raw = self.ornament_output['duration'](ornament_hidden).squeeze(-1)
        velocity_raw = self.ornament_output['velocity'](ornament_hidden).squeeze(-1)
        position_raw = self.ornament_output['position'](ornament_hidden).squeeze(-1)
        
        # 生成装饰音风格
        style_logits = self.ornament_output['style'](ornament_hidden)
        style_probs = F.softmax(style_logits, dim=-1)
        style_idx = torch.argmax(style_probs, dim=-1)  # 0:轻短倚音型，1:融合旋律型
        
        # 对音高使用softmax获取概率分布
        pitch_probs = F.softmax(pitch_logits, dim=-1)
        
        # 获取最可能的音高
        pitch = torch.argmax(pitch_probs, dim=-1).float()
        
        # 根据风格调整装饰音参数
        duration = torch.zeros_like(duration_raw)
        velocity = torch.zeros_like(velocity_raw)
        position = torch.zeros_like(position_raw)
        
        # 应用不同风格的参数
        light_mask = (style_idx == 0)  # 轻短倚音型
        melodic_mask = (style_idx == 1)  # 融合旋律型
        
        # 1. 轻短倚音型参数
        if torch.any(light_mask):
            duration[light_mask] = duration_raw[light_mask] * self.ornament_styles['light_appoggiatura']['duration_factor']
            velocity[light_mask] = velocity_raw[light_mask] * self.ornament_styles['light_appoggiatura']['velocity_factor']
            position[light_mask] = position_raw[light_mask] + self.ornament_styles['light_appoggiatura']['position_shift']
            
            # 确保音程符合轻短倚音型要求（大二度或小三度）
            valid_intervals = torch.tensor(self.ornament_styles['light_appoggiatura']['pitch_intervals'], 
                                          device=pitch.device)
            
            # 计算与主音的音程差
            for i, idx in enumerate(main_to_ornament_idx):
                if light_mask[i]:
                    # 获取主音音高
                    main_pitch = pitch[idx]
                    # 计算音程差
                    interval = torch.abs(pitch[i] - main_pitch)
                    # 如果音程不在有效范围内，调整为最接近的有效音程
                    if not torch.any(valid_intervals == interval):
                        # 找到最接近的有效音程
                        closest_interval = valid_intervals[torch.argmin(torch.abs(valid_intervals - interval))]
                        # 调整装饰音音高
                        if pitch[i] > main_pitch:
                            pitch[i] = main_pitch + closest_interval
                        else:
                            pitch[i] = main_pitch - closest_interval
        
        # 2. 融合旋律型参数
        if torch.any(melodic_mask):
            duration[melodic_mask] = duration_raw[melodic_mask] * self.ornament_styles['melodic_integration']['duration_factor']
            velocity[melodic_mask] = velocity_raw[melodic_mask] * self.ornament_styles['melodic_integration']['velocity_factor']
            position[melodic_mask] = position_raw[melodic_mask] + self.ornament_styles['melodic_integration']['position_shift']
            
            # 融合旋律型允许更多的音程变化，不需要额外限制
        
        return {
            'pitch': pitch,
            'duration': duration,
            'velocity': velocity,
            'position': position,
            'style': style_idx  # 返回风格信息，便于后续处理
        }

    def _generate_rhythm_structure(self, graph):
        """从图中生成节奏结构
        
        Args:
            graph (dgl.DGLHeteroGraph): 输入异构图
            
        Returns:
            dict: 包含节奏结构信息的字典，如果无法生成则返回空字典
        """
        try:
            # 检查图是否有效
            if graph is None:
                print("警告: 输入图为None")
                return {}
                
            # 确保note节点存在
            if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
                print("警告: 图中没有note节点")
                return {}
                
            # 确保position属性存在
            if 'position' not in graph.nodes['note'].data:
                print("警告: note节点没有position属性")
                # 创建默认位置
                graph.nodes['note'].data['position'] = torch.arange(
                    graph.num_nodes('note'),
                    dtype=torch.long,
                    device=next(self.parameters()).device
                )
                
            # 提取音符位置
            positions = graph.nodes['note'].data['position']
            
            # 确保是长整型
            positions = positions.long()
            
            # 构建节奏结构
            rhythm_struct = {
                'positions': positions,
                'note_count': graph.num_nodes('note'),
                'max_position': positions.max().item() if positions.numel() > 0 else 0,
                'min_position': positions.min().item() if positions.numel() > 0 else 0,
            }
            
            # 如果有pitch属性，计算额外的统计信息
            if 'pitch' in graph.nodes['note'].data:
                pitches = graph.nodes['note'].data['pitch']
                rhythm_struct['pitch_mean'] = pitches.float().mean().item() if pitches.numel() > 0 else 0
                rhythm_struct['pitch_range'] = (pitches.max() - pitches.min()).item() if pitches.numel() > 0 else 0
            
            # 如果有duration属性，计算节奏统计信息
            if 'duration' in graph.nodes['note'].data:
                durations = graph.nodes['note'].data['duration']
                rhythm_struct['duration_mean'] = durations.float().mean().item() if durations.numel() > 0 else 0
                rhythm_struct['duration_std'] = durations.float().std().item() if durations.numel() > 0 else 0
            
            return rhythm_struct
            
        except Exception as e:
            print(f"生成节奏结构时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    def on_train_epoch_start(self):
        """在每个训练epoch开始时检查和更新训练阶段"""
        current_epoch = self.current_epoch
        
        # 计算当前应该处于哪个阶段
        accumulated_epochs = 0
        for stage in range(1, 5):
            stage_config = self.progressive_config[f'stage{stage}']
            accumulated_epochs += stage_config['epochs']
            if current_epoch < accumulated_epochs:
                if stage != self.current_stage:
                    self.current_stage = stage
                    self._update_stage_config(stage)
                break
    
    def _update_stage_config(self, stage):
        """更新当前阶段的配置"""
        stage_config = self.progressive_config[f'stage{stage}']
        
        # 更新学习率
        for g in self.optimizers().param_groups:
            g['lr'] = stage_config['learning_rate']
        
        # 更新batch size (通过callback处理)
        # 检查stage_config中是否有batch_size字段，如果没有则使用默认值
        if 'batch_size' in stage_config:
            self.trainer.datamodule.batch_size = stage_config['batch_size']
        else:
            # 使用训练配置中的默认batch_size
            default_batch_size = self.config['training']['batch_size']
            print(f"警告: stage{stage}配置中缺少batch_size字段，使用默认值: {default_batch_size}")
            self.trainer.datamodule.batch_size = default_batch_size
        
        # 打印阶段信息
        print(f"\n进入训练阶段 {stage}:")
        print(f"- 启用的模块: {stage_config.get('enabled_modules', ['all'])}")
        print(f"- 启用的损失: {stage_config.get('enabled_losses', ['all'])}")
        print(f"- 监控指标: {stage_config.get('metrics', ['loss'])}")
        print(f"- 学习率: {stage_config['learning_rate']}")
        print(f"- 批次大小: {self.trainer.datamodule.batch_size}\n")
    
    def _is_module_enabled(self, module_name):
        """检查某个模块是否在当前阶段启用"""
        stage_config = self.progressive_config[f'stage{self.current_stage}']
        enabled_modules = stage_config.get('enabled_modules', ['all'])
        return 'all' in enabled_modules or module_name in enabled_modules
    
    def _is_loss_enabled(self, loss_name):
        """检查某个损失是否在当前阶段启用"""
        stage_config = self.progressive_config[f'stage{self.current_stage}']
        enabled_losses = stage_config.get('enabled_losses', ['all'])
        return 'all' in enabled_losses or loss_name in enabled_losses

    def training_step(self, batch, batch_idx):
        """训练步骤
        
        Args:
            batch: 输入批次
            batch_idx: 批次索引
            
        Returns:
            torch.Tensor: 损失值
        """
        try:
            # 确保在每个epoch开始时更新训练阶段
            if batch_idx == 0 and self.current_epoch > 0:
                self._update_training_stage()
                
            # 前向传播
            processed_graphs, rhythm_structs = self.forward(batch)
            
            # 检查结果
            if processed_graphs is None or len(processed_graphs) == 0:
                print("警告: 训练时前向传播返回空结果")
                # 返回一个可训练的零损失
                return torch.zeros(1, device=self._device, requires_grad=True)
            
            # 计算损失
            loss_dict = self._calculate_total_loss(processed_graphs, batch, rhythm_structs)
            
            # 确保损失字典包含总损失
            if 'loss' not in loss_dict or loss_dict['loss'] is None:
                loss_dict['loss'] = torch.zeros(1, device=self._device, requires_grad=True)
            
            # 分别记录各个损失组件 (确保传递张量而非字典)
            for loss_name, loss_value in loss_dict.items():
                if loss_value is not None:
                    # 使用标量值，而不是字典
                    self.log(f'train/{loss_name}', loss_value.detach(), 
                           prog_bar=(loss_name == 'loss'), batch_size=1)
            
            # 返回总损失张量
            return loss_dict['loss']
            
        except Exception as e:
            print(f"训练步骤执行时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回一个可训练的零损失
            return torch.zeros(1, device=self._device, requires_grad=True)

    def _calculate_total_loss(self, processed_graphs, original_batch, rhythm_structs):
        """计算所有损失的总和
        
        Args:
            processed_graphs: 处理后的图列表
            original_batch: 原始批次数据
            rhythm_structs: 节奏结构列表
            
        Returns:
            dict: 包含各种损失和总损失
        """
        # 初始化总损失
        total_loss = 0.0
        loss_dict = {}
        
        try:
            # 计算重建损失
            if self._is_loss_enabled('recon'):
                recon_loss = self._calculate_reconstruction_loss(processed_graphs, original_batch)
                if recon_loss is not None:
                    total_loss += recon_loss
                    loss_dict['recon_loss'] = recon_loss
                    # 明确指定batch_size=1避免推断错误
                    self.log('train/recon_loss', recon_loss, prog_bar=True, batch_size=1)
                
            # 计算规则损失
            if self._is_loss_enabled('rule'):
                rule_loss = self._calculate_rule_loss(processed_graphs, original_batch)
                if rule_loss is not None:
                    total_loss += rule_loss
                    loss_dict['rule_loss'] = rule_loss
                    self.log('train/rule_loss', rule_loss, batch_size=1)
                
            # 计算节奏损失
            if self._is_loss_enabled('rhythm') and rhythm_structs:
        rhythm_loss = self._calculate_rhythm_loss(processed_graphs, rhythm_structs)
                if rhythm_loss is not None:
                    total_loss += rhythm_loss
                    loss_dict['rhythm_loss'] = rhythm_loss
                    self.log('train/rhythm_loss', rhythm_loss, batch_size=1)
                
            # 计算调式感知损失
            if self._is_loss_enabled('mode'):
                mode_loss = self._calculate_mode_aware_loss(processed_graphs)
                if mode_loss is not None:
                    total_loss += mode_loss
                    loss_dict['mode_loss'] = mode_loss
                    self.log('train/mode_loss', mode_loss, batch_size=1)
                
            # 计算对比损失
            if self._is_loss_enabled('contrastive'):
                contrastive_loss = self._calculate_contrastive_loss(processed_graphs, original_batch)
                if contrastive_loss is not None:
                    total_loss += contrastive_loss
                    loss_dict['contrastive_loss'] = contrastive_loss
                    self.log('train/contrastive_loss', contrastive_loss, batch_size=1)
                
            # 计算装饰音损失
            if self._is_loss_enabled('ornament'):
                ornament_loss = self._calculate_ornament_loss(processed_graphs, original_batch)
                if ornament_loss is not None:
                    total_loss += ornament_loss
                    loss_dict['ornament_loss'] = ornament_loss
                    self.log('train/ornament_loss', ornament_loss, batch_size=1)
                
            # 记录总损失
            loss_dict['loss'] = total_loss
            self.log('train/loss', total_loss, prog_bar=True, batch_size=1)
            
        except Exception as e:
            print(f"计算损失时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 出错时返回默认损失
            if not loss_dict:
                loss_dict['loss'] = torch.tensor(5.0, device=self._device)
                self.log('train/loss', loss_dict['loss'], prog_bar=True, batch_size=1)
                
        return loss_dict

    def validation_step(self, batch, batch_idx):
        """验证步骤
        
        Args:
            batch: 批次数据
            batch_idx: 批次索引
            
        Returns:
            dict: 包含验证损失
        """
        try:
            # 前向传播
            processed_graphs, rhythm_structs = self.forward(batch)
            
            # 检查处理结果
            if processed_graphs is None or rhythm_structs is None or len(processed_graphs) == 0:
                print("验证时前向传播返回空结果")
                self.log('val_loss', torch.tensor(10.0, device=self._device), batch_size=1)
                return {'val_loss': 10.0}
            
            try:
                # 计算损失
                total_loss = self._calculate_validation_loss(processed_graphs, batch, rhythm_structs)
                # 明确指定batch_size避免推断错误
                batch_size = 1
                if isinstance(batch, (list, tuple)) and len(batch) > 0:
                    batch_size = len(batch)
                elif hasattr(batch, 'batch_size'):
                    batch_size = batch.batch_size
                elif isinstance(batch, dict) and 'batch_size' in batch:
                    batch_size = batch['batch_size']
                    
                # 记录损失
                self.log('val_loss', total_loss, prog_bar=True, batch_size=batch_size)
                return {'val_loss': total_loss}
            except Exception as e:
                print(f"验证损失计算时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                self.log('val_loss', torch.tensor(8.0, device=self._device), batch_size=1)
                return {'val_loss': 8.0}
        except Exception as e:
            print(f"验证步骤执行时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            self.log('val_loss', torch.tensor(9.0, device=self._device), batch_size=1)
            return {'val_loss': 9.0}
            
    def _calculate_validation_loss(self, processed_graphs, batch, rhythm_structs):
        """计算验证损失
        
        Args:
            processed_graphs: 处理后的图
            batch: 原始批次数据
            rhythm_structs: 节奏结构
            
        Returns:
            torch.Tensor: 总损失
        """
        # 使用与训练相同的损失计算方法，但明确指定batch_size
        total_loss = torch.zeros(1, device=self._device)
        try:
            loss_dict = self._calculate_detailed_losses(processed_graphs, batch)
            if loss_dict:
                # 汇总所有启用的损失
                for loss_name, loss_value in loss_dict.items():
                    if self._is_loss_enabled(loss_name) and loss_value is not None:
                        total_loss += loss_value
                        # 验证时的日志添加batch_size=1
                        self.log(f'val/{loss_name}', loss_value, batch_size=1)
            return total_loss
        except Exception as e:
            print(f"验证损失详细计算出错: {str(e)}")
            return torch.tensor(7.0, device=self._device)

    def on_validation_epoch_end(self):
        """验证epoch结束时的回调函数"""
        # 在这里不需要额外的操作，因为我们已经在validation_step中使用on_epoch=True记录了指标
        pass

    def _calculate_reconstruction_loss(self, processed_graphs, original_batch):
        """计算重建损失
        
        Args:
            processed_graphs: 处理后的图列表
            original_batch: 原始批次数据
            
        Returns:
            torch.Tensor: 重建损失值
        """
        try:
            # 初始化设备和损失
            device = self._device
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 检查输入是否有效
            if processed_graphs is None or len(processed_graphs) == 0:
                print("警告: 重建损失计算收到空图列表")
                return total_loss
                
            # 解包原始批次，获取原始图
            if isinstance(original_batch, dgl.DGLHeteroGraph):
                # 如果是单个图，转换为列表
                original_graphs = [original_batch]
            elif isinstance(original_batch, list):
                # 如果已经是列表，直接使用
                original_graphs = original_batch
            else:
                try:
                    # 尝试解包为列表
                    original_graphs = dgl.unbatch(original_batch)
                except:
                    print("警告: 无法解包原始批次，使用处理后的图作为参考")
                    original_graphs = processed_graphs
            
            # 确保图列表长度匹配
            min_len = min(len(processed_graphs), len(original_graphs))
            if min_len == 0:
                return total_loss
                
            # 计算每个图的重建损失
            for idx in range(min_len):
                proc_g = processed_graphs[idx]
                orig_g = original_graphs[idx]
                
                # 检查图是否有效
                if proc_g is None or orig_g is None:
                    continue
                    
                # 检查是否同为异构图
                if not isinstance(proc_g, dgl.DGLHeteroGraph) or not isinstance(orig_g, dgl.DGLHeteroGraph):
                    continue
                
                # 确保两个图具有'note'节点类型
                if 'note' not in proc_g.ntypes or 'note' not in orig_g.ntypes:
                    continue
                    
                # 提取节点特征
                try:
                    proc_feats = proc_g.nodes['note'].data.get('feat')
                    orig_feats = orig_g.nodes['note'].data.get('feat')
                    
                    # 检查特征是否存在
                    if proc_feats is None or orig_feats is None:
                        continue
                        
                    # 计算节点特征MSE损失
                    # 确保形状相同，可能需要截断或填充
                    min_nodes = min(proc_feats.size(0), orig_feats.size(0))
                    if min_nodes > 0:
                        p_feats = proc_feats[:min_nodes]
                        o_feats = orig_feats[:min_nodes]
                        
                        # 确保特征维度匹配
                        if p_feats.size(-1) != o_feats.size(-1):
                            if p_feats.size(-1) > o_feats.size(-1):
                                # 截断处理特征
                                p_feats = p_feats[:, :o_feats.size(-1)]
                            else:
                                # 截断原始特征
                                o_feats = o_feats[:, :p_feats.size(-1)]
                                
                        # 计算均方误差
                        node_loss = F.mse_loss(p_feats, o_feats)
                        total_loss = total_loss + node_loss
                except Exception as e:
                    print(f"计算节点特征损失时出错: {str(e)}")
                
                # 计算边结构损失 (仅处理存在于两个图中的边类型)
                try:
                    # 获取共同的边类型
                    common_etypes = set(proc_g.canonical_etypes).intersection(orig_g.canonical_etypes)
                    
                    if not common_etypes:
                        # 如果没有共同边类型，跳过边损失计算
                        continue
                        
                    for etype in common_etypes:
                        # 获取边索引
                        proc_src, proc_dst = proc_g.edges(etype=etype)
                        orig_src, orig_dst = orig_g.edges(etype=etype)
                        
                        # 如果边存在，计算边结构的二元交叉熵损失
                        if len(proc_src) > 0 and len(orig_src) > 0:
                            # 使用简化方法:比较边数量的差异
                            edge_count_diff = abs(len(proc_src) - len(orig_src))
                            edge_loss = torch.tensor(edge_count_diff, device=device, dtype=torch.float) * 0.01
                            total_loss = total_loss + edge_loss
                except Exception as e:
                    print(f"计算边结构损失时出错: {str(e)}")
            
            # 返回平均损失
            if min_len > 0:
                total_loss = total_loss / min_len
            
            return total_loss
            
        except Exception as e:
            print(f"重建损失计算出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回默认损失
            return torch.tensor(0.0, device=self._device, requires_grad=True)

    def _calculate_rule_loss(self, processed_graphs, original_graphs):
        """计算规则损失
        Args:
            processed_graphs: 处理后的图列表
            original_graphs: 原始图列表
        Returns:
            torch.Tensor: 规则损失
        """
        device = next(self.parameters()).device
        special_loss = torch.tensor(0.0, device=device, requires_grad=True)
        tech_loss = torch.tensor(0.0, device=device, requires_grad=True)
        num_valid_graphs = 0
        
        for p, o in zip(processed_graphs, original_graphs):
            try:
                # 安全获取特征
                p_special = p.nodes['note'].data.get('special_logits', None)
                o_special = o.nodes['note'].data.get('special_label', None)
                
                # 如果特征不存在，创建默认值
                if p_special is None:
                    p_special = torch.zeros((p.num_nodes('note'), 2), device=device, requires_grad=True)
                if o_special is None:
                    o_special = torch.zeros(o.num_nodes('note'), dtype=torch.long, device=device)
                
                # 获取技术特征
                if ('tech', 'trigger', 'note') in p.canonical_etypes:
                    p_tech = p.edges[('tech', 'trigger', 'note')].data.get('tech_logits', None)
                    o_tech = o.edges[('tech', 'trigger', 'note')].data.get('tech_mask', None)
                    
                    if p_tech is None:
                        num_edges = p.num_edges(('tech', 'trigger', 'note'))
                        p_tech = torch.zeros(num_edges, device=device, requires_grad=True)
                    if o_tech is None:
                        num_edges = o.num_edges(('tech', 'trigger', 'note'))
                        o_tech = torch.zeros(num_edges, device=device)
                else:
                    p_tech = torch.tensor([], device=device)
                    o_tech = torch.tensor([], device=device)
                
                # 计算特殊音符损失
                if p_special.size(0) > 0 and o_special.size(0) > 0:
                    min_len = min(p_special.size(0), o_special.size(0))
                    special_loss = special_loss + F.cross_entropy(
                        p_special[:min_len], 
                        o_special[:min_len],
                        reduction='mean'
                    )
                
                # 计算技术特征损失
                if p_tech.size(0) > 0 and o_tech.size(0) > 0:
                    min_len = min(p_tech.size(0), o_tech.size(0))
                    tech_loss = tech_loss + F.binary_cross_entropy_with_logits(
                        p_tech[:min_len], 
                        o_tech[:min_len],
                        reduction='mean'
                    )
                
                num_valid_graphs += 1
                
            except Exception as e:
                print(f"计算规则损失时出错: {str(e)}")
                continue
        
        # 避免除零错误
        if num_valid_graphs == 0:
            print("警告：没有有效的图用于计算规则损失")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # 计算加权总损失，添加损失缩放
        total_rule_loss = (
            self.rule_weights['special_note'] * special_loss / num_valid_graphs * 0.5 +  # 缩小特殊音符损失的权重
            self.rule_weights['tech_consistency'] * tech_loss / num_valid_graphs * 0.3    # 缩小技术特征损失的权重
        )
        
        return total_rule_loss

    def _calculate_rhythm_loss(self, processed_graphs, rhythm_structs):
        """计算节奏损失
        Args:
            processed_graphs: 处理后的图列表
            rhythm_structs: 节奏结构列表
        Returns:
            torch.Tensor: 节奏损失
        """
        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for graph, rhythm_struct in zip(processed_graphs, rhythm_structs):
            # 跳过无效图
            if not graph or not rhythm_struct:
                continue
                
            try:
                # 获取目标节奏位置
                target_onsets = self._get_target_onsets(rhythm_struct)
                if not target_onsets:
                    continue
                    
                # 获取实际音符位置
                if 'position' in graph.nodes['note'].data:
                    actual_onsets = graph.nodes['note'].data['position']
                else:
                    continue
                
                # 如果位置不是张量，转换为张量
                if not isinstance(actual_onsets, torch.Tensor):
                    actual_onsets = torch.tensor(actual_onsets, device=device, dtype=torch.float32)
                    
                # 转换为相同设备
                target_onsets = target_onsets.to(device)
                
                # 计算与最近目标位置的距离
                # 对每个实际位置，找到最近的目标位置
                actual_expanded = actual_onsets.unsqueeze(1)  # [num_actual, 1]
                target_expanded = target_onsets.unsqueeze(0)  # [1, num_target]
                
                distances = torch.abs(actual_expanded - target_expanded)  # [num_actual, num_target]
                min_distances, _ = torch.min(distances, dim=1)  # [num_actual]
                
                # 避免除零错误
                if len(target_onsets) == 0:
                    rhythm_score = torch.tensor(0.0, device=device)
                else:
                    # 归一化距离
                    rhythm_score = min_distances.sum() / (len(target_onsets) + 1e-8)
                
                # 累加损失
                total_loss = total_loss + rhythm_score
                count += 1
                
            except Exception as e:
                print(f"计算节奏损失时出错: {str(e)}")
                continue
        
        # 避免除零错误
        if count == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / count

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 获取训练配置
        training_config = self.config.get('training', {})
        optimizer_config = self.config.get('optimizer', {})
        
        # 设置默认值
        lr = training_config.get('learning_rate', 0.0005)
        weight_decay = training_config.get('weight_decay', 0.01)
        betas = optimizer_config.get('betas', (0.9, 0.999))
        
        optimizer = AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
        
        # 添加预热策略
        total_steps = self.trainer.estimated_stepping_batches
        warmup_epochs = training_config.get('warmup_epochs', 10)
        max_epochs = training_config.get('max_epochs', 1000)
        warmup_steps = int(warmup_epochs * total_steps / max_epochs)
        
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=warmup_steps/total_steps,
                div_factor=25.0,
                final_div_factor=1000.0
            ),
            'interval': 'step',
            'frequency': 1
        }
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

    def _extract_main_notes(self, graph):
        """从图中提取主要音符特征
        
        Args:
            graph: 输入图
            
        Returns:
            node_features: 主要音符的特征向量
        """
        if 'note' not in graph.ntypes or graph.num_nodes('note') == 0:
            return None
            
        try:
            device = next(self.parameters()).device
            
            # 1. 首先获取所有特征
            all_pitches = []
            all_positions = []
            all_durations = []
            all_velocities = []
            
            # 2. 从图中收集特征
            note_features = graph.nodes['note'].data
            
            if 'pitch' in note_features:
                all_pitches = note_features['pitch'].tolist()
            if 'position' in note_features:
                all_positions = note_features['position'].tolist()
            if 'duration' in note_features:
                all_durations = note_features['duration'].tolist()
            if 'velocity' in note_features:
                all_velocities = note_features['velocity'].tolist()
            
            # 3. 转换为张量
            if all_pitches:  # 确保有数据
                pitches = torch.tensor(all_pitches, dtype=torch.long, device=device)
                positions = torch.tensor(all_positions, dtype=torch.float, device=device)
                durations = torch.tensor(all_durations, dtype=torch.float, device=device)
                velocities = torch.tensor(all_velocities, dtype=torch.float, device=device)
                
                # 4. 返回组合特征
                return {
                    'pitch': pitches,
                    'position': positions,
                    'duration': durations,
                    'velocity': velocities
                }
            else:
                return None
                
        except Exception as e:
            print(f"提取主要音符特征时出错: {str(e)}")
            return None
            
    def _add_ornaments_to_graph(self, graph, ornament_data):
        """向图中添加装饰音
        
        Args:
            graph: 输入图
            ornament_data: 装饰音数据
        """
        try:
            device = next(self.parameters()).device
            
            # 从装饰音数据中获取特征
            all_pitches = ornament_data.get('pitches', [])
            all_positions = ornament_data.get('positions', [])
            all_durations = ornament_data.get('durations', [])
            all_velocities = ornament_data.get('velocities', [])
            all_base_notes = ornament_data.get('base_notes', [])
            all_instruments = ornament_data.get('instruments', [])
            
            # 转换为张量并添加节点
            if all_pitches:  # 只在有数据时创建节点
                # 移除现有节点
                if 'ornament' in graph.ntypes and graph.num_nodes('ornament') > 0:
                    graph.remove_nodes(torch.arange(graph.num_nodes('ornament'), device=device), ntype='ornament')
                
                # 创建装饰音节点
                num_ornaments = len(all_pitches)
                graph.add_nodes(num_ornaments, ntype='ornament')
                
                # 添加装饰音特征
                graph.nodes['ornament'].data['pitch'] = torch.tensor(all_pitches, dtype=torch.long, device=device)
                graph.nodes['ornament'].data['position'] = torch.tensor(all_positions, dtype=torch.float, device=device)
                graph.nodes['ornament'].data['duration'] = torch.tensor(all_durations, dtype=torch.float, device=device)
                graph.nodes['ornament'].data['velocity'] = torch.tensor(all_velocities, dtype=torch.float, device=device)
                
                # 添加装饰音到主音符的边
                if all_base_notes:
                    # 创建从主音符到装饰音的边
                    src_nodes = torch.tensor(all_base_notes, dtype=torch.long, device=device)
                    dst_nodes = torch.arange(num_ornaments, device=device)
                    graph.add_edges(src_nodes, dst_nodes, etype=('note', 'decorate', 'ornament'))
                
                # 添加乐器信息（如果有）
                if all_instruments:
                    graph.nodes['ornament'].data['instrument'] = torch.tensor(all_instruments, dtype=torch.long, device=device)
                
        except Exception as e:
            print(f"添加装饰音时出错: {str(e)}")
            # 出错时重置装饰音节点
            try:
                if 'ornament' in graph.ntypes:
                    graph.remove_nodes(torch.arange(graph.num_nodes('ornament'), device=device), ntype='ornament')
            except:
                pass

    def _get_target_onsets(self, rhythm_struct):
        """从节奏结构中提取目标时间点"""
        onsets = []
        for section in rhythm_struct['sections']:
            beat_seq = section['beat_sequence']
            tempo_curve = section['tempo_curve']
            for i, (beat, tempo) in enumerate(zip(beat_seq, tempo_curve)):
                if beat == 1:
                    time = section['start'] + i * 60 / tempo
                    onsets.append(time)
        return torch.tensor(onsets, device=self._device)

    def _calculate_detailed_losses(self, processed_graphs, original_graphs):
        """计算详细的重建损失"""
        device = next(self.parameters()).device
        pitch_loss = torch.tensor(0.0, device=device)
        position_loss = torch.tensor(0.0, device=device)
        duration_loss = torch.tensor(0.0, device=device)
        
        for p, o in zip(processed_graphs, original_graphs):
            try:
                # 1. 安全获取特征
                p_pitch = p.nodes['note'].data.get('pitch')
                o_pitch = o.nodes['note'].data.get('pitch')
                p_position = p.nodes['note'].data.get('position')
                o_position = o.nodes['note'].data.get('position')
                p_duration = p.nodes['note'].data.get('duration')
                o_duration = o.nodes['note'].data.get('duration')
                
                if all(x is not None for x in [p_pitch, o_pitch, p_position, o_position, p_duration, o_duration]):
                    # 2. 转换为浮点类型
                    p_pitch = p_pitch.float()
                    o_pitch = o_pitch.float()
                    p_position = p_position.float()
                    o_position = o_position.float()
                    p_duration = p_duration.float()
                    o_duration = o_duration.float()
                    
                    # 3. 计算损失
                    pitch_loss += F.mse_loss(p_pitch, o_pitch)
                    position_loss += F.mse_loss(p_position, o_position)
                    duration_loss += F.mse_loss(p_duration, o_duration)
                else:
                    print("警告：某些特征缺失")
                    continue
                
            except Exception as e:
                print(f"计算损失时出错: {str(e)}")
                continue
        
        num_graphs = len(processed_graphs)
        if num_graphs == 0:
            return torch.zeros(3, device=device)
        
        return (pitch_loss/num_graphs, 
                position_loss/num_graphs, 
                duration_loss/num_graphs)

    def _calculate_mode_compliance(self, graphs):
        """计算调式符合度
        
        评估生成的音符是否符合南音调式规则
        """
        device = next(self.parameters()).device
        mode_scores = torch.tensor(0.0, device=device)
        num_valid_graphs = 0
        
        if not graphs:
            return mode_scores
        
        for graph in graphs:
            try:
                # 获取音高和位置信息
                pitches = graph.nodes['note'].data.get('pitch')
                positions = graph.nodes['note'].data.get('position')
                
                if pitches is None or len(pitches) == 0 or positions is None:
                    continue
                
                # 使用第一个音符作为基准音高
                base_pitch = pitches[0].item() if len(pitches) > 0 else 60
                
                # 获取所有可能的南音调式
                mode_scores_per_mode = []
                for mode in ["WUKONG", "SIKONG", "WUKONG_SIYI", "BEISI"]:
                    try:
                        # 获取调式音高集合
                        scale = self._get_nanyin_scale(base_pitch, mode)
                        
                        # 计算在调式内的音符比例
                        in_scale_mask = torch.tensor([p.item() in scale for p in pitches], 
                                                   device=device)
                        
                        if len(in_scale_mask) > 0:
                            # 计算加权得分
                            weights = torch.ones_like(in_scale_mask, dtype=torch.float32)
                            
                            # 位置权重：给小节开始和结束的音符更高的权重
                            positions = positions.float()
                            if len(positions) > 1:
                                # 归一化位置
                                positions = (positions - positions.min()) / (positions.max() - positions.min() + 1e-8)
                                # 计算位置权重
                                pos_weights = (
                                    1.0 +  # 基础权重
                                    torch.exp(-5.0 * torch.abs(positions - 0.0)) +  # 小节开始
                                    torch.exp(-5.0 * torch.abs(positions - 1.0))    # 小节结束
                                )
                                weights = weights * pos_weights
                            
                            # 计算加权平均
                            mode_score = (in_scale_mask.float() * weights).sum() / weights.sum()
                            mode_scores_per_mode.append(mode_score)
                            
                    except Exception as e:
                        print(f"计算{mode}调式符合度时出错: {str(e)}")
                        continue
                
                # 使用最高的调式符合度
                if mode_scores_per_mode:
                    mode_scores += torch.max(torch.stack(mode_scores_per_mode))
                    num_valid_graphs += 1
                
            except Exception as e:
                print(f"计算调式符合度时出错: {str(e)}")
                continue
        
        if num_valid_graphs == 0:
            return mode_scores
            
        return mode_scores / num_valid_graphs

    def _get_nanyin_scale(self, base_pitch: int, mode: str = "WUKONG") -> set:
        """获取南音调式的音高集合
        
        Args:
            base_pitch: 基准音高（整数）
            mode: 调式名称，可选值：WUKONG（五空管）、SIKONG（四空管）、
                 WUKONG_SIYI（五空四仪管）、BEISI（倍四管）
        
        Returns:
            set: 调式音高集合
        """
        # 南音调式定义
        NANYIN_MODES = {
            "WUKONG": {
                "upper": {50, 52, 55, 57, 59, 62, 64, 67, 69, 71},  # d1-b2
                "lower": {38, 40, 43, 45, 48, 50, 52, 55, 57}       # d-a1
            },
            "SIKONG": {
                "scale": {38, 41, 43, 45, 48, 50, 53, 55, 57, 60, 62, 64, 67, 69}  # d-a2
            },
            "WUKONG_SIYI": {
                "scale": {38, 40, 43, 45, 48, 50, 52, 55, 57, 60, 62, 64, 67, 69}  # d-a2
            },
            "BEISI": {
                "scale": {38, 40, 41, 45, 47, 50, 52, 54, 57, 59, 62, 64}  # d-e2
            }
        }
        
        # 获取选定调式的音高集合
        if mode not in NANYIN_MODES:
            print(f"警告：未知的调式 {mode}，使用五空管")
            mode = "WUKONG"
            
        mode_def = NANYIN_MODES[mode]
        if "scale" in mode_def:
            scale = mode_def["scale"]
        else:
            # 对于分上下声区的调式，合并两个声区
            scale = mode_def["upper"] | mode_def["lower"]
            
        # 根据基准音高调整音域
        base_octave = base_pitch // 12
        current_octave = 38 // 12  # 最低音D的八度
        octave_shift = base_octave - current_octave
        
        # 将音高集合平移到正确的八度
        if octave_shift != 0:
            scale = {pitch + (octave_shift * 12) for pitch in scale}
            
        return scale

    def _calculate_ornament_rationality(self, graphs):
        """计算装饰音合理性
        
        评估装饰音的位置和类型是否符合南音规则，并考虑不同的演奏风格
        """
        device = next(self.parameters()).device
        ornament_scores = torch.tensor(0.0, device=device)
        num_valid_graphs = 0
        
        # 验证输入
        if not graphs:
            return ornament_scores
        
        for graph in graphs:
            try:
                # 验证图结构
                if ('ornament' not in graph.ntypes or 
                    'note' not in graph.ntypes or
                    ('note', 'decorate', 'ornament') not in graph.canonical_etypes):
                    continue
                
                # 获取装饰音关系
                src, dst = graph.edges(etype='decorate')
                if len(src) == 0:
                    continue
                
                # 验证所需特征是否存在
                required_features = {
                    'note': ['pitch', 'position', 'duration'],
                    'ornament': ['pitch', 'position']
                }
                
                features_valid = True
                for ntype, features in required_features.items():
                    for feat in features:
                        if feat not in graph.nodes[ntype].data:
                            features_valid = False
                            break
                    if not features_valid:
                        break
                    
                if not features_valid:
                    continue
                
                # 获取主音和装饰音的特征
                main_pitches = graph.nodes['note'].data['pitch'][src]
                orn_pitches = graph.nodes['ornament'].data['pitch'][dst]
                main_positions = graph.nodes['note'].data['position'][src]
                orn_positions = graph.nodes['ornament'].data['position'][dst]
                main_durations = graph.nodes['note'].data['duration'][src]
                
                # 获取装饰音风格（如果存在）
                if 'ornament_style' in graph.nodes['ornament'].data:
                    orn_styles = graph.nodes['ornament'].data['ornament_style'][dst]
                else:
                    # 默认为轻短倚音型(0)
                    orn_styles = torch.zeros_like(orn_pitches, dtype=torch.long, device=device)
                
                # 分离不同风格的装饰音
                light_mask = (orn_styles == 0)  # 轻短倚音型
                melodic_mask = (orn_styles == 1)  # 融合旋律型
                
                # 初始化评分
                style_scores = torch.zeros_like(orn_pitches, dtype=torch.float32, device=device)
                
                # 1. 评估轻短倚音型
                if torch.any(light_mask):
                    # 检查音程关系（大二度或小三度）
                    intervals = torch.abs(orn_pitches[light_mask] - main_pitches[light_mask])
                    valid_intervals = torch.tensor(self.ornament_styles['light_appoggiatura']['pitch_intervals'], device=device)
                    interval_scores = torch.zeros_like(intervals, dtype=torch.float32)
                    
                    # 计算每个音程的得分
                    for i, interval in enumerate(intervals):
                        # 检查是否在有效音程列表中
                        if torch.any(valid_intervals == interval):
                            interval_scores[i] = 1.0
                        else:
                            # 找到最接近的有效音程，计算接近度得分
                            closest_dist = torch.min(torch.abs(valid_intervals - interval)).item()
                            interval_scores[i] = max(0.0, 1.0 - closest_dist / 5.0)  # 距离越远，得分越低
                    
                    # 检查位置合理性（轻短倚音应该靠近主音开始位置）
                    position_diffs = orn_positions[light_mask] - main_positions[light_mask]
                    position_scores = torch.exp(-5.0 * torch.abs(position_diffs - self.ornament_styles['light_appoggiatura']['position_shift']))
                    
                    # 检查持续时间合理性（轻短倚音应该较短）
                    orn_durations = graph.nodes['ornament'].data.get('duration', torch.zeros_like(orn_pitches))[dst]
                    duration_ratios = orn_durations[light_mask] / main_durations[light_mask]
                    duration_scores = torch.exp(-5.0 * torch.abs(duration_ratios - self.ornament_styles['light_appoggiatura']['duration_factor']))
                    
                    # 计算轻短倚音型的综合得分
                    light_scores = (
                        interval_scores * 0.5 +  # 音程关系权重
                        position_scores * 0.3 +  # 位置合理性权重
                        duration_scores * 0.2    # 持续时间合理性权重
                    )
                    
                    # 更新总得分
                    style_scores[light_mask] = light_scores
                
                # 2. 评估融合旋律型
                if torch.any(melodic_mask):
                    # 检查音程关系（更宽松的音程要求）
                    intervals = torch.abs(orn_pitches[melodic_mask] - main_pitches[melodic_mask])
                    valid_intervals = torch.tensor(self.ornament_styles['melodic_integration']['pitch_intervals'], device=device)
                    interval_scores = torch.zeros_like(intervals, dtype=torch.float32)
                    
                    # 计算每个音程的得分
                    for i, interval in enumerate(intervals):
                        # 检查是否在有效音程列表中
                        if torch.any(valid_intervals == interval):
                            interval_scores[i] = 1.0
                        else:
                            # 融合旋律型允许更多变化，但仍有一定限制
                            closest_dist = torch.min(torch.abs(valid_intervals - interval)).item()
                            interval_scores[i] = max(0.0, 1.0 - closest_dist / 7.0)  # 更宽松的评分
                    
                    # 检查位置合理性（融合旋律型应该与主音同时开始）
                    position_diffs = orn_positions[melodic_mask] - main_positions[melodic_mask]
                    position_scores = torch.exp(-3.0 * torch.abs(position_diffs - self.ornament_styles['melodic_integration']['position_shift']))
                    
                    # 检查持续时间合理性（融合旋律型应该较长）
                    orn_durations = graph.nodes['ornament'].data.get('duration', torch.zeros_like(orn_pitches))[dst]
                    duration_ratios = orn_durations[melodic_mask] / main_durations[melodic_mask]
                    duration_scores = torch.exp(-3.0 * torch.abs(duration_ratios - self.ornament_styles['melodic_integration']['duration_factor']))
                    
                    # 计算融合旋律型的综合得分
                    melodic_scores = (
                        interval_scores * 0.4 +  # 音程关系权重
                        position_scores * 0.3 +  # 位置合理性权重
                        duration_scores * 0.3    # 持续时间合理性权重
                    )
                    
                    # 更新总得分
                    style_scores[melodic_mask] = melodic_scores
                
                # 3. 检查装饰音密度
                density_scores = torch.ones_like(style_scores)
                for i in range(len(src)):
                    # 计算当前主音的装饰音数量
                    note_ornaments = (src == src[i]).float().sum()
                    # 如果装饰音过多，降低得分（最多允许3个装饰音）
                    if note_ornaments > 3:
                        density_scores[src == src[i]] *= (3 / note_ornaments)
                
                # 4. 计算综合得分
                if len(style_scores) > 0:
                    # 组合所有规则的得分
                    combined_scores = style_scores * 0.8 + density_scores * 0.2
                    
                    # 计算加权平均分
                    ornament_scores += combined_scores.mean()
                    num_valid_graphs += 1
                
            except Exception as e:
                print(f"计算装饰音合理性时出错: {str(e)}")
                continue
        
        if num_valid_graphs == 0:
            return ornament_scores
            
        return ornament_scores / num_valid_graphs

    def _init_contrastive_queue(self, hidden_dim, queue_size=1024):
        """
        初始化对比学习队列
        Args:
            hidden_dim: 隐藏层维度
            queue_size: 队列大小
        """
        self.register_buffer("queue", torch.randn(queue_size, hidden_dim))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue = F.normalize(self.queue, dim=1)
        
        # 对比学习温度参数
        self.register_buffer("temperature", torch.tensor(0.07))

    def _enqueue_and_dequeue(self, keys):
        """
        更新对比学习队列
        Args:
            keys: 新的特征向量 [batch_size, hidden_dim]
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # 替换队列中的旧特征
        if batch_size <= self.queue.shape[0]:
            self.queue[ptr:ptr + batch_size] = keys
            ptr = (ptr + batch_size) % self.queue.shape[0]  # 循环队列
            self.queue_ptr[0] = ptr

    def _calculate_contrastive_loss(self, processed_graphs, original_graphs):
        """
        计算对比学习损失，增强主音和装饰音之间的关系学习
        Args:
            processed_graphs: 处理后的图
            original_graphs: 原始图
        Returns:
            contrastive_loss: 对比学习损失
        """
        total_loss = 0.0
        num_valid_graphs = 0
        
        for p_graph, o_graph in zip(processed_graphs, original_graphs):
            # 获取节点特征
            node_h = p_graph.nodes['note'].data.get('h')
            if node_h is None:
                continue
                
            # 区分主音和装饰音节点
            is_main = o_graph.nodes['note'].data.get('is_main', torch.ones_like(o_graph.nodes['note'].data['pitch'], dtype=torch.bool))
            
            if torch.any(is_main) and torch.any(~is_main):
                # 获取主音和装饰音特征
                main_features = node_h[is_main]
                ornament_features = node_h[~is_main]
                
                # 归一化特征
                main_features = F.normalize(main_features, dim=1)
                ornament_features = F.normalize(ornament_features, dim=1)
                
                # 计算相似度矩阵
                similarity = torch.matmul(main_features, ornament_features.transpose(0, 1)) / self.temperature
                
                # 获取装饰音对应的主音索引
                main_to_ornament = p_graph.edges['trigger'].data.get('main_to_ornament_idx')
                
                if main_to_ornament is not None:
                    # 创建标签矩阵
                    labels = torch.zeros_like(similarity)
                    for i, idx in enumerate(main_to_ornament):
                        if idx < main_features.size(0):
                            labels[idx, i] = 1.0
                    
                    # 计算交叉熵损失
                    loss = F.cross_entropy(similarity, labels)
                    total_loss += loss
                    
                    # 更新队列
                    self._enqueue_and_dequeue(main_features)
                    
                    num_valid_graphs += 1
        
        # 计算平均损失
        if num_valid_graphs > 0:
            total_loss = total_loss / num_valid_graphs
        
        return total_loss

    def _calculate_mode_aware_loss(self, processed_graphs):
        """计算模式感知损失（使用对比学习）
        
        Args:
            processed_graphs: 处理后的图列表
            
        Returns:
            torch.Tensor: 对比损失
        """
        device = next(self.parameters()).device
        
        # 如果没有启用对比学习，返回零损失
        if not self.config['training'].get('use_contrastive', False):
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        try:
            # 使用对比学习模块计算损失
            contrastive_result = self.contrastive_learning(processed_graphs)
            contrastive_loss = contrastive_result['loss']
            
            # 记录对比学习准确率
            self.log('train/contrastive_acc', contrastive_result['accuracy'], prog_bar=True)
            
            return contrastive_loss
        except Exception as e:
            print(f"计算模式感知损失时出错: {str(e)}")
            # 返回零损失
            return torch.tensor(0.0, device=device, requires_grad=True)

    def _setup_value_logger(self):
        """设置值域检查的日志记录器"""
        return {
            'mode_logits': {
                'min': float('inf'),
                'max': float('-inf'),
                'mean': 0,
                'std': 0,
                'count': 0
            }
        }
        
    def _log_value_checks(self, name, tensor):
        """记录张量的统计信息"""
        with torch.no_grad():
            stats = self.value_check_logger[name]
            stats['min'] = min(stats['min'], tensor.min().item())
            stats['max'] = max(stats['max'], tensor.max().item())
            stats['mean'] = (stats['mean'] * stats['count'] + tensor.mean().item()) / (stats['count'] + 1)
            stats['std'] = tensor.std().item()
            stats['count'] += 1
            
            # 记录到tensorboard
            self.log(f'value_check/{name}_min', stats['min'])
            self.log(f'value_check/{name}_max', stats['max'])
            self.log(f'value_check/{name}_mean', stats['mean'])
            self.log(f'value_check/{name}_std', stats['std'])
            
    def _update_training_stage(self):
        """
        根据当前epoch更新训练阶段
        """
        current_epoch = self.current_epoch
        
        # 检查是否需要更新阶段
        for stage_name, stage_config in sorted(self.progressive_config.items()):
            if current_epoch < stage_config.get('epochs', 0):
                if self.current_stage != stage_name:
                    self.current_stage = stage_name
                    print(f"训练阶段更新为: {stage_name}, 启用损失: {stage_config.get('enabled_losses', [])}")
                    
                    # 更新学习率
                    if 'learning_rate' in stage_config:
                        for optimizer in self.optimizers():
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = stage_config['learning_rate']
                                print(f"学习率更新为: {stage_config['learning_rate']}")
                break

    def _calculate_ornament_loss(self, processed_graphs, original_graphs):
        """计算装饰音损失
        
        Args:
            processed_graphs: 模型处理后的图列表
            original_graphs: 原始图列表
            
        Returns:
            torch.Tensor: 装饰音损失值
        """
        device = self._device
        ornament_losses = []
        
        # 遍历每个图
        for proc_g, orig_g in zip(processed_graphs, original_graphs):
            # 跳过空图
            if proc_g is None or orig_g is None:
                continue
                
            try:
                # 获取装饰音节点
                proc_ornaments = proc_g.nodes['ornament'].data.get('feat', None)
                orig_ornaments = orig_g.nodes['ornament'].data.get('feat', None)
                
                # 如果没有装饰音节点，跳过
                if proc_ornaments is None or orig_ornaments is None:
                    continue
                    
                # 确保特征是浮点类型
                proc_ornaments = proc_ornaments.float()
                orig_ornaments = orig_ornaments.float()
                
                # 计算MSE损失
                if proc_ornaments.size(0) > 0 and orig_ornaments.size(0) > 0:
                    # 确保维度匹配
                    min_size = min(proc_ornaments.size(0), orig_ornaments.size(0))
                    mse_loss = F.mse_loss(
                        proc_ornaments[:min_size], 
                        orig_ornaments[:min_size]
                    )
                    ornament_losses.append(mse_loss)
            except Exception as e:
                print(f"计算装饰音损失时出错: {str(e)}")
                continue
        
        # 如果有有效的损失，计算平均值；否则返回零
        if ornament_losses:
            return torch.mean(torch.stack(ornament_losses))
        else:
            return torch.tensor(0.0, device=device)