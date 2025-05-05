import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import torch
import dgl
import yaml
from .graph_adapter import NanyinGraphAdapter
from .tokenizer import NanyinTok, NanyinTokenizerConfig
from .graph_assembler import NanyinGraphAssembler
from .rule_injector import RuleInjector
import json
from typing import Dict
from pathlib import Path
from tqdm import tqdm
import random
import logging

# 配置日志
logger = logging.getLogger(__name__)

class NanyinDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8, config=None):
        """初始化数据模块
        
        Args:
            data_dir: 数据目录
            batch_size: 批次大小
            config: 配置字典
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.config = config or {}
        self.train_graphs = []
        self.val_graphs = []
        
        # 从配置中获取 num_workers，如果没有则使用默认值
        self.num_workers = self.config.get('data', {}).get('num_workers', min(23, os.cpu_count() or 1))
        
        # 确保目录存在
        os.makedirs(self.data_dir / "processed", exist_ok=True)
        os.makedirs(self.data_dir / "graphs", exist_ok=True)
        
        # 初始化数据处理器
        tokenizer_config = {
            'tokenizer': {
                'beat_res': {(0, 4): 8, (4, 12): 4},
                'num_velocities': 32,
                'num_microtiming_bins': 8,
                'ticks_per_quarter': 480
            }
        }
        self.tokenizer = NanyinTok(tokenizer_config)
        
        # 初始化图适配器
        self.graph_adapter = NanyinGraphAdapter(self.config)
        
        # 加载数据
        self._setup()

    def prepare_data(self):
        """数据预处理（仅执行一次）"""
        if not self.config:
            logger.warning("未提供配置信息，跳过数据预处理")
            return
            
        try:
            # 确保使用正确的目录路径
            raw_dir = self.data_dir
            processed_dir = self.data_dir / "processed"
            graphs_dir = self.data_dir / "graphs"
            
            # 确保目录存在
            os.makedirs(processed_dir, exist_ok=True)
            os.makedirs(graphs_dir, exist_ok=True)
            
            # 初始化tokenizer和adapter
            logger.info("初始化tokenizer...")
            tokenizer_config = {
                'tokenizer': {
                    'beat_res': {(0, 4): 8, (4, 12): 4},
                    'num_velocities': 32,
                    'num_microtiming_bins': 8,
                    'ticks_per_quarter': 480
                }
            }
            tokenizer = NanyinTok(tokenizer_config)
            
            # 检查MIDI文件
            midi_files = list(raw_dir.glob('*.mid')) + list(raw_dir.glob('*.midi'))
            if midi_files:
                logger.info(f"找到 {len(midi_files)} 个MIDI文件，开始处理...")
                tokenizer.process_directory(raw_dir, processed_dir)
            else:
                logger.info(f"在 {raw_dir} 中没有找到MIDI文件，跳过token化处理")
            
            # 转换为图数据
            tok_files = list(processed_dir.glob('*.tok'))
            if tok_files:
                logger.info(f"找到 {len(tok_files)} 个token文件，开始转换为图结构...")
                self.convert_to_graphs(self.graph_adapter, processed_dir, graphs_dir)
            else:
                logger.info(f"在 {processed_dir} 中没有找到.tok文件，跳过图转换")
                return  # 如果没有tok文件，提前返回
                
            # 图结构增强
            metagraph_config = {
                'node_types': ['note', 'ornament', 'tech'],
                'edge_types': ['temporal', 'decorate', 'trigger']
            }
            assembler = NanyinGraphAssembler(metagraph_config)
            
            # 规则注入
            rule_config = {
                'special_notes': {
                    'special_note_boost': self.config.get('model', {}).get('special_notes', {}).get('special_note_boost', 7.0),
                    'pentatonic_boost': self.config.get('rule_injection', {}).get('pentatonic_boost', 3.0)
                },
                'liaopai': self.config.get('rule_injection', {}).get('liaopai', {})
            }
            injector = RuleInjector(rule_config)
            
            # 处理图
            processed_graphs = self.graph_adapter.get_processed_graphs()
            if not processed_graphs:
                logger.warning("没有找到需要处理的图")
                return
                
            logger.info(f"开始处理 {len(processed_graphs)} 个图...")
            for raw_graph in processed_graphs:
                try:
                    enriched_graph = assembler.assemble(raw_graph)
                    final_graph = injector.apply(enriched_graph)
                except Exception as e:
                    logger.error(f"处理图时出错: {str(e)}")
                    continue
                    
        except Exception as e:
            logger.error(f"数据预处理失败: {str(e)}")
            raise

    def _initialize_node_features(self, graph):
        """初始化图节点特征"""
        # 检查是否存在音符节点
        if 'note' not in graph.ntypes:
            return graph
            
        # 获取节点数量
        num_nodes = graph.num_nodes('note')
        if num_nodes == 0:
            return graph
            
        # 确保基本特征存在
        device = graph.device
        hidden_dim = self.config.get('model', {}).get('hidden_dim', 512)
        
        # 初始化基础特征
        if 'feat' not in graph.nodes['note'].data:
            graph.nodes['note'].data['feat'] = torch.randn(num_nodes, hidden_dim, device=device) * 0.02
        
        # 初始化并规范化音高特征
        if 'pitch' not in graph.nodes['note'].data:
            graph.nodes['note'].data['pitch'] = torch.zeros(num_nodes, device=device)
        else:
            # 将音高值映射到0-87范围
            pitches = graph.nodes['note'].data['pitch']
            # MIDI音高通常在21-108范围内，我们将其映射到0-87
            pitches = torch.clamp(pitches - 21, min=0, max=87).long()
            graph.nodes['note'].data['pitch'] = pitches
        
        # 初始化时值特征
        if 'duration' not in graph.nodes['note'].data:
            graph.nodes['note'].data['duration'] = torch.ones(num_nodes, device=device)
        
        # 初始化力度特征
        if 'velocity' not in graph.nodes['note'].data:
            graph.nodes['note'].data['velocity'] = torch.ones(num_nodes, device=device) * 80
        
        # 初始化位置特征
        if 'position' not in graph.nodes['note'].data:
            graph.nodes['note'].data['position'] = torch.linspace(0, 1, num_nodes, device=device)
        
        # 简化的装饰音预测目标
        if 'ornament_target' not in graph.nodes['note'].data:
            ornament_targets = torch.zeros(num_nodes, 2, device=device)  # [是否有装饰音, 位置偏移]
            
            for note_idx in range(num_nodes):
                # 60%概率设置装饰音
                if torch.rand(1).item() < 0.6:
                    ornament_targets[note_idx, 0] = 1.0  # 有装饰音
                    ornament_targets[note_idx, 1] = -0.1  # 固定的位置偏移（装饰音在主音符前）
        
            graph.nodes['note'].data['ornament_target'] = ornament_targets
            
        # 初始化装饰音权重（统一设置为0.8）
        if 'ornament_weight' not in graph.nodes['note'].data:
            graph.nodes['note'].data['ornament_weight'] = torch.ones(num_nodes, device=device) * 0.8
            
        # 记录初始化完成
        logger.info(f"节点特征初始化完成: {num_nodes}个音符节点，特征维度: {hidden_dim}")
        
        return graph
    
    def _safe_collate(self, graphs):
        """安全的批处理函数
        
        Args:
            graphs: 图列表
            
        Returns:
            list: 处理后的图列表
        """
        processed_graphs = []
        
        try:
            for i, g in enumerate(graphs):
                if not isinstance(g, dgl.DGLHeteroGraph):
                    logger.warning(f"跳过第{i}个图：不是DGLHeteroGraph类型")
                    continue
                    
                # 检查图是否为空
                if g.num_nodes('note') == 0:
                    logger.warning(f"跳过第{i}个图：没有音符节点")
                    continue
                    
                # 检查并记录特征状态
                required_features = ['feat', 'pitch', 'duration', 'velocity', 'position', 'ornament_target']
                missing_features = [feat for feat in required_features if feat not in g.nodes['note'].data]
                
                if missing_features:
                    logger.info(f"第{i}个图缺少特征: {missing_features}，尝试初始化...")
                    g = self._initialize_node_features(g)
                    # 再次检查是否所有特征都已添加
                    still_missing = [feat for feat in required_features if feat not in g.nodes['note'].data]
                    if still_missing:
                        logger.warning(f"第{i}个图在初始化后仍然缺少特征: {still_missing}")
                        continue
                
                # 验证特征维度
                num_nodes = g.num_nodes('note')
                invalid_features = []
                for feat in required_features:
                    if feat in g.nodes['note'].data:
                        feat_shape = g.nodes['note'].data[feat].shape
                        if feat_shape[0] != num_nodes:
                            invalid_features.append(f"{feat}({feat_shape})")
                
                if invalid_features:
                    logger.warning(f"第{i}个图的特征维度不匹配: {invalid_features}, 节点数: {num_nodes}")
                    continue
                
                # 确保所有特征都是float类型
                for feat in required_features:
                    if feat in g.nodes['note'].data:
                        g.nodes['note'].data[feat] = g.nodes['note'].data[feat].float()
                
                # 添加到处理后的图列表
                processed_graphs.append(g)
                logger.info(f"成功处理第{i}个图，节点数: {num_nodes}")
            
            # 如果没有有效的图，返回None
            if not processed_graphs:
                logger.warning("批次中没有有效的图")
                return None
            
            # 批处理
            try:
                batched_graph = dgl.batch(processed_graphs)
                logger.info(f"成功创建批次图，包含 {len(processed_graphs)} 个图")
                return batched_graph
            except Exception as e:
                logger.error(f"批处理图时出错: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"_safe_collate处理时出错: {str(e)}")
            return None
    
    def setup(self, stage=None):
        """准备数据集
        
        Args:
            stage: 训练阶段
        """
        try:
            # 加载图数据
            graph_dir = self.data_dir / 'graphs'
            if not graph_dir.exists():
                raise RuntimeError(f"图数据目录不存在: {graph_dir}")
            
            # 读取所有图文件
            graph_files = list(graph_dir.glob('**/*.dgl'))
            if not graph_files:
                raise RuntimeError(f"在 {graph_dir} 中没有找到.dgl文件")
            
            logger.info(f"找到 {len(graph_files)} 个图文件")
            
            # 随机打乱文件列表
            random.shuffle(graph_files)
            
            # 划分训练集和验证集
            split_idx = int(len(graph_files) * 0.8)  # 80%用于训练
            train_files = graph_files[:split_idx]
            val_files = graph_files[split_idx:]
            
            # 加载训练图
            logger.info("加载训练集...")
            self.train_graphs = []
            for file in tqdm(train_files, desc="加载训练图"):
                try:
                    graphs, _ = dgl.load_graphs(str(file))
                    if graphs:
                        graph = graphs[0]
                        graph = self._initialize_node_features(graph)
                        self.train_graphs.append(graph)
                except Exception as e:
                    logger.error(f"加载训练图 {file} 时出错: {str(e)}")
                    continue
            
            # 加载验证图
            logger.info("加载验证集...")
            self.val_graphs = []
            for file in tqdm(val_files, desc="加载验证图"):
                try:
                    graphs, _ = dgl.load_graphs(str(file))
                    if graphs:
                        graph = graphs[0]
                        graph = self._initialize_node_features(graph)
                        self.val_graphs.append(graph)
                except Exception as e:
                    logger.error(f"加载验证图 {file} 时出错: {str(e)}")
                    continue
            
            logger.info(f"数据集加载完成: 总样本数={len(graph_files)}, "
                       f"训练样本数={len(self.train_graphs)}, "
                       f"验证样本数={len(self.val_graphs)}")
                       
            if not self.train_graphs or not self.val_graphs:
                raise RuntimeError("训练集或验证集为空")
                
        except Exception as e:
            logger.error(f"setup阶段出错: {str(e)}")
            raise

    def train_dataloader(self):
        """返回训练数据加载器"""
        # 过滤掉空的图
        valid_graphs = [g for g in self.train_graphs if g.num_nodes('note') > 0]
        if not valid_graphs:
            raise RuntimeError("没有有效的训练数据")
            
        return DataLoader(
            valid_graphs,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._safe_collate,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """返回验证数据加载器"""
        # 过滤掉空的图
        valid_graphs = [g for g in self.val_graphs if g.num_nodes('note') > 0]
        if not valid_graphs:
            raise RuntimeError("没有有效的验证数据")
            
        return DataLoader(
            valid_graphs,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._safe_collate,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        """返回测试数据加载器，使用验证集作为测试集"""
        return DataLoader(
            self.val_graphs,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._safe_collate,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def convert_to_graphs(self, graph_adapter, processed_dir, output_dir):
        """转换为图结构，添加批处理和进度显示"""
        try:
            processed_dir = Path(processed_dir)
            output_dir = Path(output_dir)
            
            # 查找所有.tok文件
            tok_files = list(processed_dir.glob('**/*.tok'))
            if not tok_files:
                raise RuntimeError(f"在 {processed_dir} 中没有找到处理后的文件")
            
            print(f"开始转换 {len(tok_files)} 个文件为图结构...")
            
            for tok_file in tqdm(tok_files, desc="转换为图结构"):
                try:
                    # 读取并验证数据
                    with open(tok_file, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError as je:
                            print(f"文件 {tok_file} 不是有效的JSON格式: {str(je)}")
                            continue
                    
                    # 详细的数据验证
                    if not isinstance(data, dict):
                        print(f"文件 {tok_file} 中的数据不是字典格式: {type(data)}")
                        continue
                    
                    # 验证必要的键
                    required_keys = ['token_sequence', 'mode_mask', 
                                   'tech_positions', 'tech_types']
                    missing_keys = [key for key in required_keys if key not in data]
                    if missing_keys:
                        print(f"文件 {tok_file} 缺少必要的键: {missing_keys}")
                        continue
                    
                    # 验证token_sequence
                    token_sequence = data.get('token_sequence', [])
                    if not isinstance(token_sequence, list):
                        print(f"文件 {tok_file} 中的token_sequence不是列表: {type(token_sequence)}")
                        continue
                        
                    if not token_sequence:
                        print(f"文件 {tok_file} 中的token_sequence为空")
                        continue
                    
                    # 打印调试信息
                    print(f"处理文件 {tok_file}:")
                    print(f"  - token_sequence长度: {len(token_sequence)}")
                    print(f"  - 前几个token: {token_sequence[:8]}")
                    
                    # 转换为图
                    try:
                        graph = graph_adapter.convert(data)
                        
                        # 验证图的有效性
                        if graph.num_nodes('note') == 0:
                            print(f"文件 {tok_file} 生成的图没有音符节点")
                            continue
                            
                        # 验证必要的特征
                        required_features = ['pitch', 'duration', 'velocity']
                        if not all(feat in graph.nodes['note'].data for feat in required_features):
                            print(f"文件 {tok_file} 生成的图缺少必要的特征: {[feat for feat in required_features if feat not in graph.nodes['note'].data]}")
                            continue
                            
                        # 验证特征的维度
                        if any(graph.nodes['note'].data[feat].shape[0] == 0 for feat in required_features):
                            print(f"文件 {tok_file} 生成的图特征维度为0")
                            continue
                            
                    except Exception as ge:
                        print(f"转换图结构时出错: {str(ge)}")
                        continue
                    
                    # 保存图
                    output_path = output_dir / tok_file.relative_to(processed_dir).with_suffix('.dgl')
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        dgl.save_graphs(str(output_path), [graph])
                    except Exception as se:
                        print(f"保存图结构时出错: {str(se)}")
                        continue
                    
                except Exception as e:
                    print(f"处理文件 {tok_file} 时出错: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"转换图结构时出错: {str(e)}")
            raise

    def _setup(self):
        """初始化数据加载"""
        try:
            # 检查图数据目录
            graph_dir = self.data_dir / 'graphs'
            if not graph_dir.exists() or not list(graph_dir.glob('*.dgl')):
                logger.info("图数据目录不存在或为空，将在setup阶段进行数据预处理")
                return
                
            logger.info(f"找到图数据目录: {graph_dir}")
        except Exception as e:
            logger.error(f"初始化数据加载失败: {str(e)}")
            return