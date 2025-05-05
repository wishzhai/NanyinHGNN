#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
两阶段训练脚本
第一阶段：使用优化的LoRA训练基本特征预测（音高、力度等）
第二阶段：使用Adapter训练音高预测，同时保持其他特征稳定
"""

import os
import sys
import yaml
import torch
import logging
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import TensorBoardLogger
import traceback
from datetime import datetime  # 添加datetime模块导入
import shutil
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enhanced_lightning_module import EnhancedNanyinModel, OrnamentCoordinator
from dataflow.data_module import NanyinDataModule
from core.ornament_processor import OrnamentProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_two_stage.log')
    ]
)

logger = logging.getLogger(__name__)

def init_tokenizer(config):
    """初始化分词器（在这里我们不需要实际的分词器，因为我们使用图结构）"""
    logger.info("跳过分词器初始化（使用图结构）")
    return None

def print_tokenizer_info(tokenizer, config):
    """打印分词器信息（在这里我们打印图结构相关信息）"""
    logger.info("图结构配置信息:")
    if 'model' in config:
        model_config = config['model']
        logger.info(f"- hidden_dim: {model_config.get('hidden_dim', 384)}")
        logger.info(f"- num_heads: {model_config.get('num_heads', 8)}")
        if 'graph' in model_config:
            graph_config = model_config['graph']
            for key, value in graph_config.items():
                logger.info(f"- {key}: {value}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='南音模型两阶段训练脚本')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据目录路径')
    parser.add_argument('--config', type=str, default='configs/two_stage_training.yaml',
                        help='配置文件路径')
    parser.add_argument('--log_dir', type=str, default='logs/enhanced_two_stage',
                        help='日志目录路径')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器的工作进程数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='训练精度，可选：32, 16, 16-mixed')
    parser.add_argument('--current_stage', type=int, default=1,
                        help='当前训练阶段，1 表示第一阶段，2 表示第二阶段')
    
    # 模型检查点相关参数
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='从检查点恢复训练')
    parser.add_argument('--stage1_ckpt', type=str, default=None,
                        help='第一阶段的检查点路径（用于直接开始第二阶段训练）')
    
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def setup_callbacks(config, stage):
    """设置训练回调函数"""
    callbacks = []
    
    # 根据阶段设置不同的保存目录
    stage_dir = f"checkpoints/stage{stage}"
    os.makedirs(stage_dir, exist_ok=True)
    
    # 设置模型检查点回调 - 监控验证损失
    checkpoint_callback = ModelCheckpoint(
        dirpath=stage_dir,  # 使用阶段特定的目录
        filename=f'nanyin-stage{stage}-' + '{epoch:02d}-{val_loss:.2f}',
        save_top_k=5,  # 增加保存的检查点数量
        verbose=True,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Stage 2: 添加装饰音指标监控
    if stage == 2:
        ornament_checkpoint = ModelCheckpoint(
            dirpath=stage_dir,  # 使用阶段特定的目录
            filename=f'nanyin-stage{stage}-ornament-' + '{epoch:02d}-{val_ornament_rationality:.4f}',
            save_top_k=3,  # 保存最好的3个模型
            verbose=True,
            monitor='val_ornament_rationality',  # 使用装饰音合理性总分
            mode='max',
            save_last=False
        )
        callbacks.append(ornament_checkpoint)
    
    # 设置早停回调
    early_stop_config = config.get(f'stage{stage}', {}).get('early_stopping', {})
    early_stop_callback = EarlyStopping(
        monitor='val_loss',  # 使用 val_loss 作为监控指标
        mode='min',  # 因为监控 loss，所以使用 min 模式
        patience=10,
        min_delta=0.001,  # 添加最小变化阈值
        verbose=True
    )
    logger.info(f"早停配置: patience={early_stop_callback.patience}, "
                f"monitor={early_stop_callback.monitor}, "
                f"mode={early_stop_callback.mode}, "
                f"min_delta={early_stop_callback.min_delta}")
    callbacks.append(early_stop_callback)
    
    # 设置学习率监控
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    return callbacks

def main(args):
    """主函数"""
    try:
        # 设置随机种子
        seed = args.seed if args.seed is not None else 42
        print(f"Seed set to {seed}")
        pl.seed_everything(seed)
        
        # 初始化配置
        logger.info("初始化模型配置...")
        config = load_config(args.config)
        
        # 设置当前训练阶段
        config['current_stage'] = args.current_stage
        logger.info(f"设置训练阶段为: {args.current_stage}")
        
        # 应用阶段特定的配置
        stage_key = f'stage{args.current_stage}'
        if stage_key in config:
            stage_config = config[stage_key]
            logger.info(f"找到阶段{args.current_stage}的特定配置")
            
            # 应用阶段特定的损失权重
            if 'train' in stage_config and 'loss_weights' in stage_config['train']:
                if 'loss_weights' not in config:
                    config['loss_weights'] = {}
                    
                # 将阶段特定的损失权重复制到根级别配置
                for key, value in stage_config['train']['loss_weights'].items():
                    config['loss_weights'][key] = value
                    logger.info(f"应用阶段{args.current_stage}特定的损失权重: {key}={value}")
            
            # 应用阶段特定的学习率
            if 'train' in stage_config and 'learning_rate' in stage_config['train']:
                config['learning_rate'] = stage_config['train']['learning_rate']
                logger.info(f"应用阶段{args.current_stage}特定的学习率: {config['learning_rate']}")
            
            # 应用阶段特定的自监督学习配置
            if 'self_supervised' in config:
                ss_weight = stage_config['train']['loss_weights'].get('self_supervised', 0.0)
                if ss_weight > 0:
                    config['self_supervised']['enabled'] = True
                    logger.info(f"启用自监督学习，权重为: {ss_weight}")
                    
                    # 特别强调撚指学习
                    if 'nianzhi_weight' in config['self_supervised']:
                        nianzhi_weight = config['self_supervised']['nianzhi_weight']
                        logger.info(f"启用撚指特征学习，撚指权重为: {nianzhi_weight}")
                    else:
                        # 如果没有指定撚指权重，设置一个默认值
                        config['self_supervised']['nianzhi_weight'] = 1.0
                        logger.info(f"未指定撚指权重，使用默认值: 1.0")
                    
                    # 确保撚指详细配置存在
                    if 'nianzhi' not in config['self_supervised']:
                        config['self_supervised']['nianzhi'] = {
                            'min_duration': 3.0,
                            'pitch_range': [55, 85],
                            'probability_threshold': 0.7,
                            'count_range': [3, 4]
                        }
                        logger.info("使用默认的撚指详细配置")
                else:
                    config['self_supervised']['enabled'] = False
                    logger.info("禁用自监督学习")
        
        # 添加数据加载器配置
        if 'data' not in config:
            config['data'] = {}
        config['data']['num_workers'] = args.num_workers
        config['data']['pin_memory'] = True
        config['data']['batch_size'] = args.batch_size
        
        # 跳过分词器初始化（使用图结构）
        logger.info("跳过分词器初始化（使用图结构）")
        
        # 打印图结构配置信息
        logger.info("图结构配置信息:")
        logger.info(f"- hidden_dim: {config.get('model', {}).get('hidden_dim', 384)}")
        logger.info(f"- num_heads: {config.get('model', {}).get('num_heads', 8)}")
        
        # 检查数据目录
        logger.info("检查数据目录...")
        if os.path.exists("data/graphs"):
            logger.info("找到图数据目录: data/graphs")
            # 检查是否有 .dgl 文件
            dgl_files = list(Path("data/graphs").glob("*.dgl"))
            if dgl_files:
                logger.info(f"找到 {len(dgl_files)} 个 .dgl 文件")
            else:
                raise FileNotFoundError("在 data/graphs 中没有找到 .dgl 文件")
        else:
            raise FileNotFoundError("未找到图数据目录")
        
        # 初始化数据模块
        data_module = NanyinDataModule(
            data_dir="data",  # 修改为根数据目录
            batch_size=args.batch_size,
            config=config  # 通过配置传入其他参数
        )
        
        # 从第一阶段检查点加载模型
        if args.stage1_ckpt:
            logger.info(f"正在从第一阶段检查点加载模型: {args.stage1_ckpt}")
            try:
                # 如果提供了第一阶段检查点，则强制设置为第二阶段
                if args.current_stage != 2:
                    logger.warning(f"提供了第一阶段检查点但当前阶段设置为 {args.current_stage}，自动调整为第二阶段")
                    config['current_stage'] = 2
                
                # 创建一个新的模型实例
                model = EnhancedNanyinModel(config)
                
                # 加载检查点
                checkpoint = torch.load(args.stage1_ckpt, map_location='cpu')
                
                # 过滤掉装饰音处理器相关的参数，因为尺寸可能不匹配
                filtered_state_dict = {}
                for key, value in checkpoint['state_dict'].items():
                    # 跳过装饰音处理器相关的参数
                    if 'ornament_processor.' in key:
                        logger.info(f"跳过加载参数: {key}，尺寸可能不匹配")
                        continue
                    filtered_state_dict[key] = value
                
                # 只加载模型权重，不加载完整状态
                # 使用 strict=False 允许跳过不匹配的键
                incompatible_keys = model.load_state_dict(filtered_state_dict, strict=False)
                
                if incompatible_keys.missing_keys:
                    logger.warning(f"以下键在检查点中缺失: {incompatible_keys.missing_keys}")
                if incompatible_keys.unexpected_keys:
                    logger.warning(f"以下键在检查点中未使用: {incompatible_keys.unexpected_keys}")
                
                # 确保规则组件已正确初始化
                if not hasattr(model, 'rule_injector') or model.rule_injector is None:
                    logger.info("正在初始化规则组件...")
                    model._init_rule_components()
                
                # 重新初始化装饰音处理器，以适应新的装饰音风格定义
                if hasattr(model, 'ornament_processor'):
                    logger.info("正在重新初始化装饰音处理器以适应新的装饰音风格定义...")
                    model.ornament_processor = OrnamentProcessor(config)
                    logger.info(f"装饰音处理器已重新初始化，支持的风格数量: {len(config.get('ornament_styles', {}))}")
                    
                    # 初始化装饰音协调器
                    if hasattr(model, 'rule_injector') and model.rule_injector is not None:
                        logger.info("正在初始化装饰音协调器...")
                        model.ornament_coordinator = OrnamentCoordinator(
                            model.rule_injector,
                            model.ornament_processor
                        )
                        logger.info("装饰音协调器初始化完成")
                
                logger.info("成功加载第一阶段模型权重")
            except Exception as e:
                logger.error(f"加载第一阶段检查点失败: {str(e)}")
                logger.error(traceback.format_exc())
                raise e
        else:
            # 创建新模型
            model = EnhancedNanyinModel(config)
            logger.info(f"创建了新的模型实例，当前阶段: {config['current_stage']}")
        
        # 配置训练器
        # 根据当前阶段选择相应的配置
        current_stage = config['current_stage']
        stage_key = f'stage{current_stage}'
        
        trainer = pl.Trainer(
            max_epochs=config[stage_key]['train'].get('max_epochs', 30),
            accelerator='auto',
            devices=1 if torch.cuda.is_available() else None,
            precision='32-true',  # 改为使用 32 位精度
            callbacks=setup_callbacks(config, current_stage),
            logger=TensorBoardLogger(
                save_dir=os.path.join('logs', 'enhanced_two_stage'),
                name=f'stage{current_stage}',
                version=datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
            gradient_clip_val=config[stage_key]['train'].get('gradient_clip_val', 1.0),
            accumulate_grad_batches=config[stage_key]['train'].get('accumulate_grad_batches', 1),
            val_check_interval=config[stage_key]['train'].get('val_check_interval', 0.25),
            log_every_n_steps=10,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            deterministic=True,
            detect_anomaly=True
        )
        
        # 开始训练
        logger.info("开始训练...")
        try:
            trainer.fit(model, data_module)
            logger.info("训练完成")
            
            # 获取当前时间作为文件名的一部分
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 保存最终模型，使用时间戳命名
            final_model_path = os.path.join('logs', 'enhanced_two_stage', f'stage{current_stage}', f'nanyin_model_stage{current_stage}_{current_time}.ckpt')
            trainer.save_checkpoint(final_model_path)
            logger.info(f"最终模型已保存到: {final_model_path}")
            
        except Exception as e:
            logger.error(f"训练过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            raise e
        
    except Exception as e:
        logger.error(f"训练过程出错: {str(e)}")
        logger.error(traceback.format_exc())
        raise e

if __name__ == '__main__':
    args = parse_args()
    main(args) 