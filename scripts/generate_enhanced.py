import os
import sys
# 注意: 此脚本需要 numpy<2.0 版本
# 如果遇到NumPy版本错误，请运行: pip install numpy==1.24.3
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import argparse
import dgl
from symusic import Score, Track, Note, Tempo
import time
import random
import traceback
import glob
import math
from datetime import datetime

# MIDI时间单位常量
TICKS_PER_QUARTER = 480  # 每四分音符的tick数

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from core.enhanced_lightning_module import EnhancedNanyinModel
from dataflow.rule_injector import RuleInjector
from dataflow.special_note_processor import SpecialNoteProcessor
from dataflow.graph_adapter import NanyinGraphAdapter
from core.instrument_rules.dongxiao import DongxiaoGenerator
from core.instrument_rules.erxian import ErxianGenerator
from core.utils import find_latest_checkpoint, ensure_dir
from core.model_adapter import ModelStructureAdapter
from scripts.inference_adapter import InferenceModelAdapter

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局字典用于存储撚指数据
NIANZHI_DATA_DICT = {}

def load_model(checkpoint_path, config, device='cuda'):
    """加载模型并适配参数
    
    Args:
        checkpoint_path: 检查点路径
        config: 配置字典
        device: 设备
        
    Returns:
        model: 加载并适配后的模型
    """
    try:
        # 1. 容错式加载检查点
        checkpoint = InferenceModelAdapter.load_checkpoint_tolerant(checkpoint_path, device)
        if checkpoint is None:
            logger.error("加载检查点失败")
            return None
            
        # 2. 创建模型实例
        model = EnhancedNanyinModel(config)
        
        # 3. 适配模型结构
        logger.info("正在适配模型结构...")
        model = ModelStructureAdapter.align_model_structure(model)
        
        # 4. 自适应加载参数
        logger.info("正在自适应加载参数...")
        model = InferenceModelAdapter.adaptive_parameter_loading(model, checkpoint, strict=False)
        
        # 5. 将模型移到设备上并设置为评估模式
        model = model.to(device)
        model.eval()
        
        return model
        
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_nianzhi_notes(position, duration, pitch, velocity, nianzhi_type=2, next_note_time=None, predicted_features=None):
    """生成撚指音符序列，使用自监督预测特征增强表现力
    
    Args:
        position: 起始位置
        duration: 持续时间
        pitch: 音高
        velocity: 力度
        nianzhi_type: 撚指类型 (1=快速, 2=标准, 3=慢速)
        next_note_time: 下一个音符的开始时间
        predicted_features: 自监督预测的特征 (可选)
    
    Returns:
        list: 撚指音符列表
    """
    try:
        notes = []
        max_available_time = next_note_time if next_note_time is not None else position + duration * 1.2
        
        # 根据撚指类型和预测特征设置参数
        if nianzhi_type == 1:  # 快速撚指
            nianzhi_count = random.randint(8, 12)
            base_interval = random.uniform(0.15, 0.25)  # 更短的基础间隔
            velocity_range = (0.85, 1.15)  # 更大的力度变化范围
            
        elif nianzhi_type == 3:  # 慢速撚指
            nianzhi_count = random.randint(4, 6)
            base_interval = random.uniform(0.4, 0.6)  # 更长的基础间隔
            velocity_range = (0.92, 1.08)  # 较小的力度变化范围
            
        else:  # 标准撚指
            nianzhi_count = random.randint(6, 9)
            base_interval = random.uniform(0.25, 0.4)
            velocity_range = (0.88, 1.12)
        
        # 计算基础持续时间
        total_duration = duration * 0.92
        base_duration = total_duration / (nianzhi_count * 1.2)
        base_duration = max(base_duration, 0.1)
        
        # 生成表现力曲线
        if predicted_features is not None and len(predicted_features) >= 2:
            # 使用预测特征调整间隔和力度
            timing_curve = predicted_features[0]  # 假设第一个特征控制时间
            velocity_curve = predicted_features[1]  # 假设第二个特征控制力度
        else:
            # 生成富有表现力的时间曲线（非线性）
            timing_curve = []
            for i in range(nianzhi_count):
                # 使用正弦波和指数衰减的组合
                phase = i / nianzhi_count * math.pi
                timing_factor = (1 + 0.3 * math.sin(phase)) * math.exp(-i / nianzhi_count * 0.5)
                timing_curve.append(timing_factor)
            
            # 生成富有表现力的力度曲线
            velocity_curve = []
            for i in range(nianzhi_count):
                # 使用多个正弦波的叠加
                phase1 = i / nianzhi_count * 2 * math.pi
                phase2 = i / nianzhi_count * 4 * math.pi
                vel_factor = 1.0 + 0.15 * math.sin(phase1) + 0.1 * math.sin(phase2)
                velocity_curve.append(vel_factor)
        
        # 生成音符
        current_position = position
        current_velocity = velocity
        
        for i in range(nianzhi_count):
            # 计算时间间隔（使用非线性曲线）
            interval_factor = timing_curve[i % len(timing_curve)]
            current_interval = base_interval * interval_factor
            
            # 计算持续时间
            current_duration = base_duration * random.uniform(0.95, 1.05)
            
            # 计算力度（使用非线性曲线）
            velocity_factor = velocity_curve[i % len(velocity_curve)]
            current_velocity = int(velocity * min(max(
                velocity_factor * random.uniform(velocity_range[0], velocity_range[1]),
                0.3  # 最小力度比例
            ), 1.5))  # 最大力度比例
            
            # 确保力度在有效范围内
            current_velocity = max(35, min(127, current_velocity))
            
            # 检查是否超出可用时间
            if current_position + current_duration > max_available_time:
                break
            
            # 创建音符
            note = Note(
                time=int(current_position),
                duration=max(1, int(current_duration)),
                pitch=pitch,
                velocity=current_velocity
            )
            notes.append(note)
            
            # 更新下一个音符的位置
            if i < nianzhi_count - 1:
                # 添加微小的随机变化
                jitter = random.uniform(0.95, 1.05)
                current_position += current_duration + (current_interval * jitter)
        
        # 确保最后一个音符不超出最大时间
        if notes and notes[-1].time + notes[-1].duration > max_available_time:
            notes[-1].duration = max(1, int(max_available_time - notes[-1].time))
        
        return notes
        
    except Exception as e:
        logger.error(f"撚指生成失败: {str(e)}")
        logger.error(traceback.format_exc())
        return [Note(
            time=int(position),
            duration=int(duration * 0.5),
            pitch=pitch,
            velocity=velocity
        )]

def generate_velocity_curve(velocity_pattern, nianzhi_count, finger_energy):
    """生成力度变化曲线,模拟真实演奏的力度变化
    
    Args:
        velocity_pattern: 力度变化模式
        nianzhi_count: 撚指音符数量
        finger_energy: 初始手指能量
        
    Returns:
        list: 力度曲线
    """
    velocity_curve = []
    
    if velocity_pattern == "gradual_decay":
        # 自然衰减 - 使用非线性衰减
        for i in range(nianzhi_count):
            # 使用指数衰减
            decay_base = random.uniform(0.75, 0.85)  # 基础衰减率
            decay_factor = decay_base ** (i * random.uniform(0.8, 1.2))
            # 添加微小的随机波动
            jitter = random.uniform(-0.05, 0.05)
            velocity_curve.append(max(0.4, decay_factor + jitter))
            
    elif velocity_pattern == "accent_first":
        # 第一音重强 - 更突出的对比
        for i in range(nianzhi_count):
            if i == 0:
                velocity_curve.append(random.uniform(0.95, 1.0))  # 首音更强
            else:
                # 后续音符使用快速衰减
                decay = 0.85 - (0.12 * i)  # 更大的衰减
                jitter = random.uniform(-0.03, 0.03)  # 小波动
                velocity_curve.append(max(0.4, decay + jitter))
                
    elif velocity_pattern == "accent_last":
        # 末音重强 - 渐强效果
        for i in range(nianzhi_count):
            if i == nianzhi_count - 1:
                velocity_curve.append(random.uniform(0.85, 0.95))
            else:
                # 前面的音符渐强
                base = 0.6 + (i / (nianzhi_count - 1)) * 0.2
                jitter = random.uniform(-0.04, 0.04)
                velocity_curve.append(base + jitter)
                
    elif velocity_pattern == "wave":
        # 波浪式 - 更自然的起伏
        for i in range(nianzhi_count):
            phase = i / nianzhi_count * 2 * math.pi
            # 使用多个正弦波叠加
            wave1 = 0.15 * math.sin(phase)
            wave2 = 0.08 * math.sin(2 * phase)  # 二倍频率
            wave = wave1 + wave2
            # 基础衰减
            base = 1.0 - 0.08 * i
            velocity_curve.append(max(0.4, base + wave))
            
    elif velocity_pattern == "dynamic_contrast":
        # 强弱对比 - 新增模式
        for i in range(nianzhi_count):
            if i % 2 == 0:
                # 强音
                velocity_curve.append(random.uniform(0.85, 0.95))
            else:
                # 弱音
                velocity_curve.append(random.uniform(0.55, 0.65))
                
    elif velocity_pattern == "expressive_rubato":
        # 富有表现力的自由速度 - 新增模式
        prev = random.uniform(0.8, 0.9)
        velocity_curve.append(prev)
        
        for i in range(1, nianzhi_count):
            # 生成与前一个音符相关的力度
            change = random.uniform(-0.15, 0.15)
            # 限制变化范围
            new_value = max(0.5, min(1.0, prev + change))
            velocity_curve.append(new_value)
            prev = new_value
            
    else:  # random_expressive
        # 随机但富有表现力
        base_curve = []
        # 生成基础曲线
        for i in range(nianzhi_count):
            if i == 0:
                base_curve.append(random.uniform(0.85, 0.95))
            else:
                prev = base_curve[-1]
                change = random.uniform(-0.12, 0.12)
                new_value = max(0.5, min(0.95, prev + change))
                base_curve.append(new_value)
        
        # 添加表现力变化
        for i in range(nianzhi_count):
            expression = random.uniform(-0.05, 0.05)
            final_value = max(0.4, min(1.0, base_curve[i] + expression))
            velocity_curve.append(final_value)
    
    # 确保所有力度值在合理范围内
    velocity_curve = [max(0.4, min(1.0, v)) for v in velocity_curve]
    
    # 根据手指能量调整整体力度
    energy_factor = finger_energy ** 0.5  # 使用平方根关系使调整更自然
    velocity_curve = [v * energy_factor for v in velocity_curve]
    
    return velocity_curve

def generate_expression_curve(rhythm_style, nianzhi_count):
    """生成表情变化曲线
    
    Args:
        rhythm_style: 韵律风格
        nianzhi_count: 撚指音符数量
        
    Returns:
        list: 表情曲线
    """
    if rhythm_style == "natural":
        # 自然波动
        return [1.0 + random.uniform(-0.1, 0.1) for _ in range(nianzhi_count)]
        
    elif rhythm_style == "accelerando":
        # 渐快
        base_curve = [1.0 - (i/nianzhi_count) * random.uniform(0.15, 0.3) for i in range(nianzhi_count)]
        # 添加小波动
        return [v + random.uniform(-0.05, 0.05) for v in base_curve]
        
    elif rhythm_style == "ritardando":
        # 渐慢
        base_curve = [1.0 + (i/nianzhi_count) * random.uniform(0.1, 0.25) for i in range(nianzhi_count)]
        # 添加小波动
        return [v + random.uniform(-0.05, 0.05) for v in base_curve]
        
    elif rhythm_style == "rubato":
        # 自由速度
        expression_curve = []
        prev = 1.0
        for _ in range(nianzhi_count):
            # 大幅度波动
            change = random.uniform(-0.2, 0.2)
            # 限制连续变化
            change = min(0.2, max(-0.2, change))
            value = max(0.7, min(1.3, prev + change))
            expression_curve.append(value)
            prev = value
        return expression_curve
        
    else:  # steady
        # 稳定节奏
        return [1.0 + random.uniform(-0.03, 0.03) for _ in range(nianzhi_count)]

def generate_note_timings(position, total_duration, nianzhi_count, interval_style, base_duration, max_available_time):
    """生成音符时间位置和持续时间，模拟人为演奏的自然时值变化
    
    Args:
        position: 起始位置
        total_duration: 总可用持续时间
        nianzhi_count: 音符数量
        interval_style: 间隔风格
        base_duration: 基础音符持续时间
        max_available_time: 最大可用时间
        
    Returns:
        tuple: (音符位置列表, 音符持续时间列表)
    """
    positions = []
    durations = []
    
    # 确保不会超出最大可用时间
    effective_duration = min(total_duration, max_available_time - position)
    
    # 生成表情曲线用于调制时值
    expression_curve = generate_expression_curve(interval_style, nianzhi_count)
    
    # 计算基础音符持续时间和最小间隔（减小间隔以实现更连续的效果）
    adjusted_base_duration = base_duration * 0.95  # 缩短音符以确保有微小间隔
    min_interval = max(0.02 * base_duration, 0.1)  # 显著减小最小间隔
    
    # 根据不同间隔风格生成间隔因子
    interval_factors = []
    
    if interval_style == "natural_variation":
        # 自然变化 - 使用更细微的变化
        prev_factor = 1.0
        for _ in range(nianzhi_count - 1):
            # 生成与前一个间隔相关的微小变化
            change = random.uniform(-0.08, 0.08)  # 减小变化范围
            # 使用指数平滑来保持连续性
            new_factor = prev_factor * 0.8 + (1.0 + change) * 0.2
            new_factor = max(0.85, min(1.15, new_factor))  # 限制变化范围
            interval_factors.append(new_factor)
            prev_factor = new_factor
            
    elif interval_style == "rubato":
        # 自由速度 - 更细腻的表现力变化
        base_curve = []
        # 使用多个正弦波叠加生成基础曲线
        for i in range(nianzhi_count - 1):
            phase1 = i / (nianzhi_count - 1) * 2 * math.pi
            phase2 = i / (nianzhi_count - 1) * 4 * math.pi
            phase3 = i / (nianzhi_count - 1) * 6 * math.pi
            
            wave1 = 0.08 * math.sin(phase1)  # 减小主波动
            wave2 = 0.04 * math.sin(phase2)  # 减小次波动
            wave3 = 0.02 * math.sin(phase3)  # 添加更高频率的细微变化
            
            factor = 1.0 + wave1 + wave2 + wave3
            base_curve.append(factor)
        
        # 添加微小的随机变化
        for factor in base_curve:
            jitter = random.uniform(-0.03, 0.03)  # 减小随机变化
            interval_factors.append(max(0.9, min(1.1, factor + jitter)))
            
    else:  # "expressive" 或其他
        # 富有表现力但保持连续 - 使用马尔可夫链
        states = ["normal", "slight_rush", "slight_delay"]
        transition_probs = {
            "normal": {"normal": 0.7, "slight_rush": 0.15, "slight_delay": 0.15},
            "slight_rush": {"normal": 0.6, "slight_rush": 0.3, "slight_delay": 0.1},
            "slight_delay": {"normal": 0.6, "slight_delay": 0.3, "slight_rush": 0.1}
        }
        
        current_state = "normal"
        prev_factor = 1.0
        
        for _ in range(nianzhi_count - 1):
            next_state = random.choices(
                list(transition_probs[current_state].keys()),
                list(transition_probs[current_state].values())
            )[0]
            
            # 根据状态生成更小的变化
            if next_state == "normal":
                change = random.uniform(-0.05, 0.05)
            elif next_state == "slight_rush":
                change = random.uniform(-0.08, -0.02)
            else:  # slight_delay
                change = random.uniform(0.02, 0.08)
            
            # 使用指数平滑
            new_factor = prev_factor * 0.7 + (1.0 + change) * 0.3
            new_factor = max(0.9, min(1.1, new_factor))
            interval_factors.append(new_factor)
            
            current_state = next_state
            prev_factor = new_factor
    
    # 应用表情曲线调制间隔因子，但减小影响
    interval_factors = [f * (1.0 + (e - 1.0) * 0.3) for f, e in zip(interval_factors, expression_curve[:-1])]
    
    # 计算实际的时间间隔（减小基础间隔）
    base_interval = adjusted_base_duration * 0.15  # 显著减小基础间隔
    intervals = [max(min_interval, base_interval * factor) for factor in interval_factors]
    
    # 计算总间隔时间，如果超出了总可用时间，按比例缩小所有间隔
    total_intervals = sum(intervals)
    total_required = adjusted_base_duration * nianzhi_count + total_intervals
    
    if total_required > effective_duration:
        # 按比例缩小所有间隔和持续时间，但保持表现力的相对关系
        scale_factor = (effective_duration / total_required) * 0.98  # 留出一点余量
        adjusted_base_duration *= scale_factor
        intervals = [interval * scale_factor for interval in intervals]
    
    # 计算音符位置和持续时间
    current_position = position
    positions.append(current_position)
    
    for i in range(nianzhi_count):
        # 设置当前音符的持续时间，使用更小的变化
        if i == 0:
            current_duration = adjusted_base_duration * (1.0 + (expression_curve[i] - 1.0) * 0.2)
        else:
            # 为后续音符添加更细微的持续时间变化
            duration_base = adjusted_base_duration * (1.0 + (expression_curve[i] - 1.0) * 0.2)
            micro_change = random.uniform(-0.02, 0.02)  # 减小随机变化
            current_duration = duration_base * (1.0 + micro_change)
        
        durations.append(max(1, current_duration))
        
        # 更新下一个音符的位置（确保连续性）
        if i < nianzhi_count - 1:
            current_position += current_duration + intervals[i]
            positions.append(current_position)
    
    return positions, durations

def generate_enhanced_music(
    model_path,
    config_path='configs/two_stage_training.yaml',
    output_dir="output",
    output_prefix="enhanced",
    tempo=30,
    max_length=512,  # 增加最大长度以适应更长的序列
    temperature=0.8,
    top_p=0.9,
    seed=None,
    apply_ornaments=True,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    """生成增强版南音音乐
    
    Args:
        model_path: 模型检查点路径
        config_path: 配置文件路径
        output_dir: 输出目录
        output_prefix: 输出文件前缀
        tempo: 速度（BPM）
        max_length: 生成的最大长度
        temperature: 生成温度，控制随机性
        top_p: top-p概率限制
        seed: 随机种子
        apply_ornaments: 是否应用装饰音规则
        device: 使用的设备
        
    Returns:
        tuple: (生成的图, 输出目录)
    """
    try:
        # 清空全局撚指数据字典
        global NIANZHI_DATA_DICT
        NIANZHI_DATA_DICT.clear()
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 如果没有指定模型路径，则使用最新的检查点
        if model_path is None:
            checkpoint_dir = os.path.join('logs', 'enhanced_two_stage', 'stage2')
            checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
            if not checkpoints:
                raise ValueError(f"在 {checkpoint_dir} 中未找到检查点文件")
            model_path = max(checkpoints, key=os.path.getctime)
            logger.info(f"使用最新的检查点: {model_path}")
        
        # 更新配置
        config['hidden_dim'] = 256  # 改回256
        config['feature_dim'] = 256  # 改回256
        config['current_stage'] = 2  # 设置为第二阶段
        
        # 启用自监督学习模块
        if 'self_supervised' not in config:
            config['self_supervised'] = {
                'enabled': True,
                'loss_weight': 0.5,
                'contour_weight': 0.3,
                'ornament_weight': 1.0,
                'nianzhi_weight': 0.5,
                'dropout': 0.25
            }
        else:
            config['self_supervised'].update({
                'enabled': True,
                'loss_weight': 0.5,
                'contour_weight': 0.3,
                'ornament_weight': 1.0,
                'nianzhi_weight': 0.5,
                'dropout': 0.25
            })
        
        # 更新装饰音处理器配置
        if 'ornament_processor' not in config:
            config['ornament_processor'] = {
                'num_layers': 2,
                'dropout': 0.1,
                'rnn_type': 'LSTM',
                'input_dim': 256,
                'hidden_dim': 256,
                'output_dim': 512,
                'projection_dim': 512,
                'bidirectional': True,
                'projection_first': False,
                'use_residual': True
            }
        else:
            config['ornament_processor'].update({
                'num_layers': 2,
                'dropout': 0.1,
                'rnn_type': 'LSTM',
                'input_dim': 256,
                'hidden_dim': 256,
                'output_dim': 512,
                'projection_dim': 512,
                'bidirectional': True,
                'projection_first': False,
                'use_residual': True
            })
            
        # 设置默认损失权重
        if 'loss_weights' not in config:
            config['loss_weights'] = {
                'pitch': 1.0,
                'pitch_smoothness': 0.2,
                'pitch_range': 0.1,
                'self_supervised': 0.5,
                'duration': 0.5,
                'velocity': 0.5
            }
        else:
            config['loss_weights'].update({
                'pitch': 1.0,
                'pitch_smoothness': 0.2,
                'pitch_range': 0.1,
                'self_supervised': 0.5,
                'duration': 0.5,
                'velocity': 0.5
            })
            
        # 确保模型配置中包含必要的参数
        if 'model' not in config:
            config['model'] = {}
        config['model'].update({
            'hidden_dim': 256,  # 基础隐藏维度
            'feature_dim': 256, # 特征维度
            'dropout': 0.2,
            'use_projection': True,  # 启用投影层
            'projection_dim': 256    # 投影维度改为256
        })
        
        # 更新dropout相关的配置
        if 'dropout' not in config:
            config['dropout'] = {
                'attn': 0.2,
                'feat': 0.2,
                'ffn': 0.2
            }
        
        # 加载模型
        logger.info(f"从 {model_path} 加载模型...")
        model = load_model(model_path, config, device)
        if model is None:
            raise ValueError("模型加载失败")
        
        # 创建一个起始的图结构
        logger.info("创建初始图结构...")
        
        # 添加南音音高映射配置
        if 'tokenizer' not in config:
            config['tokenizer'] = {}
            
        if 'NANYIN_PITCHES' not in config['tokenizer']:
            logger.info("添加默认的南音音高映射配置到tokenizer...")
            # 按照配置文件中的格式，键是MIDI音高值，值是南音符号
            config['tokenizer']['NANYIN_PITCHES'] = {
                50: 'd',    # D3 (小字组D)
                52: 'e',    # E3 (小字组E)
                53: 'f',    # F3 (小字组F)
                54: '#f',   # F#3 (小字组F#)
                55: 'g',    # G3 (小字组G)
                57: 'a',    # A3 (小字组A)
                59: 'b',    # B3 (小字组B)
                60: 'c1',   # C4 (小字一组C)
                61: '#c1',  # C#4 (小字一组C#)
                62: 'd1',   # D4 (小字一组D)
                64: 'e1',   # E4 (小字一组E)
                65: 'f1',   # F4 (小字一组F)
                66: '#f1',  # F#4 (小字一组F#)
                67: 'g1',   # G4 (小字一组G)
                69: 'a1',   # A4 (小字一组A)
                71: 'b1',   # B4 (小字一组B)
                72: 'c2',   # C5 (小字二组C)
                74: 'd2',   # D5 (小字二组D)
                76: 'e2',   # E5 (小字二组E)
                79: 'g2',   # G5 (小字二组G)
                81: 'a2',   # A5 (小字二组A)
                83: 'b2'    # B5 (小字二组B)
            }
            
        # 添加MODES配置
        if 'MODES' not in config['tokenizer']:
            logger.info("添加默认的南音调式配置到tokenizer...")
            config['tokenizer']['MODES'] = {
                'sikong': {
                    'tonic': 'g1',
                    'upper': {'g1', 'a1', 'c2', 'd2', 'e2'},
                    'lower': {'c1', 'd1', 'e1', 'g1', 'a1'}
                },
                'sizheng': {
                    'tonic': 'd1',
                    'upper': {'d1', 'e1', 'g1', 'a1', 'c2'},
                    'lower': {'g', 'a', 'c1', 'd1', 'e1'}
                }
            }
            
        # 添加反向映射（这个不需要在tokenizer中）
        config['PITCH_MAPPING'] = {v: k for k, v in config['tokenizer']['NANYIN_PITCHES'].items()}
            
        # 添加特色音配置
        if 'special_notes' not in config:
            logger.info("添加特色音配置...")
            config['special_notes'] = {'#f': 54, '#c1': 61, '#f1': 66}
        
        # 创建一个随机的起始音高序列
        seed_length = 64  # 从16增加到64，增加初始音符数量
        
        # 从配置中获取有效的南音音高
        valid_pitches = list(config['tokenizer']['NANYIN_PITCHES'].keys())
        valid_pitches.sort()  # 按音高排序
        
        # 包含特色音的候选音高
        special_notes = [54, 61, 66]  # #F(F#3), #C1(C#4), #F1(F#4) (特色音)
        
        # 选择中音区的普通音符作为基础
        mid_range_pitches = [p for p in valid_pitches if 60 <= p <= 72 and p not in special_notes]
        
        # 生成以级进为主的种子序列
        seed_pitches = []
        
        # 选择一个起始音高（从中音区开始）
        current_pitch = random.choice(mid_range_pitches)
        seed_pitches.append(current_pitch)
        
        # 生成剩余的音高
        for i in range(seed_length - 1):
            # 决定是否添加特色音
            if random.random() < 0.2:  # 降低特色音概率到20%
                next_pitch = random.choice(special_notes)
            else:
                # 生成级进为主的音高
                max_interval = 3  # 最大允许音程（小三度）
                if random.random() < 0.1:  # 10%概率允许更大的跳进
                    max_interval = 5  # 允许最大四度跳进
                
                # 获取当前音高附近的可用音高
                nearby_pitches = [p for p in valid_pitches 
                                if abs(p - current_pitch) <= max_interval
                                and p != current_pitch]
                
                if not nearby_pitches:
                    # 如果没有合适的音高，回到中音区
                    next_pitch = random.choice(mid_range_pitches)
                else:
                    # 优先选择更小的音程
                    weights = [1.0 / (abs(p - current_pitch) + 0.5) for p in nearby_pitches]
                    next_pitch = random.choices(nearby_pitches, weights=weights)[0]
            
            seed_pitches.append(next_pitch)
            current_pitch = next_pitch
        
        # 确保序列中包含足够的特色音
        special_note_count = sum(1 for p in seed_pitches if p in special_notes)
        min_special_notes = seed_length // 8  # 至少1/8是特色音
        
        if special_note_count < min_special_notes:
            # 随机位置插入特色音
            non_special_indices = [i for i, p in enumerate(seed_pitches) if p not in special_notes]
            indices_to_change = random.sample(non_special_indices, min_special_notes - special_note_count)
            
            for idx in indices_to_change:
                # 选择最近的特色音
                current_pitch = seed_pitches[idx]
                closest_special = min(special_notes, key=lambda x: abs(x - current_pitch))
                seed_pitches[idx] = closest_special
        
        logger.info(f"种子音高序列: {seed_pitches}，包含特色音: {[p for p in seed_pitches if p in special_notes]}")
        
        # 准备模型输入：创建初始图结构
        graph = dgl.heterograph({
            ('note', 'temporal', 'note'): ([], []),
            ('note', 'decorate', 'ornament'): ([], []),
            ('tech', 'trigger', 'note'): ([], [])
        })
        
        # 将图移到设备上
        graph = graph.to(device)
        logger.info(f"初始图结构已创建，设备: {device}")
        
        # 添加种子音符到图中
        success = add_seed_notes_to_graph(graph, seed_pitches)
        if not success:
            logger.error("添加种子音符失败，无法继续生成")
            return
        
        # 检查图中的节点数量
        note_count = graph.num_nodes('note')
        logger.info(f"图中现有 {note_count} 个音符节点")
        
        # 如果音符数量不足以应用撚指，添加更多的种子音符
        if note_count < 5:
            logger.info(f"音符数量不足以支持撚指功能，添加额外的种子音符...")
            # 准备更多的种子音符
            extra_pitches = []
            for _ in range(max(5 - note_count, 2)):  # 至少再添加2个，确保总数达到5个
                if random.random() < 0.3:  # 30%的概率添加特色音
                    extra_pitches.append(random.choice([54, 61, 66]))  # 特色音
                else:
                    extra_pitches.append(random.choice(mid_range_pitches))
            
            # 添加额外的种子音符
            success = add_seed_notes_to_graph(graph, extra_pitches, min_pos=8.0)  # 从位置8.0开始添加
            if success:
                note_count = graph.num_nodes('note')
                logger.info(f"成功添加额外种子音符，当前共有 {note_count} 个音符节点")
        
        # 使用模型扩展图结构
        logger.info("使用模型生成音乐...")
        
        # 由于模型没有generate方法，我们直接使用现有的图结构继续
        logger.info("检测到模型没有generate方法，使用种子图结构继续...")
        generated_graph = graph
        
        # 应用规则
        if apply_ornaments:
            logger.info("应用装饰音规则...")
            # 创建规则注入器配置
            rule_config = {
                'rule_injection': {
                    'pentatonic_boost': 3.0,  # 增加五声音阶提升权重
                    'base_decorate_weight': 1.0,  # 增加基础装饰权重
                    'ornament_density': 0.8,  # 增加装饰音密度
                    'enable_ornaments': True,
                    'min_interval': 2,  # 最小音程间隔
                    'max_interval': 12,  # 最大音程间隔
                    'ornament_types': ['grace', 'trill', 'mordent', 'turn'],  # 支持更多装饰音类型
                    'style_weights': {
                        'grace': 0.4,
                        'trill': 0.3,
                        'mordent': 0.2,
                        'turn': 0.1
                    }
                },
                'special_notes': config.get('special_notes', {'#f': 54, '#c1': 61, '#f1': 66})
            }
            
            logger.info(f"使用特色音配置: {rule_config['special_notes']}")
            
            # 创建规则注入器
            rule_injector = RuleInjector(rule_config)
                
                # 应用规则
            logger.info("正在应用规则注入，包括特色音和装饰音...")
            generated_graph = rule_injector.apply(generated_graph)
            
            # 添加撚指标记，传递模型实例以使用自监督学习模块进行预测
            generated_graph = add_manual_nianzhi(generated_graph, model)
        
        # 将图转换为MIDI文件
        logger.info("转换为MIDI文件...")
        scores = graph_to_enhanced_scores(generated_graph, tempo)
        
        # 保存MIDI文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        for instrument, score in scores.items():
            inst_output_path = os.path.join(output_dir, f"{output_prefix}_{instrument}_{timestamp}.mid")
            try:
                if hasattr(score, 'dump_midi') and callable(score.dump_midi):
                    score.dump_midi(inst_output_path)
                    logger.info(f"生成的{instrument}音乐已保存到 {inst_output_path} (使用dump_midi)")
                elif hasattr(score, 'dumps_midi') and callable(score.dumps_midi):
                    midi_bytes = score.dumps_midi()
                    with open(inst_output_path, 'wb') as f:
                        f.write(midi_bytes)
                    logger.info(f"生成的{instrument}音乐已保存到 {inst_output_path} (使用dumps_midi)")
                elif hasattr(score, 'write') and callable(score.write):
                    score.write(inst_output_path)
                    logger.info(f"生成的{instrument}音乐已保存到 {inst_output_path} (使用write)")
                elif hasattr(score, 'dump') and callable(score.dump):
                    score.dump(inst_output_path)
                    logger.info(f"使用dump方法保存{instrument}音乐到 {inst_output_path}")
                elif hasattr(score, 'save') and callable(score.save):
                    score.save(inst_output_path)
                    logger.info(f"使用save方法保存{instrument}音乐到 {inst_output_path}")
            except Exception as e:
                logger.error(f"保存{instrument}音乐时出错: {str(e)}")
                logger.error(traceback.format_exc())
                
        # 合并所有音轨
        combined_score = Score()
        combined_score.tempos.append(Tempo(0, tempo))
        
        for score in scores.values():
            for track in score.tracks:
                combined_score.tracks.append(track)
                
        combined_path = os.path.join(output_dir, f"{output_prefix}_combined_{timestamp}.mid")
        try:
            # 尝试使用不同的方法保存合并的Score
            if hasattr(combined_score, 'dump_midi') and callable(combined_score.dump_midi):
                combined_score.dump_midi(combined_path)
                logger.info(f"合并的音乐已保存到 {combined_path} (使用dump_midi)")
            elif hasattr(combined_score, 'dumps_midi') and callable(combined_score.dumps_midi):
                midi_bytes = combined_score.dumps_midi()
                with open(combined_path, 'wb') as f:
                    f.write(midi_bytes)
                logger.info(f"合并的音乐已保存到 {combined_path} (使用dumps_midi)")
            elif hasattr(combined_score, 'write') and callable(combined_score.write):
                combined_score.write(combined_path)
                logger.info(f"合并的音乐已保存到 {combined_path}")
            elif hasattr(combined_score, 'dump') and callable(combined_score.dump):
                combined_score.dump(combined_path)
                logger.info(f"使用dump方法保存合并音乐到 {combined_path}")
            elif hasattr(combined_score, 'save') and callable(combined_score.save):
                combined_score.save(combined_path)
                logger.info(f"使用save方法保存合并音乐到 {combined_path}")
            else:
                logger.error(f"无法保存合并音乐，Score对象类型 {type(combined_score).__name__} 没有可用的保存方法")
                logger.info(f"可用方法: {[m for m in dir(combined_score) if not m.startswith('_') and callable(getattr(combined_score, m))]}")
        except Exception as e:
            logger.error(f"保存合并音乐时出错: {str(e)}")
            logger.error(traceback.format_exc())
        
        return generated_graph, output_dir
        
    except Exception as e:
        logger.error(f"生成音乐时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None

def graph_to_enhanced_scores(graph, tempo=120):
    """将图结构转换为多个MIDI文件，每个乐器一个文件"""
    logging.info("开始将图转换为增强版MIDI格式...")
    
    # 使用全局撚指数据字典
    global NIANZHI_DATA_DICT
    
    # 创建Score对象
    pipa_score = Score()
    sanxian_score = Score()
    dongxiao_score = Score()
    erxian_score = Score()
    
    # 创建轨道
    pipa_track = Track()
    pipa_track.program = 0  # 琵琶音色
    
    sanxian_track = Track()
    sanxian_track.program = 1  # 三弦音色
    
    dongxiao_track = Track()
    dongxiao_track.program = 2  # 洞箫音色
    
    erxian_track = Track()
    erxian_track.program = 3  # 二弦音色
    
    # 添加速度设置
    tempo_event = Tempo(0, tempo)
    pipa_score.tempos.append(tempo_event)
    sanxian_score.tempos.append(tempo_event)
    dongxiao_score.tempos.append(tempo_event)
    erxian_score.tempos.append(tempo_event)
    
    # 设置每个Score的PPQ (Pulses Per Quarter note)
    pipa_score.ticks_per_quarter = TICKS_PER_QUARTER
    sanxian_score.ticks_per_quarter = TICKS_PER_QUARTER
    dongxiao_score.ticks_per_quarter = TICKS_PER_QUARTER
    erxian_score.ticks_per_quarter = TICKS_PER_QUARTER
    
    try:
        # 获取音符数据
        note_nodes = graph.nodes['note'].data
        pitches = note_nodes['pitch']
        positions = note_nodes['position']
        durations = note_nodes['duration']
        velocities = note_nodes['velocity']
        
        # 检查是否有装饰音和撚指标记
        is_ornament = note_nodes.get('is_ornament', None)
        is_nianzhi = note_nodes.get('is_nianzhi', None)
        nianzhi_types = note_nodes.get('nianzhi_type', None)
        
        # 构建音符数据列表
        note_data = []
        valid_notes = 0
        nianzhi_notes = 0
        
        for i in range(len(pitches)):
            pitch = int(pitches[i])
            position = float(positions[i])
            duration = float(durations[i])
            velocity = int(velocities[i])
            
            # 检查音符数据是否有效
            if not (0 <= pitch <= 127):
                logger.warning(f"无效的音高: {pitch}")
                continue
            if duration <= 0:
                logger.warning(f"无效的持续时间: {duration}")
                continue
            if not (0 <= velocity <= 127):
                logger.warning(f"无效的力度: {velocity}")
                continue
            
            # 是否为装饰音
            is_ornament_note = False
            if is_ornament is not None and is_ornament[i] > 0.5:
                is_ornament_note = True
                
            # 是否为撚指音符
            is_nianzhi_note = False
            nianzhi_type_value = 2  # 默认为标准类型
            if is_nianzhi is not None and is_nianzhi[i] > 0.5:
                is_nianzhi_note = True
                if nianzhi_types is not None:
                    nianzhi_type_value = int(nianzhi_types[i])
                    if not (1 <= nianzhi_type_value <= 3):
                        nianzhi_type_value = 2  # 类型无效使用默认值
            
            # 添加到音符数据列表
            note_data.append({
                'pitch': pitch,
                'position': position,
                'duration': duration,
                'velocity': velocity,
                'is_ornament': is_ornament_note,
                'is_nianzhi': is_nianzhi_note,
                'nianzhi_type': nianzhi_type_value
            })
            
            # 如果是撚指音符，增加计数
            if is_nianzhi_note:
                nianzhi_notes += 1
            
            valid_notes += 1
        
        # 获取基准音高（用于生成装饰音）
        base_pitch = int(np.median([note['pitch'] for note in note_data if not note['is_ornament']]))
        
        # 初始化洞箫和二弦生成器
        dongxiao_gen = DongxiaoGenerator(base_pitch)
        erxian_gen = ErxianGenerator(base_pitch)
        
        # 构建主旋律音符列表（用于生成洞箫和二弦的装饰音）
        main_notes = []
        for note in note_data:
            if not note['is_ornament']:
                main_notes.append({
                    'pitch': note['pitch'],
                    'start': note['position'],
                    'duration': note['duration'],
                    'velocity': note['velocity']
                })
        
        # 生成洞箫装饰音
        dongxiao_notes = dongxiao_gen.generate(main_notes)
        for note in dongxiao_notes:
            dongxiao_track.notes.append(Note(
                time=int(note['start'] * TICKS_PER_QUARTER),
                duration=int(note['duration'] * TICKS_PER_QUARTER),
                pitch=note['pitch'],
                velocity=note['velocity']
            ))
        
        # 生成二弦装饰音
        erxian_notes = erxian_gen.generate(main_notes)
        for note in erxian_notes:
            erxian_track.notes.append(Note(
                time=int(note['start'] * TICKS_PER_QUARTER),
                duration=int(note['duration'] * TICKS_PER_QUARTER),
                pitch=note['pitch'],
                velocity=note['velocity']
            ))
        
        # 添加琵琶音符（主旋律）
        for i, note in enumerate(note_data):
            # 跳过装饰音
            if note['is_ornament']:
                continue
            
            # 计算下一个非装饰音音符的位置
            next_note_time = None
            try:
                for next_idx in range(i + 1, len(note_data)):
                    if next_idx < len(note_data) and 'is_ornament' in note_data[next_idx] and not note_data[next_idx]['is_ornament']:
                        next_note_time = note_data[next_idx]['position']
                        break
            except Exception as e:
                logger.warning(f"查找下一个非装饰音音符时出错: {str(e)}")
                # 继续执行，使用None作为下一个音符时间
            
            # 检查是否为撚指音符
            if note['is_nianzhi']:
                # 优先使用预先存储的撚指音符数据
                nianzhi_notes_list = []
                
                # 使用全局字典获取撚指数据
                global NIANZHI_DATA_DICT
                if i in NIANZHI_DATA_DICT and NIANZHI_DATA_DICT[i]:
                    # 使用预先生成的撚指音符数据
                    logger.info(f"使用预先生成的撚指音符数据，包含 {len(NIANZHI_DATA_DICT[i])} 个音符")
                    for nianzhi_item in NIANZHI_DATA_DICT[i]:
                        nianzhi_note = Note(
                            time=int(nianzhi_item['time']),
                            duration=int(nianzhi_item['duration']),
                            pitch=nianzhi_item['pitch'],
                            velocity=nianzhi_item['velocity']
                        )
                        nianzhi_notes_list.append(nianzhi_note)
                else:
                    # 如果没有预先生成的数据，重新生成撚指音符
                    logger.info(f"没有预先生成的撚指数据，重新生成撚指音符")
                    nianzhi_notes_list = generate_nianzhi_notes(
                        note['position'], 
                        note['duration'],
                        note['pitch'],
                        note['velocity'],
                        note['nianzhi_type'],
                        next_note_time
                    )
                
                # 添加撚指音符到琵琶轨道，并转换时间单位
                if nianzhi_notes_list:
                    for nianzhi_note in nianzhi_notes_list:
                        # 撚指音符在生成时已经是整数类型，这里只需乘以TICKS_PER_QUARTER
                        nianzhi_note.time = int(nianzhi_note.time * TICKS_PER_QUARTER)
                        nianzhi_note.duration = int(nianzhi_note.duration * TICKS_PER_QUARTER)
                        pipa_track.notes.append(nianzhi_note)
                    
                    # 增加撚指计数
                    nianzhi_notes += 1
                    logger.info(f"为音符 {i} 添加了 {len(nianzhi_notes_list)} 个撚指音符")
                else:
                    # 如果撚指生成失败，添加普通音符
                    logger.warning(f"撚指生成失败，添加普通音符")
                    pipa_track.notes.append(Note(
                        time=int(note['position'] * TICKS_PER_QUARTER),
                        duration=int(note['duration'] * TICKS_PER_QUARTER),
                        pitch=note['pitch'],
                        velocity=note['velocity']
                    ))
            else:
                # 添加普通音符
                pipa_track.notes.append(Note(
                    time=int(note['position'] * TICKS_PER_QUARTER),
                    duration=int(note['duration'] * TICKS_PER_QUARTER),
                    pitch=note['pitch'],
                    velocity=note['velocity']
                ))
        
        # 添加三弦音符（比琵琶低八度）
        for i, note in enumerate(note_data):
            # 跳过装饰音
            if note['is_ornament']:
                continue
            
            # 计算下一个非装饰音音符的位置
            next_note_time = None
            try:
                for next_idx in range(i + 1, len(note_data)):
                    if next_idx < len(note_data) and 'is_ornament' in note_data[next_idx] and not note_data[next_idx]['is_ornament']:
                        next_note_time = note_data[next_idx]['position']
                        break
            except Exception as e:
                logger.warning(f"查找下一个非装饰音音符时出错: {str(e)}")
                # 继续执行，使用None作为下一个音符时间
            
            # 检查是否为撚指音符
            if note['is_nianzhi']:
                # 优先使用预先存储的撚指音符数据，并降低八度
                nianzhi_notes_list = []
                
                # 尝试从全局字典获取撚指数据
                if i in NIANZHI_DATA_DICT and NIANZHI_DATA_DICT[i]:
                    # 使用预先生成的撚指音符数据，但降低八度
                    logger.info(f"三弦使用预生成的撚指数据，包含 {len(NIANZHI_DATA_DICT[i])} 个音符")
                    for note_info in NIANZHI_DATA_DICT[i]:
                        # 复制数据但降低八度
                        nianzhi_note = Note(
                            time=int(note_info['time']),
                            duration=int(note_info['duration']),
                            pitch=note_info['pitch'] - 12,  # 比琵琶低八度
                            velocity=note_info['velocity']
                        )
                        nianzhi_notes_list.append(nianzhi_note)
                else:
                    # 如果没有预先生成的数据，重新生成撚指音符
                    logger.info(f"三弦重新生成撚指音符")
                    nianzhi_notes_list = generate_nianzhi_notes(
                        note['position'], 
                        note['duration'],
                        note['pitch'] - 12,  # 比琵琶低八度
                        note['velocity'],
                        note['nianzhi_type'],
                        next_note_time
                    )
                
                # 添加撚指音符到三弦轨道，并转换时间单位
                for nianzhi_note in nianzhi_notes_list:
                    # 撚指音符在生成时已经是整数类型，这里只需乘以TICKS_PER_QUARTER
                    nianzhi_note.time = int(nianzhi_note.time * TICKS_PER_QUARTER)
                    nianzhi_note.duration = int(nianzhi_note.duration * TICKS_PER_QUARTER)
                    sanxian_track.notes.append(nianzhi_note)
            else:
                # 添加普通音符
                sanxian_track.notes.append(Note(
                    time=int(note['position'] * TICKS_PER_QUARTER),
                    duration=int(note['duration'] * TICKS_PER_QUARTER),
                    pitch=note['pitch'] - 12,  # 比琵琶低八度
                    velocity=note['velocity']
                ))
        
        # 添加轨道到Score
        pipa_score.tracks.append(pipa_track)
        sanxian_score.tracks.append(sanxian_track)
        dongxiao_score.tracks.append(dongxiao_track)
        erxian_score.tracks.append(erxian_track)
        
        # 记录音符信息
        logger.info(f"成功添加 {valid_notes} 个有效音符，其中 {nianzhi_notes} 个撚指音符")
        
        # 返回所有乐器的Score
        return {
            'pipa': pipa_score,
            'sanxian': sanxian_score,
            'dongxiao': dongxiao_score,
            'erxian': erxian_score
        }
        
    except Exception as e:
        logger.error(f"处理音符数据时出错: {str(e)}")
        logger.error(traceback.format_exc())
        
    # 出错时返回空的Score
    return {
        'pipa': pipa_score,
        'sanxian': sanxian_score,
        'dongxiao': dongxiao_score,
        'erxian': erxian_score
    }

def add_melody_notes(track, pitches, positions, durations, velocities, is_ornament=None, is_nianzhi=None, nianzhi_types=None):
    """添加主旋律音符到轨道，确保单音限制
    
    Args:
        track: 音乐轨道
        pitches: 音高列表
        positions: 位置列表
        durations: 持续时间列表
        velocities: 力度列表
        is_ornament: 是否为装饰音列表
        is_nianzhi: 是否为撚指音列表
        nianzhi_types: 撚指类型列表
    """
    # 按时间排序音符
    note_data = []
    for i in range(len(pitches)):
        # 跳过装饰音
        if is_ornament is not None and is_ornament[i]:
            continue
            
        # 是否为撚指音符
        is_nianzhi_note = False
        nianzhi_type = 2  # 默认标准撚指类型
        if is_nianzhi is not None and is_nianzhi[i]:
            is_nianzhi_note = True
            if nianzhi_types is not None:
                nianzhi_type = nianzhi_types[i]
            
        note_data.append({
            'pitch': int(pitches[i]),
            'time': int(positions[i]),
            'duration': int(durations[i]),
            'velocity': int(velocities[i]),
            'is_nianzhi': is_nianzhi_note,
            'nianzhi_type': nianzhi_type
        })
    
    # 按时间排序
    note_data.sort(key=lambda x: x['time'])
    
    # 添加音符，确保不重叠
    for i, note in enumerate(note_data):
        # 计算下一个音符的开始时间，用于避免撚指音符重叠
        next_note_time = None
        if i < len(note_data) - 1:
            next_note_time = note_data[i+1]['time']
        
        # 如果是撚指音符，生成撚指音符序列
        if note['is_nianzhi']:
            logger.info(f"生成撚指音符: 音高={note['pitch']}, 持续时间={note['duration'] / 100:.2f}, 类型={note['nianzhi_type']}")
            nianzhi_notes = generate_nianzhi_notes(
                note['time'],
                note['duration'],
                note['pitch'],
                note['velocity'],
                note['nianzhi_type'],
                next_note_time
            )
            
            # 添加撚指音符到轨道，并应用时间单位转换
            for nianzhi_note in nianzhi_notes:
                # 撚指音符的时间和持续时间在生成时已经是整数类型
                # 我们只需要乘以TICKS_PER_QUARTER来应用时间单位转换
                nianzhi_note.time = int(nianzhi_note.time * TICKS_PER_QUARTER)
                nianzhi_note.duration = int(nianzhi_note.duration * TICKS_PER_QUARTER)
                track.notes.append(nianzhi_note)
        else:
            # 创建普通音符，应用时间单位转换
            new_note = Note(
                time=int(note['time'] * TICKS_PER_QUARTER),
                duration=int(note['duration'] * TICKS_PER_QUARTER),
                pitch=note['pitch'],
                velocity=note['velocity']
            )
            
            # 添加到轨道
            track.notes.append(new_note)

def add_ornament_notes(graph, dongxiao_track):
    """添加装饰音音符
    
    Args:
        graph: 生成的图
        dongxiao_track: 洞箫音轨
    """
    try:
        # 获取所有音符节点
        note_nodes = graph.nodes['note'].data
        
        # 检查是否存在 is_ornament 特征，如果不存在则初始化
        if 'is_ornament' not in note_nodes:
            logger.info("图中不存在装饰音标记，初始化为全零")
            device = graph.device
            is_ornament = torch.zeros(graph.num_nodes('note'), device=device)
            note_nodes['is_ornament'] = is_ornament
            return  # 由于没有装饰音标记，直接返回
        
        # 遍历所有音符
        for i in range(graph.num_nodes('note')):
            # 检查是否是装饰音
            if note_nodes['is_ornament'][i] > 0:
                # 获取装饰音的音高和时值
                pitch = int(note_nodes['pitch'][i].item())
                duration = int(note_nodes['duration'][i].item())  # 注释说已经是tick单位
                position = int(note_nodes['position'][i].item())  # 注释说已经是tick单位
                velocity = int(note_nodes['velocity'][i].item())
                
                # 添加装饰音音符，使用正确的参数类型
                # 注释表明position和duration已经是tick单位，但为安全起见，我们检查一下它们的数值范围
                if position < 100 or duration < 10:  # 如果值太小，可能需要转换
                    position = int(position * TICKS_PER_QUARTER)
                    duration = int(duration * TICKS_PER_QUARTER)
                
                dongxiao_track.notes.append(Note(
                    time=position,
                    duration=duration,
                    pitch=pitch,
                    velocity=velocity
                ))
    except Exception as e:
        logging.error(f"添加装饰音音符时出错: {str(e)}")
        logging.error(traceback.format_exc())

def add_seed_notes_to_graph(graph, seed_pitches, min_pos=0.0):
    """添加种子音符到图中
    
    Args:
        graph: 要添加音符的图
        seed_pitches: 种子音符的音高列表
        min_pos: 最小位置值
    
    Returns:
        bool: 是否成功添加种子音符
    """
    if not seed_pitches:
        logger.warning("没有种子音符可添加")
        return False
        
    # 获取图的设备
    device = graph.device
    logger.info(f"添加种子音符 - 图的设备: {device}")
    
    try:
        # 获取当前图中的音符数量
        current_notes = graph.num_nodes('note')
        logger.info(f"当前图中已有 {current_notes} 个音符节点")
        
        # 为所有种子音符创建特征张量
        num_seeds = len(seed_pitches)
        
        # 音高张量
        pitches = torch.tensor(seed_pitches, dtype=torch.long, device=device)
        
        # 随机选择持续时间
        durations = []
        for _ in range(num_seeds):
            durations.append(random.choice([2.0, 4.0, 8.0]))
        durations = torch.tensor(durations, dtype=torch.float, device=device)
        
        # 使用标准力度
        velocities = torch.ones(num_seeds, dtype=torch.long, device=device) * 80
        
        # 计算位置
        positions = []
        pos = min_pos
        for i in range(num_seeds):
            positions.append(pos)
            pos += durations[i].item()
        positions = torch.tensor(positions, dtype=torch.float, device=device)
        
        # 添加节点
        logger.info(f"正在添加 {num_seeds} 个种子音符节点...")
        graph.add_nodes(num_seeds, ntype='note')
        
        # 检查特征是否已存在
        if current_notes > 0:
            # 已有特征，需要合并
            logger.info("已有音符特征，进行合并...")
            
            for feature_name, new_tensor in [
                ('pitch', pitches),
                ('duration', durations),
                ('velocity', velocities),
                ('position', positions)
            ]:
                if feature_name in graph.nodes['note'].data:
                    # 获取当前特征
                    current_tensor = graph.nodes['note'].data[feature_name]
                    
                    # 创建新的合并张量 - 确保在同一设备上
                    combined = torch.zeros(current_notes + num_seeds, 
                                          dtype=current_tensor.dtype, 
                                          device=device)
                    
                    # 复制现有特征
                    combined[:current_notes] = current_tensor
                    
                    # 添加新特征
                    combined[current_notes:] = new_tensor
                    
                    # 更新图中的特征
                    graph.nodes['note'].data[feature_name] = combined
                    logger.info(f"合并了 {feature_name} 特征，形状: {combined.shape}")
                else:
                    # 特征不存在，创建新的
                    padded = torch.zeros(current_notes + num_seeds, 
                                        dtype=new_tensor.dtype, 
                                        device=device)
                    padded[current_notes:] = new_tensor
                    graph.nodes['note'].data[feature_name] = padded
                    logger.info(f"创建了新的 {feature_name} 特征，形状: {padded.shape}")
                    
            # 处理 is_ornament 特征
            if 'is_ornament' in graph.nodes['note'].data:
                # 获取当前特征
                current_is_ornament = graph.nodes['note'].data['is_ornament']
                # 创建新的合并张量
                combined_is_ornament = torch.zeros(current_notes + num_seeds, device=device)
                # 复制现有特征
                combined_is_ornament[:current_notes] = current_is_ornament
                # 新添加的音符默认不是装饰音
                combined_is_ornament[current_notes:] = 0
                # 更新图中的特征
                graph.nodes['note'].data['is_ornament'] = combined_is_ornament
            else:
                # 创建新的 is_ornament 特征
                is_ornament = torch.zeros(current_notes + num_seeds, device=device)
                graph.nodes['note'].data['is_ornament'] = is_ornament
                logger.info("初始化了 is_ornament 特征")
        else:
            # 没有现有特征，直接设置
            graph.nodes['note'].data['pitch'] = pitches
            graph.nodes['note'].data['duration'] = durations
            graph.nodes['note'].data['velocity'] = velocities
            graph.nodes['note'].data['position'] = positions
            # 初始化 is_ornament 特征
            graph.nodes['note'].data['is_ornament'] = torch.zeros(num_seeds, device=device)
            logger.info("直接设置了种子音符特征，包括 is_ornament")
            
        # 添加时序边（按位置顺序）
        if num_seeds > 1:
            # 获取所有新添加的节点索引
            new_node_indices = torch.arange(current_notes, current_notes + num_seeds, device=device)
            
            # 创建按位置排序的时序边
            sorted_indices = torch.argsort(positions)
            sorted_nodes = new_node_indices[sorted_indices]
            
            # 添加边
            src_nodes = sorted_nodes[:-1]  # 除了最后一个
            dst_nodes = sorted_nodes[1:]   # 除了第一个
            graph.add_edges(src_nodes, dst_nodes, etype='temporal')
            
            logger.info(f"添加了 {len(src_nodes)} 条时序边连接种子音符")
            
        logger.info(f"成功添加 {num_seeds} 个种子音符，当前图中共有 {graph.num_nodes('note')} 个音符节点")
        return True
        
    except Exception as e:
        logger.error(f"添加种子音符到图时出错: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def add_manual_nianzhi(graph, model=None, device=None):
    """为图中的特定音符添加撚指标记，并支持不同类型的撚指
    
    Args:
        graph: DGL图结构
        model: 预测撚指类型的模型 (可选)
        device: 计算设备 (可选)
    
    Returns:
        修改后的图
    """
    try:
        logger.info("开始添加撚指标记...")
        
        # 从环境变量获取撚指生成参数
        force_nianzhi = os.environ.get('FORCE_NIANZHI', 'FALSE').upper() == 'TRUE'
        nianzhi_threshold = float(os.environ.get('NIANZHI_THRESHOLD', '0.03'))
        use_self_supervised = os.environ.get('USE_SELF_SUPERVISED', 'FALSE').upper() == 'TRUE'
        
        if force_nianzhi:
            logger.info("启用强制撚指生成模式")
        if use_self_supervised:
            logger.info("启用自监督撚指预测")
        if nianzhi_threshold != 0.03:
            logger.info(f"使用自定义撚指阈值: {nianzhi_threshold}")
        
        # 如果图中节点太少则返回原图
        if graph.num_nodes('note') < 3:
            logger.info("图中音符数量不足，无法添加撚指")
            return graph
        
        # 获取音符特征
        if 'pitch' not in graph.nodes['note'].data:
            logger.warning("图中没有pitch特征，无法添加撚指")
            return graph
        
        # 确定设备
        device = graph.device if device is None else device
        
        # 从图中获取音符特征
        pitches = graph.nodes['note'].data['pitch']
        positions = graph.nodes['note'].data['position']
        durations = graph.nodes['note'].data['duration']
        
        # 创建撚指标记特征
        is_nianzhi = torch.zeros(graph.num_nodes('note'), device=device)
        nianzhi_type = torch.zeros(graph.num_nodes('note'), device=device)
        
        # 为自监督预测生成features特征
        if model is not None and 'features' not in graph.nodes['note'].data:
            try:
                logger.info("正在为自监督预测生成features特征...")
                
                # 提取基础特征
                num_nodes = graph.num_nodes('note')
                
                # 准备基础特征向量 [pitch, position, duration, velocity]
                if 'velocity' in graph.nodes['note'].data:
                    velocities = graph.nodes['note'].data['velocity']
                else:
                    velocities = torch.ones(num_nodes, device=device) * 80
                
                # 归一化特征
                norm_pitch = (pitches.float() - 60.0) / 24.0
                norm_duration = torch.log1p(durations) / 2.0
                norm_velocity = velocities.float() / 127.0
                
                # 确保所有特征都是二维的
                if norm_pitch.dim() == 1:
                    norm_pitch = norm_pitch.unsqueeze(1)
                if positions.dim() == 1:
                    pos = positions.unsqueeze(1)
                else:
                    pos = positions
                if norm_duration.dim() == 1:
                    norm_duration = norm_duration.unsqueeze(1)
                if norm_velocity.dim() == 1:
                    norm_velocity = norm_velocity.unsqueeze(1)
                
                # 合并特征
                base_features = torch.cat([norm_pitch, pos, norm_duration, norm_velocity], dim=1)
                
                # 保存基础特征
                graph.nodes['note'].data['feat'] = base_features
                
                # 使用模型的特征提取器
                if hasattr(model, 'self_supervised') and model.self_supervised is not None:
                    try:
                        logger.info("使用模型的特征提取器处理特征")
                        
                        if hasattr(model.self_supervised, 'nianzhi_predictor'):
                            if isinstance(model.self_supervised.nianzhi_predictor, dict):
                                feature_extractor = model.self_supervised.nianzhi_predictor.get('feature_extractor')
                                if feature_extractor is not None:
                                    # 处理特征
                                    batch_features = base_features.unsqueeze(0)
                                    projected_features = feature_extractor(batch_features)
                                    
                                    if projected_features.dim() == 3:
                                        projected_features = projected_features.squeeze(0)
                                    
                                    graph.nodes['note'].data['features'] = projected_features
                                    logger.info(f"特征提取成功，形状: {projected_features.shape}")
                    except Exception as e:
                        logger.warning(f"特征提取失败: {str(e)}")
                        logger.warning("使用简单投影方法")
                        
                        # 创建简单的投影层
                        projection = nn.Linear(4, 256, device=device)
                        projected_features = projection(base_features)
                        graph.nodes['note'].data['features'] = projected_features
                
            except Exception as e:
                logger.error(f"特征生成失败: {str(e)}")
                logger.error(traceback.format_exc())
        
        # 使用模型预测撚指
        model_prediction_successful = False
        predicted_features_dict = {}  # 存储每个音符的预测特征
        
        if model is not None and hasattr(model, 'self_supervised') and model.self_supervised is not None and use_self_supervised:
            try:
                logger.info("使用自监督学习模块预测撚指...")
                
                # 准备特征输入
                if 'features' in graph.nodes['note'].data:
                    node_features = graph.nodes['note'].data['features']
                    
                    # 确保特征形状正确
                    if node_features.dim() == 2:
                        features_batch = node_features.unsqueeze(0)
                    else:
                        features_batch = node_features
                    
                    # 调用自监督模块进行预测
                    predictions = model.self_supervised(features_batch, positions.long())
                    nianzhi_pred = predictions.get('nianzhi_pred')
                    
                    if nianzhi_pred is not None:
                        expected_length = graph.num_nodes('note')
                        
                        # 解析预测结果
                        if nianzhi_pred.shape[-1] >= 3:
                            nianzhi_prob = torch.sigmoid(nianzhi_pred[0, :expected_length, 0])
                            
                            # 记录预测概率
                            logger.info(f"撚指预测概率样本: {nianzhi_prob[:10].tolist()}")
                            
                            # 获取撚指类型
                            nianzhi_type_logits = nianzhi_pred[0, :expected_length, 1:]
                            if nianzhi_type_logits.shape[-1] >= 3:
                                pred_nianzhi_type = torch.argmax(nianzhi_type_logits[:, :3], dim=1) + 1
                            else:
                                second_dim_value = torch.sigmoid(nianzhi_pred[0, :expected_length, 1])
                                pred_nianzhi_type = torch.ones_like(second_dim_value)
                                pred_nianzhi_type[second_dim_value < 0.33] = 3
                                pred_nianzhi_type[second_dim_value > 0.66] = 1
                            
                            # 提取表现力特征
                            if nianzhi_pred.shape[-1] >= 5:
                                timing_features = nianzhi_pred[0, :expected_length, 3]  # 时间特征
                                velocity_features = nianzhi_pred[0, :expected_length, 4]  # 力度特征
                            else:
                                # 生成默认的表现力特征
                                timing_features = torch.ones(expected_length, device=device)
                                velocity_features = torch.ones(expected_length, device=device)
                            
                            # 使用自定义阈值
                            base_threshold = nianzhi_threshold
                            min_duration = 2.0
                            
                            # 跟踪添加的撚指数量
                            nianzhi_added = 0
                            
                            for i in range(expected_length):
                                pitch = pitches[i].item()
                                duration = durations[i].item()
                                position = positions[i].item()
                                
                                # 检查音符是否适合撚指
                                if (50 <= pitch <= 90 and duration >= min_duration):
                                    prob_value = nianzhi_prob[i].item()
                                    
                                    # 调整阈值
                                    duration_factor = min(duration / 10.0, 1.0)
                                    pitch_factor = min(abs(pitch - 70) / 20.0, 1.0)
                                    adjusted_threshold = base_threshold * (1.0 - 0.7 * duration_factor) * (1.0 - 0.5 * pitch_factor)
                                    
                                    # 如果强制生成，使用更低的阈值
                                    if force_nianzhi:
                                        adjusted_threshold *= 0.3
                                    
                                    if prob_value > adjusted_threshold:
                                        is_nianzhi[i] = 1.0
                                        nianzhi_type[i] = pred_nianzhi_type[i]
                                        
                                        # 存储预测的表现力特征
                                        predicted_features_dict[i] = {
                                            'timing': timing_features[i].item(),
                                            'velocity': velocity_features[i].item()
                                        }
                                        
                                        nianzhi_added += 1
                                        logger.info(f"添加自监督撚指: 音符={i}, 音高={pitch}, 概率={prob_value:.4f}, 阈值={adjusted_threshold:.4f}")
                            
                            logger.info(f"自监督预测添加了 {nianzhi_added} 个撚指")
                            if nianzhi_added > 0:
                                model_prediction_successful = True
                            elif force_nianzhi:
                                # 强制添加至少一个撚指
                                best_idx = torch.argmax(nianzhi_prob).item()
                                is_nianzhi[best_idx] = 1.0
                                nianzhi_type[best_idx] = pred_nianzhi_type[best_idx]
                                predicted_features_dict[best_idx] = {
                                    'timing': timing_features[best_idx].item(),
                                    'velocity': velocity_features[best_idx].item()
                                }
                                logger.info(f"强制添加自监督撚指: 音符={best_idx}, 音高={pitches[best_idx].item()}")
                                model_prediction_successful = True
                                
            except Exception as e:
                logger.error(f"自监督预测失败: {str(e)}")
                logger.error(traceback.format_exc())
        
        # 如果模型预测失败，使用规则方法
        if not model_prediction_successful:
            logger.info("使用规则方法添加撚指...")
            # 撚指音区范围 - 扩大范围
            min_pitch, max_pitch = 55, 85  # 适用音高范围
            nianzhi_probability = 0.25  # 提高基础概率
            nianzhi_count = 0
            min_duration = 2.0  # 最小持续时间要求
            
            # 逐个检查音符，按一定概率添加撚指
            for i in range(graph.num_nodes('note')):
                pitch = pitches[i].item()
                duration = durations[i].item()
                position = positions[i].item()
                
                # 只为中高音区和足够长的音符添加撚指
                if min_pitch <= pitch <= max_pitch and duration >= min_duration:
                    # 前后2个音符不应该是撚指（避免过于集中）
                    nearby_is_nianzhi = False
                    for j in range(max(0, i-2), min(i+3, graph.num_nodes('note'))):
                        if j != i and is_nianzhi[j] > 0.5:
                            nearby_is_nianzhi = True
                            break
                    
                    if not nearby_is_nianzhi:
                        # 特色音和持续时间对撚指概率的影响
                        special_notes = [54, 61, 66]  # 特色音
                        base_prob = nianzhi_probability
                        
                        # 增加某些特色音和长持续时间音符的撚指概率
                        if pitch in special_notes:
                            base_prob *= 1.8
                        if duration > 5.0:
                            base_prob *= 1.5
                            
                        # 为种子音符单独增加概率
                        if i < 8:  # 假设前8个是种子音符
                            base_prob *= 1.5
                            
                        # 对于中音区音符单独增加概率
                        if 60 <= pitch <= 72:
                            base_prob *= 1.3
                        
                        # 按概率添加撚指
                        if random.random() < base_prob:
                            is_nianzhi[i] = 1.0
                            
                            # 根据音高和持续时间确定撚指类型 (1=快速, 2=标准, 3=慢速)
                            if pitch > 75 and duration < 5.0:
                                # 高音+短音倾向于快速撚指
                                type_weights = [0.6, 0.3, 0.1]
                            elif pitch < 65 and duration > 7.0:
                                # 低音+长音倾向于慢速撚指
                                type_weights = [0.1, 0.3, 0.6]
                            else:
                                # 其他情况倾向于标准撚指
                                type_weights = [0.2, 0.6, 0.2]
                            
                            # 按权重随机选择撚指类型
                            rand_val = random.random()
                            if rand_val < type_weights[0]:
                                nianzhi_type[i] = 1  # 快速
                            elif rand_val < type_weights[0] + type_weights[1]:
                                nianzhi_type[i] = 2  # 标准
                            else:
                                nianzhi_type[i] = 3  # 慢速
                            
                            nianzhi_count += 1
                            logger.info(f"添加撚指 (规则): 音高={pitch}, 持续时间={duration:.2f}, 类型={int(nianzhi_type[i].item())}")
            
            logger.info(f"通过规则方式添加了 {nianzhi_count} 个撚指标记")
        
        # 强制至少添加一个撚指以进行测试
        if is_nianzhi.sum() == 0 and graph.num_nodes('note') >= 4:
            logger.info("强制添加撚指用于测试...")
            # 找到最佳音符添加撚指
            best_idx = -1
            best_score = -1
            
            for i in range(graph.num_nodes('note')):
                pitch = pitches[i].item()
                duration = durations[i].item()
                
                # 为中音区的音符计算分数
                if 55 <= pitch <= 85 and duration >= 1.0:
                    # 计算适合度分数
                    pitch_score = 1.0 - abs(pitch - 70) / 20.0  # 越接近70越好
                    duration_score = min(duration / 5.0, 1.0)  # 持续时间越长越好
                    score = pitch_score * 0.6 + duration_score * 0.4
                    
                    if score > best_score:
                        best_score = score
                        best_idx = i
            
            # 为最佳音符添加撚指
            if best_idx >= 0:
                is_nianzhi[best_idx] = 1.0
                nianzhi_type[best_idx] = 2  # 使用标准撚指
                pitch = pitches[best_idx].item()
                duration = durations[best_idx].item()
                logger.info(f"添加强制撚指: 音高={pitch}, 持续时间={duration:.2f}, 类型=2 (标准)")
        
        # 将撚指标记添加到图中
        graph.nodes['note'].data['is_nianzhi'] = is_nianzhi
        graph.nodes['note'].data['nianzhi_type'] = nianzhi_type
        
        # 生成撚指音符
        for i in range(graph.num_nodes('note')):
            if is_nianzhi[i] > 0.5:
                # 获取音符信息
                pitch = pitches[i].item()
                position = positions[i].item()
                duration = durations[i].item()
                velocity = graph.nodes['note'].data['velocity'][i].item() if 'velocity' in graph.nodes['note'].data else 80
                nianzhi_type_value = int(nianzhi_type[i].item())
                
                # 获取下一个音符的时间（如果有）
                next_note_time = None
                if i < graph.num_nodes('note') - 1:
                    next_note_time = positions[i + 1].item()
                
                # 获取预测的表现力特征
                predicted_features = None
                if i in predicted_features_dict:
                    predicted_features = [
                        predicted_features_dict[i]['timing'],
                        predicted_features_dict[i]['velocity']
                    ]
                
                # 生成撚指音符
                nianzhi_notes = generate_nianzhi_notes(
                    position=position,
                    duration=duration,
                    pitch=pitch,
                    velocity=velocity,
                    nianzhi_type=nianzhi_type_value,
                    next_note_time=next_note_time,
                    predicted_features=predicted_features
                )
                
                if nianzhi_notes:
                    # 更新原始音符的持续时间
                    graph.nodes['note'].data['duration'][i] = torch.tensor(
                        nianzhi_notes[-1].time + nianzhi_notes[-1].duration - position,
                        device=device
                    )
                    
                    # 添加撚指音符到图中
                    nianzhi_note_data = []
                    for note in nianzhi_notes:
                        # 记录每个音符的数据
                        nianzhi_note_data.append({
                            'time': note.time,
                            'duration': note.duration,
                            'pitch': note.pitch,
                            'velocity': note.velocity
                        })
                    
                    # 使用全局字典存储撚指数据，而不是图属性
                    try:
                        global NIANZHI_DATA_DICT
                        NIANZHI_DATA_DICT[i] = nianzhi_note_data
                        logger.info(f"成功将 {len(nianzhi_note_data)} 个撚指音符添加到音符 {i}")
                    except Exception as e:
                        logger.error(f"添加撚指音符数据时出错: {str(e)}")
                        logger.error(traceback.format_exc())
        
        return graph
        
    except Exception as e:
        logger.error(f"添加撚指失败: {str(e)}")
        logger.error(traceback.format_exc())
        return graph

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用增强模型生成南音音乐')
    parser.add_argument('--config', type=str, default='configs/enhanced_small.yaml',
                      help='配置文件路径')
    parser.add_argument('--model', type=str, default=None,
                      help='模型检查点路径')
    parser.add_argument('--output', type=str, default='generated',
                      help='输出目录')
    parser.add_argument('--tempo', type=int, default=20,
                      help='生成音乐的速度 (BPM)')
    parser.add_argument('--max_length', type=int, default=512,
                      help='生成的最大长度')
    parser.add_argument('--temperature', type=float, default=1.0,
                      help='生成时的温度参数')
    parser.add_argument('--seed', type=int, default=None,
                      help='随机种子')
    parser.add_argument('--no_ornaments', action='store_true',
                      help='不应用装饰音规则')
    parser.add_argument('--disable_ornament_processor', action='store_true',
                      help='禁用装饰音处理器（当参数维度不匹配时使用）')
    parser.add_argument('--force_nianzhi', action='store_true',
                      help='强制生成撚指音符')
    parser.add_argument('--nianzhi_threshold', type=float, default=0.03,
                      help='撚指生成的概率阈值，越小越容易生成撚指')
    parser.add_argument('--use_self_supervised', action='store_true',
                      help='使用自监督预测撚指')
    
    args = parser.parse_args()
    
    # 修改配置以禁用装饰音处理器
    if args.disable_ornament_processor:
        logger.info("禁用装饰音处理器")
        # 创建一个配置文件加载器
        config_loader = None
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_loader = yaml.safe_load(f)
            # 设置装饰音处理器为禁用状态
            if 'ornament_processor' not in config_loader:
                config_loader['ornament_processor'] = {}
            config_loader['ornament_processor']['enabled'] = False
            
            # 保存到临时配置文件
            temp_config_path = 'configs/temp_config.yaml'
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_loader, f)
            args.config = temp_config_path
            logger.info(f"临时配置文件已保存到 {temp_config_path}")
        except Exception as e:
            logger.error(f"修改配置文件失败: {str(e)}")
            # 继续使用原始配置
    
    # 输出自监督撚指生成设置
    if args.use_self_supervised:
        logger.info("启用自监督预测撚指")
    if args.force_nianzhi:
        logger.info("启用强制生成撚指")
    if args.nianzhi_threshold != 0.03:
        logger.info(f"自定义撚指阈值: {args.nianzhi_threshold}")
    
    # 设置环境变量，供generate_enhanced_music函数使用
    os.environ['FORCE_NIANZHI'] = 'TRUE' if args.force_nianzhi else 'FALSE'
    os.environ['NIANZHI_THRESHOLD'] = str(args.nianzhi_threshold)
    os.environ['USE_SELF_SUPERVISED'] = 'TRUE' if args.use_self_supervised else 'FALSE'
    
    # 生成音乐
    generated_graph, output_dir = generate_enhanced_music(
        model_path=args.model,
        config_path=args.config,
        output_dir=args.output,
        tempo=args.tempo,
        max_length=args.max_length,
        temperature=args.temperature,
        seed=args.seed,
        apply_ornaments=not args.no_ornaments
    )
    
    if output_dir:
        logger.info(f"音乐已生成并保存到: {output_dir}")
    else:
        logger.error("生成失败")

if __name__ == '__main__':
    main() 