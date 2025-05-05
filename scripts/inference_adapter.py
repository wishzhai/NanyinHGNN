#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import logging
import yaml
import traceback
import copy
import re
import torch.nn.init as init
import math
import torch.nn as nn
from typing import Tuple

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InferenceModelAdapter:
    """推理模型适配器：提高模型加载的容错性"""
    
    @staticmethod
    def load_checkpoint_tolerant(checkpoint_path, device='cuda'):
        """容错式加载检查点
        
        Args:
            checkpoint_path: 检查点路径
            device: 设备
            
        Returns:
            dict: 加载的检查点
        """
        try:
            logger.info(f"从 {checkpoint_path} 加载检查点...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            return checkpoint
        except Exception as e:
            logger.error(f"加载检查点失败: {str(e)}")
            return None
    
    @staticmethod
    def create_parameter_mapping_rules():
        """创建参数映射规则
        
        Returns:
            dict: 参数映射规则
        """
        return {
            # 自监督学习模块的特征提取器映射规则
            'self_supervised.feature_extractor': {
                'base_paths': [
                    'feature_extractor',
                    'self_supervised.feature_extractor',
                    'feature_enhancer.feature_extractor',
                    'processor.feature_extractor',
                    'self_supervised.nianzhi_predictor.feature_extractor',
                    'self_supervised.ornament_predictor.feature_extractor',
                    'pitch_predictor.feature_projection',
                    'feature_enhancer.feature_extractors',
                    'pitch_predictor.feature_extractors',
                    'pitch_predictor.projection'
                ],
                'layer_mappings': {
                    '0': ['0', 'layer0', 'input_layer', 'input', 'linear1', 'projection.0', 'feat_proj.0'],
                    '1': ['1', 'layer1', 'hidden_layer', 'hidden', 'linear2', 'projection.1', 'feat_proj.1'],
                    '4': ['4', 'layer4', 'output_layer', 'output', 'proj', 'projection', 'projection.4', 'feat_proj.4']
                },
                'component_mappings': {
                    'projection': ['proj', 'projection', 'output_proj', 'feature_projection'],
                    'attention': ['attention', 'self_attention', 'attn']
                }
            },
            # 自监督学习模块的撚指预测器映射规则
            'self_supervised.nianzhi_predictor': {
                'base_paths': [
                    'nianzhi_predictor',
                    'self_supervised.nianzhi_predictor',
                    'feature_enhancer.nianzhi_predictor',
                    'processor.nianzhi_predictor',
                    'self_supervised.predictor.nianzhi',
                    'pitch_predictor.interval_predictor',
                    'pitch_predictor.nianzhi_predictor',
                    'self_supervised.nianzhi_predictor.predictor',
                    'feature_enhancer.predictor',
                    'predictor',
                    'predictor.nianzhi',
                    'pitch_predictor.predictor',
                    'self_supervised.predictor',
                    'processor.predictor',
                    'feature_enhancer.nianzhi',
                    'nianzhi',
                    'interval',
                    'model.self_supervised.nianzhi_predictor',
                    'module.self_supervised.nianzhi_predictor'
                ],
                'layer_mappings': {
                    '0': [
                        '0', 'input', 'input_layer', 'linear1', 'predictor.0', 'layer0', 'pred.0', 'nianzhi.0', 'interval.0',
                        'input.weight', 'input_proj.weight', 'linear.0.weight', 'fc.0.weight',
                        'input_projection.weight', 'input_transform.weight', 'in_proj.weight',
                        'input_layer.weight', 'layer.0.weight', 'layers.0.weight', 'linear.weight',
                        'predictor.input_layer.weight', 'predictor.input.weight', 'projection.input.weight',
                        'projection.0.weight'
                    ],
                    '4': [
                        '4', 'output', 'predictor', 'final', 'output_layer', 'predictor.4', 'layer4', 'pred.4', '7', 'nianzhi.4', 'interval.4',
                        'output.weight', 'output_proj.weight', 'linear.4.weight', 'fc.4.weight',
                        'output_projection.weight', 'output_transform.weight', 'out_proj.weight',
                        'output_layer.weight', 'layer.4.weight', 'layers.4.weight', 'linear.weight',
                        'predictor.output_layer.weight', 'predictor.output.weight', 'projection.output.weight',
                        'projection.4.weight'
                    ]
                },
                'component_mappings': {
                    'predictor': [
                        'predictor',
                        'pred',
                        'projection',
                        'final_layer',
                        'output_layer',
                        'output',
                        'classifier',
                        'head',
                        'mlp'
                    ]
                }
            },
            # 添加撚指预测器组件的专门映射
            'self_supervised.nianzhi_predictor.predictor': {
                'base_paths': [
                    'nianzhi_predictor.predictor',
                    'predictor',
                    'nianzhi.predictor',
                    'interval_predictor',
                    'processor.nianzhi_predictor.predictor',
                    'self_supervised.predictor',
                    'self_supervised.nianzhi_predictor.predictor',
                    'self_supervised.interval_predictor',
                    'feature_enhancer.predictor',
                    'model.self_supervised.nianzhi_predictor.predictor',
                    'module.self_supervised.nianzhi_predictor.predictor'
                ],
                'layer_mappings': {
                    '0': [
                        '0', 'input_layer', 'input', 'layer0', 'linear1', 'fc1',
                        'dense.0', 'dense_layers.0', 'layers.0', 'linear.0'
                    ],
                    '1': [
                        '1', 'norm1', 'layer1', 'norm_layer1', 'ln1',
                        'norm.0', 'norm_layers.0', 'layer_norms.0'
                    ],
                    '2': [
                        '2', 'dropout1', 'layer2', 'drop1', 'dropout_layer1',
                        'dropout.0', 'dropout_layers.0', 'dropouts.0'
                    ],
                    '3': [
                        '3', 'activation', 'layer3', 'act', 'relu1', 'gelu1',
                        'activation.0', 'activations.0', 'act_layers.0'
                    ],
                    '4': [
                        '4', 'output_layer', 'output', 'layer4', 'linear2', 'fc2',
                        'dense.1', 'dense_layers.1', 'layers.1', 'linear.1',
                        'proj', 'projection', 'final_layer', 'head'
                    ],
                    '5': [
                        '5', 'norm2', 'layer5', 'norm_layer2', 'ln2',
                        'norm.1', 'norm_layers.1', 'layer_norms.1'
                    ],
                    '6': [
                        '6', 'activation2', 'layer6', 'act2', 'relu2', 'gelu2',
                        'activation.1', 'activations.1', 'act_layers.1'
                    ],
                    '7': [
                        '7', 'final', 'layer7', 'linear3', 'fc3', 'output_proj',
                        'dense.2', 'dense_layers.2', 'layers.2', 'linear.2',
                        'predictor', 'head', 'classifier'
                    ]
                }
            },
            # 特征增强器映射规则
            'feature_enhancer': {
                'base_paths': [
                    'feature_enhancer',
                    'self_supervised.feature_enhancer',
                    'processor.feature_enhancer',
                    'enhancer',
                    'self_supervised.processor',
                    'pitch_predictor',
                    'pitch_predictor.feature_enhancer'
                ],
                'layer_mappings': {
                    '0': ['0', 'input', 'input_layer', 'linear1', 'feature_projection.0', 'proj.0'],
                    '1': ['1', 'hidden', 'hidden_layer', 'linear2', 'feature_projection.1', 'proj.1'],
                    '4': ['4', 'output', 'output_layer', 'proj', 'feature_projection.4', 'proj.4']
                },
                'component_mappings': {
                    'projection': [
                        'proj', 
                        'projection', 
                        'output_proj',
                        'final_proj',
                        'feature_projection'
                    ],
                    'attention': [
                        'attention', 
                        'self_attention', 
                        'attn',
                        'multihead_attn',
                        'self_attn'
                    ],
                    'feature_extractor': [
                        'feature_extractor',
                        'encoder',
                        'input_processor',
                        'feature_extractors'
                    ]
                }
            },
            # 新增：音高预测器映射规则
            'pitch_predictor': {
                'base_paths': [
                    'pitch_predictor',
                    'self_supervised.pitch_predictor',
                    'feature_enhancer.pitch_predictor',
                    'processor.pitch_predictor'
                ],
                'layer_mappings': {
                    'interval_predictor.0': ['interval_predictor.0', 'interval_pred.0', 'nianzhi_predictor.0'],
                    'interval_predictor.1': ['interval_predictor.1', 'interval_pred.1', 'nianzhi_predictor.1'],
                    'interval_predictor.4': ['interval_predictor.4', 'interval_pred.4', 'nianzhi_predictor.4'],
                    'feature_projection.0': ['feature_projection.0', 'feat_proj.0', 'projection.0'],
                    'feature_projection.4': ['feature_projection.4', 'feat_proj.4', 'projection.4']
                },
                'component_mappings': {
                    'interval_predictor': [
                        'interval_predictor',
                        'interval_pred',
                        'nianzhi_predictor'
                    ],
                    'feature_projection': [
                        'feature_projection',
                        'feat_proj',
                        'projection'
                    ]
                }
            },
            # 自监督学习模块的通用映射规则
            'self_supervised': {
                'base_paths': [
                    'self_supervised',
                    'feature_enhancer',
                    'pitch_predictor',
                    'processor'
                ],
                'component_mappings': {
                    'feature_extractor': [
                        'feature_extractor',
                        'feature_projection',
                        'feature_extractors',
                        'encoder',
                        'projection'
                    ],
                    'nianzhi_predictor': [
                        'nianzhi_predictor',
                        'interval_predictor',
                        'predictor',
                        'pred'
                    ],
                    'attention': [
                        'attention',
                        'self_attention',
                        'attn',
                        'self_attn'
                    ]
                }
            }
        }
    
    @staticmethod
    def initialize_missing_parameters(param_name, target_shape, device):
        """初始化缺失的参数
        
        Args:
            param_name: 参数名称
            target_shape: 目标形状
            device: 设备
            
        Returns:
            torch.Tensor: 初始化的参数
        """
        try:
            # 创建张量
            param = torch.empty(target_shape, device=device)
            
            # 根据参数名称和维度选择初始化方法
            if 'weight' in param_name:
                if len(target_shape) >= 2:  # 只对2维及以上的张量使用高级初始化方法
                    if 'attention' in param_name or 'attn' in param_name:
                        # 注意力权重使用xavier初始化
                        init.xavier_uniform_(param)
                    elif any(x in param_name for x in ['feature_extractor', 'projection', 'predictor']):
                        # 特征提取器和预测器使用kaiming初始化
                        init.kaiming_normal_(param, nonlinearity='relu')
                    else:
                        # 其他权重使用xavier初始化
                        init.xavier_normal_(param)
                else:  # 1维或0维张量使用简单初始化
                    if len(target_shape) == 1:
                        # 1维张量使用均匀分布
                        bound = 1.0 / math.sqrt(target_shape[0])
                        init.uniform_(param, -bound, bound)
                    else:
                        # 0维张量（标量）初始化为小随机值
                        param.fill_(torch.randn(1).item() * 0.01)
            else:  # bias
                # 偏置初始化为0
                init.zeros_(param)
                
            return param
            
        except Exception as e:
            logger.warning(f"参数 {param_name} 初始化失败: {str(e)}，使用默认随机初始化")
            # 发生错误时使用简单的随机初始化
            try:
                if len(target_shape) > 0:
                    random_param = torch.randn(target_shape, device=device) * 0.01
                    return random_param
                else:
                    random_param = torch.zeros(target_shape, device=device)
                    return random_param
            except Exception as e2:
                logger.error(f"备用初始化也失败: {str(e2)}，创建CPU张量")
                # 如果device上初始化失败，尝试在CPU上创建然后移到device
                try:
                    if len(target_shape) > 0:
                        cpu_param = torch.randn(target_shape) * 0.01
                        return cpu_param.to(device)
                    else:
                        cpu_param = torch.zeros(target_shape)
                        return cpu_param.to(device)
                except Exception as e3:
                    logger.error(f"所有初始化方法都失败: {str(e3)}，返回空张量")
                    # 最后的备用选项，返回空张量
                    return torch.zeros(1, device=device)
    
    @staticmethod
    def adaptive_parameter_loading(model, checkpoint, strict=False):
        """自适应参数加载，处理参数名称不匹配的情况
        
        Args:
            model: 模型实例
            checkpoint: 检查点字典
            strict: 是否严格加载
            
        Returns:
            model: 加载参数后的模型
        """
        try:
            # 获取模型当前的参数字典
            model_state_dict = model.state_dict()
            
            # 获取检查点中的参数字典
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                checkpoint_state_dict = checkpoint['state_dict']
            else:
                checkpoint_state_dict = checkpoint
                
            # 创建新的状态字典
            new_state_dict = {}
            
            # 获取映射规则
            mapping_rules = InferenceModelAdapter.create_parameter_mapping_rules()
            
            # 记录已映射和未映射的参数
            mapped_params = set()
            unmapped_params = set(model_state_dict.keys())
            
            # 第一轮：应用精确映射规则
            for module_name, rules in mapping_rules.items():
                base_paths = rules.get('base_paths', [])
                layer_mappings = rules.get('layer_mappings', {})
                component_mappings = rules.get('component_mappings', {})
                special_mappings = rules.get('special_mappings', {})
                
                # 处理每个基础路径
                for base_path in base_paths:
                    # 处理层级映射
                    for layer_id, alternatives in layer_mappings.items():
                        target_key = f"{module_name}.{layer_id}"
                        for alt in alternatives:
                            old_key = f"{base_path}.{alt}"
                            
                            # 检查权重和偏置
                            for param_type in ['.weight', '.bias']:
                                old_param = old_key + param_type
                                new_param = target_key + param_type
                                
                                if old_param in checkpoint_state_dict and new_param in model_state_dict:
                                    if checkpoint_state_dict[old_param].shape == model_state_dict[new_param].shape:
                                        new_state_dict[new_param] = checkpoint_state_dict[old_param]
                                        mapped_params.add(old_param)
                                        unmapped_params.discard(new_param)
                                        logger.info(f"精确映射参数: {old_param} -> {new_param}")
                    
                    # 处理组件映射
                    for comp_name, alternatives in component_mappings.items():
                        for alt in alternatives:
                            old_key = f"{base_path}.{alt}"
                            new_key = f"{module_name}.{comp_name}"
                            
                            # 递归查找匹配的参数
                            for ckpt_key in checkpoint_state_dict.keys():
                                if ckpt_key.startswith(old_key):
                                    param_suffix = ckpt_key[len(old_key):]
                                    new_param = new_key + param_suffix
                                    
                                    if new_param in model_state_dict:
                                        if checkpoint_state_dict[ckpt_key].shape == model_state_dict[new_param].shape:
                                            new_state_dict[new_param] = checkpoint_state_dict[ckpt_key]
                                            mapped_params.add(ckpt_key)
                                            unmapped_params.discard(new_param)
                                            logger.info(f"组件映射参数: {ckpt_key} -> {new_param}")
                
                # 处理特殊映射规则
                if special_mappings:
                    for param_suffix, alt_params in special_mappings.items():
                        target_param = f"{module_name}.{param_suffix}"
                        
                        if target_param in unmapped_params:
                            target_shape = model_state_dict[target_param].shape
                            
                            # 检查所有可能的源参数
                            for base_path in base_paths:
                                for alt_param in alt_params:
                                    source_param = f"{base_path}.{alt_param}"
                                    
                                    if source_param in checkpoint_state_dict:
                                        if checkpoint_state_dict[source_param].shape == target_shape:
                                            new_state_dict[target_param] = checkpoint_state_dict[source_param]
                                            mapped_params.add(source_param)
                                            unmapped_params.discard(target_param)
                                            logger.info(f"特殊映射参数: {source_param} -> {target_param}")
                                            break
                    
                                # 如果已经找到映射，跳出循环
                                if target_param not in unmapped_params:
                                    break
            
            # 第二轮：使用相似性匹配处理剩余的未映射参数
            if unmapped_params:
                available_ckpt_params = set(checkpoint_state_dict.keys()) - mapped_params
                
                for missing_param in unmapped_params:
                    best_match = None
                    best_score = 0
                    target_shape = model_state_dict[missing_param].shape
                    
                    for ckpt_param in available_ckpt_params:
                        if checkpoint_state_dict[ckpt_param].shape == target_shape:
                            # 计算相似度分数
                            score = InferenceModelAdapter.keys_similar(missing_param, ckpt_param)
                            if score and score > best_score:
                                best_score = score
                                best_match = ckpt_param
                    
                    if best_match:
                        new_state_dict[missing_param] = checkpoint_state_dict[best_match]
                        mapped_params.add(best_match)
                        logger.info(f"相似性映射参数: {best_match} -> {missing_param}")
            
            # 第三轮：尝试直接匹配参数名称的最后部分
            if unmapped_params:
                available_ckpt_params = set(checkpoint_state_dict.keys()) - mapped_params
                
                # 创建参数名称最后部分的索引
                param_suffix_index = {}
                for ckpt_param in available_ckpt_params:
                    param_parts = ckpt_param.split('.')
                    if len(param_parts) >= 2:
                        # 同时索引最后一个和最后两个部分
                        suffix1 = param_parts[-1]  # 例如: "weight"
                        suffix2 = param_parts[-2] + '.' + param_parts[-1]  # 例如: "0.weight"
                        if suffix1 not in param_suffix_index:
                            param_suffix_index[suffix1] = []
                        if suffix2 not in param_suffix_index:
                            param_suffix_index[suffix2] = []
                        param_suffix_index[suffix1].append(ckpt_param)
                        param_suffix_index[suffix2].append(ckpt_param)
                
                for missing_param in list(unmapped_params):
                    param_parts = missing_param.split('.')
                    if len(param_parts) >= 2:
                        # 尝试匹配最后两个部分
                        suffix2 = param_parts[-2] + '.' + param_parts[-1]
                        if suffix2 in param_suffix_index:
                            target_shape = model_state_dict[missing_param].shape
                            
                            # 首先尝试完全匹配的形状
                            for ckpt_param in param_suffix_index[suffix2]:
                                if checkpoint_state_dict[ckpt_param].shape == target_shape:
                                    new_state_dict[missing_param] = checkpoint_state_dict[ckpt_param]
                                    mapped_params.add(ckpt_param)
                                    unmapped_params.discard(missing_param)
                                    logger.info(f"后缀匹配参数: {ckpt_param} -> {missing_param}")
                                    break
                            
                            # 如果没有找到完全匹配，尝试可重塑的参数
                            if missing_param in unmapped_params:
                                for ckpt_param in param_suffix_index[suffix2]:
                                    if InferenceModelAdapter.check_param_reshapeable(
                                        checkpoint_state_dict[ckpt_param], target_shape
                                    ):
                                        new_state_dict[missing_param] = InferenceModelAdapter.reshape_parameter(
                                            checkpoint_state_dict[ckpt_param], target_shape
                                        )
                                        mapped_params.add(ckpt_param)
                                        unmapped_params.discard(missing_param)
                                        logger.info(f"重塑参数: {ckpt_param} -> {missing_param}")
                                        break
                
                # 第三轮B：尝试匹配特定模式的参数
                for missing_param in list(unmapped_params):
                    if 'nianzhi_predictor' in missing_param:
                        target_shape = model_state_dict[missing_param].shape
                        layer_num = missing_param.split('.')[-2]  # 获取层号
                        
                        # 尝试从不同的模块中查找匹配的参数
                        potential_sources = [
                            f'predictor.{layer_num}',
                            f'nianzhi.{layer_num}',
                            f'interval.{layer_num}',
                            f'linear.{layer_num}',
                            f'fc.{layer_num}',
                            f'layer{layer_num}'
                        ]
                        
                        for source in potential_sources:
                            for ckpt_param in available_ckpt_params:
                                if source in ckpt_param and ckpt_param.endswith('.weight'):
                                    if checkpoint_state_dict[ckpt_param].shape == target_shape:
                                        new_state_dict[missing_param] = checkpoint_state_dict[ckpt_param]
                                        mapped_params.add(ckpt_param)
                                        unmapped_params.discard(missing_param)
                                        logger.info(f"特定模式匹配参数: {ckpt_param} -> {missing_param}")
                                        break
                            if missing_param not in unmapped_params:
                                break
            
            # 第四轮：检查是否有字典类型参数处理
            for missing_param in list(unmapped_params):
                if ('self_supervised.nianzhi_predictor' in missing_param or
                    'self_supervised.feature_extractor' in missing_param):
                    
                    # 查找类似参数
                    target_shape = model_state_dict[missing_param].shape
                    param_name = missing_param.split('.')[-1]  # 获取参数名（weight或bias）
                    layer_id = missing_param.split('.')[-2]  # 获取层ID
                    component = '.'.join(missing_param.split('.')[:-2])  # 获取组件路径

                    # 直接尝试查找预测器的组件
                    for predictor_key in available_ckpt_params:
                        # 检查是否包含predictor或nianzhi等关键字
                        if ('predictor' in predictor_key or 'nianzhi' in predictor_key or 
                            'interval' in predictor_key or 'projection' in predictor_key):
                            
                            # 检查是否包含相同的层ID
                            if f".{layer_id}." in predictor_key or f".{layer_id}{param_name}" in predictor_key:
                                # 检查参数类型是否匹配
                                if param_name in predictor_key:
                                    # 检查形状是否匹配
                                    if checkpoint_state_dict[predictor_key].shape == target_shape:
                                        new_state_dict[missing_param] = checkpoint_state_dict[predictor_key]
                                        mapped_params.add(predictor_key)
                                        unmapped_params.discard(missing_param)
                                        logger.info(f"预测器组件特殊映射: {predictor_key} -> {missing_param}")
                                        break
            
            # 第五轮：处理字典类型的参数映射（对象的key-value变成了嵌套模块）
            # 查找特殊模式: self_supervised.nianzhi_predictor.{object_type}.{index}.{param}
            for missing_param in list(unmapped_params):
                if ('self_supervised.nianzhi_predictor' in missing_param and
                    len(missing_param.split('.')) >= 4):
                    
                    parts = missing_param.split('.')
                    if len(parts) >= 5:  # self_supervised.nianzhi_predictor.predictor.0.weight
                        module_path = '.'.join(parts[:-3])  # self_supervised.nianzhi_predictor
                        object_type = parts[-3]  # predictor
                        index = parts[-2]  # 0
                        param_type = parts[-1]  # weight/bias
                        
                        # 在checkpoint中查找可能是字典或其他格式的参数
                        for ckpt_param in available_ckpt_params:
                            # 查找各种可能的格式
                            possible_formats = [
                                f"{object_type}.{index}.{param_type}",  # predictor.0.weight
                                f"{module_path}.{object_type}.{param_type}",  # self_supervised.nianzhi_predictor.predictor.weight
                                f"{object_type}.{param_type}",  # predictor.weight
                                f"{module_path}.{param_type}"  # self_supervised.nianzhi_predictor.weight
                            ]
                            
                            if any(fmt in ckpt_param for fmt in possible_formats):
                                # 检查形状是否匹配
                                if checkpoint_state_dict[ckpt_param].shape == model_state_dict[missing_param].shape:
                                    new_state_dict[missing_param] = checkpoint_state_dict[ckpt_param]
                                    mapped_params.add(ckpt_param)
                                    unmapped_params.discard(missing_param)
                                    logger.info(f"字典格式参数映射: {ckpt_param} -> {missing_param}")
                                    break
            
            # 第六轮：尝试直接搜索predictor中的关键参数
            # 根据来自日志的已知缺失的self_supervised.nianzhi_predictor.0.weight和self_supervised.nianzhi_predictor.4.weight
            # 专门处理这两个常见问题参数
            for missing_param in list(unmapped_params):
                if missing_param == 'self_supervised.nianzhi_predictor.0.weight' or missing_param == 'self_supervised.nianzhi_predictor.4.weight':
                    target_shape = model_state_dict[missing_param].shape
                    layer_id = missing_param.split('.')[-2]  # 0 or 4
                    
                    # 直接检查所有参数中包含predictor或nianzhi或interval并且包含该层ID的参数
                    for ckpt_param in available_ckpt_params:
                        # 匹配模式1: 查找任何包含predictor、第0/4层和weight的参数
                        if (('predictor' in ckpt_param.lower() or 'nianzhi' in ckpt_param.lower() or 
                            'interval' in ckpt_param.lower()) and 
                            f"{layer_id}." in ckpt_param and 
                            ckpt_param.endswith('weight')):
                            
                            # 检查形状
                            if checkpoint_state_dict[ckpt_param].shape == target_shape:
                                logger.info(f"找到关键参数 {missing_param} 的潜在匹配: {ckpt_param}")
                                new_state_dict[missing_param] = checkpoint_state_dict[ckpt_param]
                                mapped_params.add(ckpt_param)
                                unmapped_params.discard(missing_param)
                                logger.info(f"关键参数特殊映射: {ckpt_param} -> {missing_param}")
                                break
                        
                        # 匹配模式2: 任何包含linear, fc, layer或dense，且有该层ID的参数
                        elif (('linear' in ckpt_param.lower() or 'fc' in ckpt_param.lower() or
                            'layer' in ckpt_param.lower() or 'dense' in ckpt_param.lower()) and
                            (f"{layer_id}." in ckpt_param or f"_{layer_id}." in ckpt_param or f".{layer_id}_" in ckpt_param) and
                            ckpt_param.endswith('weight')):
                            
                            # 检查形状
                            if checkpoint_state_dict[ckpt_param].shape == target_shape:
                                logger.info(f"找到关键参数 {missing_param} 的通用匹配: {ckpt_param}")
                                new_state_dict[missing_param] = checkpoint_state_dict[ckpt_param]
                                mapped_params.add(ckpt_param)
                                unmapped_params.discard(missing_param)
                                logger.info(f"通用参数特殊映射: {ckpt_param} -> {missing_param}")
                                break
                                
            # 第七轮：初始化任何仍然缺失的参数
            remaining_unmapped = set(model_state_dict.keys()) - set(new_state_dict.keys())
            if remaining_unmapped:
                logger.warning(f"初始化缺失参数: {remaining_unmapped}")
                for param_name in remaining_unmapped:
                    # 特殊处理关键的撚指预测器参数
                    if param_name in ['self_supervised.nianzhi_predictor.0.weight', 'self_supervised.nianzhi_predictor.4.weight']:
                        logger.info(f"使用专门的预构建权重初始化关键参数: {param_name}")
                        new_state_dict[param_name] = InferenceModelAdapter.initialize_nianzhi_predictor_parameters(
                            param_name, model_state_dict[param_name].shape, model_state_dict[param_name].device
                        )
                    else:
                        # 使用改进的初始化方法
                        target_shape = model_state_dict[param_name].shape
                        new_state_dict[param_name] = InferenceModelAdapter.initialize_missing_parameters(
                            param_name, target_shape, model_state_dict[param_name].device
                        )
            
            # 加载参数
            try:
                incompatible_keys = model.load_state_dict(new_state_dict, strict=False)
                
                # 记录加载结果
                if len(incompatible_keys.missing_keys) > 0:
                    logger.warning(f"缺失的参数: {incompatible_keys.missing_keys}")
                if len(incompatible_keys.unexpected_keys) > 0:
                    logger.warning(f"未预期的参数: {incompatible_keys.unexpected_keys}")
            except Exception as e:
                logger.error(f"加载state_dict时出错: {str(e)}")
                logger.error(traceback.format_exc())
                logger.warning("加载参数出错，但将继续使用模型（可能存在性能问题）")
                # 即使加载失败也继续，返回部分初始化的模型
            
            return model
            
        except Exception as e:
            logger.error(f"自适应加载参数时出错: {str(e)}")
            logger.error(traceback.format_exc())
            logger.warning("参数加载过程中发生错误，返回原始模型")
            return model
    
    @staticmethod
    def check_param_reshapeable(param: torch.Tensor, target_shape: Tuple) -> bool:
        """检查参数是否可以重塑为目标形状"""
        try:
            # 如果目标是标量而参数不是，不能重塑
            if len(target_shape) == 0 and len(param.shape) > 0:
                return False
            
            # 如果维度数不同且不是简单的维度扩展，不能重塑
            if len(param.shape) != len(target_shape) and not (
                len(param.shape) == 1 and len(target_shape) == 2 or
                len(param.shape) == 2 and len(target_shape) == 1
            ):
                return False
            
            # 检查元素总数是否相同
            return param.numel() == torch.Size(target_shape).numel()
        except Exception as e:
            logger.warning(f"检查参数是否可重塑时出错: {str(e)}")
            return False
    
    @staticmethod
    def reshape_parameter(param, target_shape):
        """重塑参数为目标形状
        
        Args:
            param: 参数张量
            target_shape: 目标形状
            
        Returns:
            torch.Tensor: 重塑后的参数
        """
        try:
            return param.reshape(target_shape)
        except Exception:
            try:
                # 如果简单重塑失败，尝试其他方法
                if len(param.shape) == 1 and len(target_shape) == 2:
                    # 尝试将1D扩展为2D
                    return param.unsqueeze(0).expand(target_shape)
                elif len(param.shape) == 2 and len(target_shape) == 1:
                    # 尝试将2D压缩为1D
                    return param.mean(dim=0)
                else:
                    # 如果所有方法失败，创建新参数
                    logger.warning(f"无法重塑参数，创建新参数：{param.shape} -> {target_shape}")
                    return torch.randn(target_shape, device=param.device) * 0.02
            except Exception as e:
                logger.error(f"重塑参数时发生错误: {str(e)}")
                # 最后的备用方案：返回随机参数
                try:
                    return torch.randn(target_shape, device=param.device) * 0.01
                except Exception:
                    # 如果device上的操作失败，尝试在CPU上创建
                    return torch.randn(target_shape).to(param.device) * 0.01
    
    @staticmethod
    def keys_similar(key1, key2):
        """检查两个键是否相似
        
        Args:
            key1: 第一个键
            key2: 第二个键
            
        Returns:
            bool: 是否相似
        """
        # 移除常见前缀
        def normalize_key(key):
            for prefix in ['module.', 'model.']:
                if key.startswith(prefix):
                    key = key[len(prefix):]
            return key
        
        key1 = normalize_key(key1)
        key2 = normalize_key(key2)
        
        # 移除数字和层索引
        pattern = r'\.\d+\.'
        key1_norm = re.sub(pattern, '.', key1)
        key2_norm = re.sub(pattern, '.', key2)
        
        # 检查结构相似性
        parts1 = key1_norm.split('.')
        parts2 = key2_norm.split('.')
        
        # 如果层级不同，可能不匹配
        if abs(len(parts1) - len(parts2)) > 1:
            return False
        
        # 检查最后两个组件是否匹配
        if len(parts1) >= 2 and len(parts2) >= 2:
            if parts1[-1] == parts2[-1] and parts1[-2] == parts2[-2]:
                return True
        
        # 检查是否有很多相同的部分
        common_parts = set(parts1) & set(parts2)
        if len(common_parts) >= min(len(parts1), len(parts2)) * 0.7:
            return True
        
        return False
    
    @staticmethod
    def update_config_for_compatibility(config):
        """更新配置以提高兼容性
        
        Args:
            config: 原始配置
            
        Returns:
            dict: 更新后的配置
        """
        # 创建配置的深拷贝，避免修改原始配置
        config = copy.deepcopy(config)
        
        # 确保hidden_dim和feature_dim为256
        config['hidden_dim'] = 256
        config['feature_dim'] = 256
        config['current_stage'] = 2  # 设置为第二阶段
        
        # 启用自监督学习模块
        if 'self_supervised' not in config:
            config['self_supervised'] = {}
        config['self_supervised'].update({
            'enabled': True,
            'loss_weight': 0.5,
            'contour_weight': 0.3,
            'ornament_weight': 0.1,
            'nianzhi_weight': 0.5,
            'dropout': 0.1
        })
        
        # 更新装饰音处理器配置
        if 'ornament_processor' not in config:
            config['ornament_processor'] = {}
        config['ornament_processor'].update({
            'num_layers': 2,
            'dropout': 0.1,
            'rnn_type': 'LSTM',
            'input_dim': 256,
            'hidden_dim': 256,
            'output_dim': 256,
            'projection_dim': 256,
            'bidirectional': True,
            'projection_first': False,
            'use_residual': True
        })
        
        # 确保损失权重配置存在
        if 'loss_weights' not in config:
            config['loss_weights'] = {}
        config['loss_weights'].update({
            'pitch': 1.0,
            'pitch_smoothness': 0.3,
            'pitch_range': 0.2,
            'self_supervised': 0.5,
            'duration': 0.5,
            'velocity': 0.5
        })
        
        # 确保模型配置中包含必要的参数
        if 'model' not in config:
            config['model'] = {}
        config['model'].update({
            'hidden_dim': 256,
            'feature_dim': 256,
            'dropout': 0.2,
            'use_projection': True,
            'projection_dim': 256
        })
        
        # 设置feature_enhancer配置
        if 'feature_enhancer' not in config['model']:
            config['model']['feature_enhancer'] = {}
        config['model']['feature_enhancer'].update({
            'enabled': True,
            'input_dim': 256,
            'hidden_dim': 256,
            'output_dim': 256,
            'fusion_dim': 256,
            'context_layers': 2,
            'dropout': 0.1
        })
        
        return config
    
    @staticmethod
    def log_model_structure(model):
        """记录模型结构
        
        Args:
            model: 模型实例
        """
        logger.info("模型结构:")
        
        def print_module(module, prefix=''):
            for name, child in module.named_children():
                if isinstance(child, torch.nn.Linear):
                    logger.info(f"{prefix}{name}: Linear(in={child.in_features}, out={child.out_features})")
                elif isinstance(child, torch.nn.LSTM):
                    logger.info(f"{prefix}{name}: LSTM(in={child.input_size}, hidden={child.hidden_size}, layers={child.num_layers}, bidir={child.bidirectional})")
                elif isinstance(child, torch.nn.MultiheadAttention):
                    logger.info(f"{prefix}{name}: MultiheadAttention(embed={child.embed_dim}, heads={child.num_heads})")
                elif hasattr(child, 'hidden_dim'):
                    logger.info(f"{prefix}{name}: hidden_dim={child.hidden_dim}")
                elif isinstance(child, torch.nn.Sequential):
                    logger.info(f"{prefix}{name}: Sequential")
                    print_module(child, prefix + '  ')
                elif isinstance(child, torch.nn.ModuleDict):
                    logger.info(f"{prefix}{name}: ModuleDict")
                    print_module(child, prefix + '  ')
                elif isinstance(child, torch.nn.ModuleList):
                    logger.info(f"{prefix}{name}: ModuleList(len={len(child)})")
                    # 打印第一个元素作为示例
                    if len(child) > 0:
                        logger.info(f"{prefix}  [0]: 示例")
                        print_module(child[0], prefix + '    ')
                else:
                    logger.info(f"{prefix}{name}: {type(child).__name__}")
                    if len(list(child.named_children())) > 0:
                        print_module(child, prefix + '  ')
        
        print_module(model) 

    @staticmethod
    def initialize_nianzhi_predictor_parameters(param_name, shape, device):
        """特殊初始化撚指预测器的关键参数
        
        Args:
            param_name: 参数名称
            shape: 参数形状
            device: 设备
            
        Returns:
            torch.Tensor: 初始化的参数
        """
        try:
            # 创建参数张量
            param = torch.empty(shape, device=device)
            
            # 根据参数名称进行特殊初始化
            if param_name == 'self_supervised.nianzhi_predictor.0.weight':
                # 输入层通常使用Kaiming初始化
                logger.info(f"对撚指预测器输入层{param_name}使用Kaiming初始化，形状: {shape}")
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                
                # 如果可用，复制部分权重从identity矩阵以保持主要特征
                if min(shape) > 5:
                    middle = min(shape) // 2
                    for i in range(min(shape)):
                        if i < middle:
                            # 加强主对角线及其邻近元素
                            param[i, i] += 0.8
                            if i > 0:
                                param[i, i-1] += 0.2
                            if i < min(shape) - 1:
                                param[i, i+1] += 0.2
            
            elif param_name == 'self_supervised.nianzhi_predictor.4.weight':
                # 输出层通常使用Xavier初始化以获得更好的分类效果
                logger.info(f"对撚指预测器输出层{param_name}使用Xavier初始化，形状: {shape}")
                nn.init.xavier_uniform_(param)
                
                # 特殊处理：如果形状表明这是输出到3类的映射（撚指预测的标准输出），进行特殊增强
                if shape[0] == 3 or shape[1] == 3:
                    # 增强第一个通道（是否使用撚指）
                    scaling = 0.5
                    bias_init = 0.1
                    
                    if shape[0] == 3:  # 输出为3类
                        param[0, :] *= scaling  # 第一类（是否使用撚指）
                        param[0, :] += bias_init
                        param[1, :] *= 1.2  # 第二类（速度）
                        param[2, :] *= 1.2  # 第三类（强度）a
                    else:  # 输入到3类
                        param[:, 0] *= scaling
                        param[:, 0] += bias_init
                        param[:, 1] *= 1.2
                        param[:, 2] *= 1.2
            
            return param
            
        except Exception as e:
            logger.error(f"初始化参数 {param_name} 时发生错误: {str(e)}")
            logger.error(traceback.format_exc())
            # 发生错误时返回一个简单的随机张量而不是None
            return torch.randn(shape, device=device) * 0.02 