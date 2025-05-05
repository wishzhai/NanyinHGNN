import torch
import torch.nn as nn
import logging
import traceback
import dgl
from core.feature_enhancer import FeatureEnhancer
from core.self_supervised import SelfSupervisedModule, NianzhiFeatureExtractor

logger = logging.getLogger(__name__)

class ModelStructureAdapter:
    """模型结构适配器：确保训练模型与推理模型结构一致"""
    
    @staticmethod
    def align_model_structure(model, inference_config=None):
        """确保模型结构与推理期望一致
        
        Args:
            model: 要适配的模型
            inference_config: 推理配置
        
        Returns:
            model: 调整后的模型
        """
        try:
            logger.info("开始适配模型结构...")
            
            # 检查并修复特征增强器
            if not hasattr(model, 'feature_enhancer'):
                logger.info("添加缺失的feature_enhancer模块")
                model.feature_enhancer = FeatureEnhancer()
            
            # 确保自监督学习模块结构正确
            if hasattr(model, 'self_supervised') and model.self_supervised is not None:
                # 检查nianzhi_predictor结构
                nianzhi_predictor = model.self_supervised.nianzhi_predictor
                if 'feature_extractor' in nianzhi_predictor:
                    feature_extractor = nianzhi_predictor['feature_extractor']
                    # 确保特征提取器使用256维度
                    if hasattr(feature_extractor, 'hidden_dim'):
                        feature_extractor.hidden_dim = 256
                    # 确保存在projection属性
                    if not hasattr(feature_extractor, 'projection'):
                        logger.info("添加缺失的projection属性到feature_extractor")
                        feature_extractor.projection = nn.Sequential(
                            nn.Linear(4, 256),
                            nn.ReLU(),
                            nn.LayerNorm(256)
                        )
                else:
                    logger.warning("模型中缺少nianzhi_predictor.feature_extractor")
            else:
                logger.info("添加缺失的self_supervised模块")
                model.self_supervised = SelfSupervisedModule(model.config)
            
            # 确保所有相关模块的隐藏维度为256
            hidden_dim_modules = [
                'feature_enhancer',
                'pitch_predictor',
                'self_supervised'
            ]
            
            for module_name in hidden_dim_modules:
                if hasattr(model, module_name) and getattr(model, module_name) is not None:
                    module = getattr(model, module_name)
                    if hasattr(module, 'hidden_dim'):
                        if module.hidden_dim != 256:
                            logger.info(f"将{module_name}的hidden_dim从{module.hidden_dim}调整为256")
                            module.hidden_dim = 256
            
            logger.info("模型结构适配完成")
            return model
            
        except Exception as e:
            logger.error(f"模型结构适配失败: {str(e)}")
            logger.error(traceback.format_exc())
            return model
    
    @staticmethod
    def verify_model_structure(model):
        """验证模型结构是否与推理代码兼容
        
        Args:
            model: 要验证的模型
        
        Returns:
            bool: 结构是否兼容
        """
        try:
            required_modules = [
                'feature_enhancer',
                'self_supervised',
                'pitch_predictor'
            ]
            
            # 检查所有必需模块是否存在
            missing_modules = []
            for module_name in required_modules:
                if not hasattr(model, module_name) or getattr(model, module_name) is None:
                    missing_modules.append(module_name)
            
            if missing_modules:
                logger.warning(f"模型缺少以下模块: {missing_modules}")
                return False
            
            # 检查自监督学习模块结构
            if hasattr(model.self_supervised, 'nianzhi_predictor'):
                nianzhi_predictor = model.self_supervised.nianzhi_predictor
                if not isinstance(nianzhi_predictor, dict) or 'feature_extractor' not in nianzhi_predictor:
                    logger.warning("nianzhi_predictor结构不兼容")
                    return False
            else:
                logger.warning("缺少nianzhi_predictor")
                return False
            
            # 检查维度是否统一为256
            for module_name in required_modules:
                module = getattr(model, module_name)
                if hasattr(module, 'hidden_dim') and module.hidden_dim != 256:
                    logger.warning(f"{module_name}的hidden_dim={module.hidden_dim}，不是256")
                    return False
            
            logger.info("模型结构验证通过")
            return True
            
        except Exception as e:
            logger.error(f"模型结构验证失败: {str(e)}")
            logger.error(traceback.format_exc())
            return False 