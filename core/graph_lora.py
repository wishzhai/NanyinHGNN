import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)

class LoRALinear(nn.Module):
    """
    实现LoRA (Low-Rank Adaptation) 线性层
    """
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1, merge_weights=False):
        """
        参数:
            in_features: 输入特征维度
            out_features: 输出特征维度
            r: LoRA的秩 (rank)
            alpha: 缩放因子
            dropout: Dropout率
            merge_weights: 是否合并权重
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # 原始线性层 (冻结)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        for param in self.linear.parameters():
            param.requires_grad = False
            
        # LoRA低秩分解
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        
        # 初始化LoRA权重
        self.reset_lora_parameters()
        
        # 是否合并权重
        self.merged = False
        if merge_weights:
            self.merge_weights()
            
    def reset_lora_parameters(self):
        """初始化LoRA参数"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def merge_weights(self):
        """合并LoRA权重到原始权重"""
        if not self.merged:
            # W = W + BA
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
            
    def unmerge_weights(self):
        """分离LoRA权重"""
        if self.merged:
            # W = W - BA
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
            
    def to(self, device):
        """将模型移动到指定设备"""
        self.linear = self.linear.to(device)
        self.lora_A = self.lora_A.to(device)
        self.lora_B = self.lora_B.to(device)
        self.lora_dropout = self.lora_dropout.to(device)
        return self
        
    def forward(self, x):
        """前向传播"""
        if self.merged:
            return self.linear(x)
        else:
            # 原始线性变换 + LoRA路径
            return self.linear(x) + (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            
class LoRAMultiHeadAttention(nn.Module):
    """
    应用LoRA的多头注意力机制
    """
    def __init__(self, embed_dim, num_heads, original_module, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.original_module = original_module
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 冻结原始模块参数
        for param in self.original_module.parameters():
            param.requires_grad = False
            
        # 检查是否是MultiScaleHybridAttention
        self.is_multiscale = "MultiScale" in original_module.__class__.__name__
        
        if self.is_multiscale:
            # 为MultiScaleHybridAttention创建LoRA层
            # 这里我们只需要替换global_attention中的QKV投影
            self.q_lora = LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout)
            self.k_lora = LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout)
            self.v_lora = LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout)
            
            # 保存原始的QKV投影
            self.original_q_proj = self.original_module.q_proj
            self.original_k_proj = self.original_module.k_proj
            self.original_v_proj = self.original_module.v_proj
        else:
            # 为标准MultiheadAttention创建LoRA层
            self.q_lora = LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout)
            self.k_lora = LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout)
            self.v_lora = LoRALinear(embed_dim, embed_dim, r=r, alpha=alpha, dropout=dropout)
        
    def forward(self, x, attention_mask=None):
        """前向传播，使用原始模块但替换QKV计算"""
        if self.is_multiscale:
            # 保存原始QKV投影
            original_q_proj = self.original_module.q_proj
            original_k_proj = self.original_module.k_proj
            original_v_proj = self.original_module.v_proj
            
            # 替换为LoRA版本
            self.original_module.q_proj = self.q_lora
            self.original_module.k_proj = self.k_lora
            self.original_module.v_proj = self.v_lora
            
            # 调用原始模块的前向传播
            output, attn_weights = self.original_module(x, attention_mask)
            
            # 恢复原始投影
            self.original_module.q_proj = original_q_proj
            self.original_module.k_proj = original_k_proj
            self.original_module.v_proj = original_v_proj
            
            return output, attn_weights
        else:
            # 保存原始QKV投影
            original_q_proj = self.original_module.q_proj
            original_k_proj = self.original_module.k_proj
            original_v_proj = self.original_module.v_proj
            
            # 替换为LoRA版本
            self.original_module.q_proj = self.q_lora
            self.original_module.k_proj = self.k_lora
            self.original_module.v_proj = self.v_lora
            
            # 调用原始模块的前向传播
            output = self.original_module(x, attention_mask)
            
            # 恢复原始投影
            self.original_module.q_proj = original_q_proj
            self.original_module.k_proj = original_k_proj
            self.original_module.v_proj = original_v_proj
            
            return output

class GraphLoRAAdapter:
    """
    为图神经网络模型添加LoRA适配器
    """
    @staticmethod
    def apply_to_model(model, r=8, alpha=16, dropout=0.1, target_modules=None):
        """
        将LoRA应用到模型的指定模块
        
        参数:
            model: 要应用LoRA的模型
            r: LoRA的秩
            alpha: 缩放因子
            dropout: Dropout率
            target_modules: 要应用LoRA的目标模块列表，如果为None则应用到所有线性层
        """
        if target_modules is None:
            target_modules = ['query', 'key', 'value', 'out', 'fc', 'linear']
            
        # 记录转换的层
        converted_layers = 0
        
        # 递归遍历模型的所有模块
        for name, module in model.named_modules():
            # 跳过非叶子模块
            if len(list(module.children())) > 0 and not isinstance(module, nn.MultiheadAttention) and not "MultiScale" in module.__class__.__name__:
                continue
                
            # 检查是否是多头注意力层
            if isinstance(module, nn.MultiheadAttention) or "MultiScale" in module.__class__.__name__:
                try:
                    # 获取嵌入维度和头数
                    if hasattr(module, 'embed_dim'):
                        embed_dim = module.embed_dim
                    elif hasattr(module, 'hidden_dim'):
                        embed_dim = module.hidden_dim
                    else:
                        embed_dim = 512  # 默认值
                        
                    if hasattr(module, 'num_heads'):
                        num_heads = module.num_heads
                    else:
                        num_heads = 8  # 默认值
                    
                    # 检查是否有q_proj, k_proj, v_proj属性
                    if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                        # 直接应用LoRA到q_proj, k_proj, v_proj
                        if any(target in 'query' for target in target_modules):
                            old_q_proj = module.q_proj
                            module.q_proj = LoRALinear(
                                old_q_proj.in_features,
                                old_q_proj.out_features,
                                r=r,
                                alpha=alpha,
                                dropout=dropout
                            )
                            converted_layers += 1
                            logger.info(f"将LoRA应用到查询投影层: {name}.q_proj")
                            
                        if any(target in 'key' for target in target_modules):
                            old_k_proj = module.k_proj
                            module.k_proj = LoRALinear(
                                old_k_proj.in_features,
                                old_k_proj.out_features,
                                r=r,
                                alpha=alpha,
                                dropout=dropout
                            )
                            converted_layers += 1
                            logger.info(f"将LoRA应用到键投影层: {name}.k_proj")
                            
                        if any(target in 'value' for target in target_modules):
                            old_v_proj = module.v_proj
                            module.v_proj = LoRALinear(
                                old_v_proj.in_features,
                                old_v_proj.out_features,
                                r=r,
                                alpha=alpha,
                                dropout=dropout
                            )
                            converted_layers += 1
                            logger.info(f"将LoRA应用到值投影层: {name}.v_proj")
                    else:
                        logger.warning(f"注意力层 {name} 没有q_proj, k_proj, v_proj属性，跳过")
                except Exception as e:
                    logger.error(f"应用LoRA到注意力层 {name} 时出错: {str(e)}")
                continue
                
            # 检查是否是线性层且名称匹配目标模块
            if isinstance(module, nn.Linear):
                try:
                    # 检查模块名称是否包含目标关键字
                    should_apply = False
                    for target in target_modules:
                        if target in name.lower():
                            should_apply = True
                            break
                    
                    if should_apply:
                        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                        module_name = name.rsplit('.', 1)[1] if '.' in name else name
                        parent = model
                        
                        # 获取父模块
                        if parent_name:
                            for part in parent_name.split('.'):
                                parent = getattr(parent, part)
                        
                        # 替换为LoRA版本
                        lora_layer = LoRALinear(
                            module.in_features,
                            module.out_features,
                            r=r,
                            alpha=alpha,
                            dropout=dropout
                        )
                        
                        # 复制权重和偏置
                        lora_layer.linear.weight.data.copy_(module.weight.data)
                        if module.bias is not None and hasattr(lora_layer.linear, 'bias'):
                            lora_layer.linear.bias.data.copy_(module.bias.data)
                        
                        # 设置到父模块
                        setattr(parent, module_name, lora_layer)
                        
                        converted_layers += 1
                        logger.info(f"将LoRA应用到线性层: {name}")
                except Exception as e:
                    logger.error(f"应用LoRA到线性层 {name} 时出错: {str(e)}")
                
        return converted_layers 