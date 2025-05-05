import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math
import traceback  # 添加traceback导入
import random

logger = logging.getLogger(__name__)

class SelfSupervisedModule(nn.Module):
    """南音自监督学习模块，包含撚指预测和音乐特征预测"""
    
    def __init__(self, config):
        super().__init__()
        
        # 正确处理dropout参数
        self.dropout = config.get('dropout', 0.1)
        if isinstance(self.dropout, dict):
            self.dropout = self.dropout.get('feat', 0.1)
        
        # 获取其他配置参数
        self.hidden_dim = config.get('hidden_dim', 256)
        self.feature_dim = config.get('feature_dim', 256)
        self.num_heads = config.get('num_heads', 4)
        self.num_layers = config.get('num_layers', 2)
        self.ffn_dim = config.get('ffn_dim', 512)
        
        # 创建特征提取器和注意力层
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # 撚指预测器
        self.nianzhi_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout),
                nn.ReLU(),
            nn.Linear(self.hidden_dim, 3)  # 预测撚指概率和类型
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, x, positions=None):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 自注意力处理
        if positions is not None:
            # 生成位置编码
            pos_encoding = self._get_position_encoding(positions, features.shape[-1])
            features = features + pos_encoding
            
        # 应用自注意力
        if features.dim() == 2:
            features = features.unsqueeze(0)
        features = features.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        attended_features, _ = self.attention(features, features, features)
        attended_features = attended_features.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 预测撚指
        nianzhi_pred = self.nianzhi_predictor(attended_features)
        
        return {
            'nianzhi_pred': nianzhi_pred,
            'features': attended_features
        }
        
    def _get_position_encoding(self, positions, dim):
        # 生成相对位置编码
        max_len = positions.max().item() + 1
        position_enc = torch.zeros(max_len, dim, device=positions.device)
        position = torch.arange(0, max_len, device=positions.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=positions.device).float() * (-math.log(10000.0) / dim))
        position_enc[:, 0::2] = torch.sin(position.float() * div_term)
        position_enc[:, 1::2] = torch.cos(position.float() * div_term)
        return position_enc[positions]
    
    def compute_ornament_loss(self, pred, target, weight=None):
        """计算装饰音预测损失
        
        Args:
            pred: 预测值 [batch_size, seq_len, 2] 或 [seq_len, 2]
            target: 目标值 [batch_size, seq_len, 2] 或 [seq_len, 2]
            weight: 样本权重 [batch_size, seq_len] 或 [seq_len]，用于掩码
        """
        try:
            device = pred.device if isinstance(pred, torch.Tensor) else 'cuda'
            
            # 确保输入是张量
            if not isinstance(pred, torch.Tensor):
                return torch.tensor(0.01, device=device, requires_grad=True)
            if not isinstance(target, torch.Tensor):
                return torch.tensor(0.01, device=device, requires_grad=True)
            
            # 确保预测值和目标值具有相同的维度
            if pred.dim() != target.dim():
                if pred.dim() == 2 and target.dim() == 3:
                    pred = pred.unsqueeze(0)
                elif target.dim() == 2 and pred.dim() == 3:
                    target = target.unsqueeze(0)
            
            # 确保序列长度匹配
            if pred.shape != target.shape:
                logger.warning(f"形状仍不匹配，处理序列长度 - 预测值: {pred.shape}, 目标值: {target.shape}")
                
                # 处理序列长度不匹配
                if pred.dim() == target.dim():
                    # 获取序列长度维度索引（通常是倒数第二个维度）
                    seq_dim = -2 if pred.dim() > 1 else 0
                    
                    if pred.shape[seq_dim] != target.shape[seq_dim]:
                        min_len = min(pred.shape[seq_dim], target.shape[seq_dim])
                        logger.warning(f"序列长度不匹配: 预测值={pred.shape[seq_dim]}, 目标值={target.shape[seq_dim]}, 截断到{min_len}")
                        
                        # 截断预测值
                        if pred.dim() == 3:
                            if pred.shape[1] > min_len:
                                pred = pred[:, :min_len, :]
                        elif pred.dim() == 2:
                            if pred.shape[0] > min_len:
                                pred = pred[:min_len, :]
                        
                        # 截断目标值
                        if target.dim() == 3:
                            if target.shape[1] > min_len:
                                target = target[:, :min_len, :]
                        elif target.dim() == 2:
                            if target.shape[0] > min_len:
                                target = target[:min_len, :]
                        
                        logger.info(f"截断后 - 预测值: {pred.shape}, 目标值: {target.shape}")
                
                # 处理特征维度不匹配
                if pred.shape != target.shape and pred.dim() == target.dim():
                    feat_dim = -1  # 最后一个维度通常是特征维度
                    
                    if pred.shape[feat_dim] != target.shape[feat_dim]:
                        logger.warning(f"特征维度不匹配: 预测值={pred.shape[feat_dim]}, 目标值={target.shape[feat_dim]}")
                        
                        # 取第一个特征计算损失
                        logger.info("只使用第一个特征计算损失")
                        pred_first = pred[..., 0:1]  # 保持维度
                        target_first = target[..., 0:1]  # 保持维度
                        return F.binary_cross_entropy_with_logits(pred_first.float(), target_first.float())
            
            # 确保张量是浮点型
            pred = pred.float()
            target = target.float()
            
            # 应用标签平滑
            eps = 0.1
            target = target * (1 - eps) + eps / 2
            
            # 确保权重维度正确并转换为浮点类型
            if weight is not None:
                # 检查weight是否为张量
                if isinstance(weight, torch.Tensor):
                    weight = weight.float()  # 确保权重是浮点类型
                    if weight.dim() != target.dim() - 1:
                        weight = weight.unsqueeze(0)
                    # 添加动态权重调整（使用浮点数）
                    weight = weight * (1.0 + 0.2 * torch.rand_like(weight, dtype=torch.float))
                else:
                    # 如果weight不是张量而是标量，创建一个全1的张量
                    logger.warning(f"权重不是张量而是 {type(weight)}，创建全1权重")
                    if target.dim() == 3:
                        weight = torch.ones(target.shape[0], target.shape[1], device=device)
                    else:  # target.dim() == 2
                        weight = torch.ones(target.shape[0], device=device)
            
            # 计算存在性损失（使用focal loss）
            alpha = 0.25
            gamma = 2.0
            p = torch.sigmoid(pred[..., 0])
            ce_loss = F.binary_cross_entropy_with_logits(
                pred[..., 0],
                target[..., 0],
                reduction='none'
            )
            p_t = p * target[..., 0] + (1 - p) * (1 - target[..., 0])
            focal_loss = ce_loss * ((1 - p_t) ** gamma)
            
            if weight is not None:
                focal_loss = focal_loss * weight
                focal_loss = focal_loss.sum() / (weight.sum() + 1e-8)
            else:
                focal_loss = focal_loss.mean()
            
            # 计算位置损失（使用Huber损失）
            ornament_mask = (target[..., 0] > 0.5).float()  # 转换为浮点类型
            if weight is not None:
                ornament_mask = ornament_mask * weight
            
            if ornament_mask.sum() > 0:
                position_loss = F.smooth_l1_loss(
                    pred[..., 1][ornament_mask > 0],
                    target[..., 1][ornament_mask > 0],
                    beta=0.1,
                    reduction='mean'
                )
            else:
                position_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            # 添加L2正则化
            l2_reg = 0.01 * (pred ** 2).mean()
            
            # 组合损失并添加正则项
            total_loss = (
                focal_loss + 
                0.5 * position_loss + 
                l2_reg + 
                1e-6
            )
            
            # 确保损失需要梯度
            if not total_loss.requires_grad:
                total_loss.requires_grad_(True)
            
            return total_loss
            
        except Exception as e:
            logger.error(f"计算装饰音损失时出错: {str(e)}")
            logger.error(traceback.format_exc())
            return torch.tensor(0.01, device=device if 'device' in locals() else 'cuda', requires_grad=True)
    
    def compute_nianzhi_loss(self, features, targets=None, graphs=None):
        """计算撚指预测的损失"""
        result = {}
        
        try:
            # 检查输入特征的维度
            logger.info(f"compute_nianzhi_loss输入特征形状: {features.shape}")
            
            # 增强特征处理逻辑 - 判断特征类型
            is_raw_features = (features.dim() == 2 or (features.dim() == 3 and features.shape[-1] != 3))
            
            # 检查特征是否已经是预测结果（形状为[batch, seq, 3]）
            if not is_raw_features and features.dim() == 3 and features.shape[-1] == 3:
                logger.warning("检测到输入特征可能已经是预测结果，跳过特征提取步骤")
                pred = features  # 直接使用输入作为预测结果
            else:
                # 正常的特征提取和预测流程
                logger.info("输入为原始特征，进行特征提取和预测")
                extracted_features = self.nianzhi_predictor(features)
                logger.info(f"从原始特征生成的预测形状: {extracted_features.shape}")
            
            if targets is not None:
                # 确保形状匹配
                if extracted_features.shape != targets.shape:
                    logger.warning(f"形状不匹配: pred {extracted_features.shape}, targets {targets.shape}")
                    # 尝试调整形状
                    if extracted_features.dim() == 3 and targets.dim() == 2:
                        targets = targets.unsqueeze(0)
                    # 为2D预测添加批次维度
                    elif extracted_features.dim() == 2 and targets.dim() == 3:
                        extracted_features = extracted_features.unsqueeze(0)
                
                # 记录形状信息    
                logger.info(f"撚指预测形状: {extracted_features.shape}, 目标形状: {targets.shape}")
                
                # 优化焦点损失参数 - 位置预测
                alpha = 0.3  # 降低alpha以减少对难样本的惩罚
                gamma = 1.5  # 降低gamma以平滑损失曲线
                p = torch.sigmoid(extracted_features[..., 0])
                ce_loss = F.binary_cross_entropy_with_logits(
                    extracted_features[..., 0],
                    targets[..., 0],
                    reduction='none'
                )
                p_t = p * targets[..., 0] + (1 - p) * (1 - targets[..., 0])
                position_loss = (ce_loss * ((1 - p_t) ** gamma)).mean()
                
                # 优化速度和强度的MSE损失
                speed_loss = F.smooth_l1_loss(extracted_features[..., 1], targets[..., 1])  # 使用smooth L1损失
                intensity_loss = F.smooth_l1_loss(extracted_features[..., 2], targets[..., 2])  # 使用smooth L1损失
                
                # 对比学习损失设置为0，避免之前的计算问题
                contrastive_loss = torch.tensor(0.0, device=extracted_features.device)
                
                # 优化损失权重
                total_loss = (
                    0.5 * position_loss +   # 降低位置损失权重
                    0.3 * speed_loss +      # 增加速度损失权重
                    0.2 * intensity_loss    # 保持强度损失权重
                )
                
                # 应用全局损失缩放
                loss_scale = 0.8  # 整体降低损失值
                total_loss = total_loss * loss_scale
                
                result.update({
                    'nianzhi_loss': total_loss,
                    'nianzhi_position_loss': position_loss,
                    'nianzhi_speed_loss': speed_loss,
                    'nianzhi_intensity_loss': intensity_loss,
                    'nianzhi_contrastive_loss': contrastive_loss
                })
                
                # 记录详细的损失信息
                logger.info(
                    f"撚指损失: 总计 {total_loss.item():.4f}, "
                    f"位置 {position_loss.item():.4f}, "
                    f"速度 {speed_loss.item():.4f}, "
                    f"强度 {intensity_loss.item():.4f}"
                )
            
            # 存储预测结果
            result['nianzhi_pred'] = extracted_features
            
        except Exception as e:
            logger.error(f"计算撚指损失出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回零损失
            device = features.device if isinstance(features, torch.Tensor) else 'cuda'
            result = {
                'nianzhi_loss': torch.tensor(0.0, device=device, requires_grad=True),
                'nianzhi_position_loss': torch.tensor(0.0, device=device),
                'nianzhi_speed_loss': torch.tensor(0.0, device=device),
                'nianzhi_intensity_loss': torch.tensor(0.0, device=device),
                'nianzhi_contrastive_loss': torch.tensor(0.0, device=device)
            }
            
        return result
    
    def compute_melody_contour_loss(self, pred, target):
        """计算旋律轮廓损失"""
        try:
            if not (isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)):
                return torch.tensor(0.0, device=pred.device if isinstance(pred, torch.Tensor) else 'cuda')
            
            # 级进概率损失
            stepwise_prob_loss = F.binary_cross_entropy_with_logits(
                pred[..., 0].float(), 
                target[..., 0].float()
            )
            
            # 级进方向损失
            direction_loss = F.binary_cross_entropy_with_logits(
                pred[..., 1].float(), 
                target[..., 1].float()
            )
            
            # 音程大小损失
            interval_loss = F.l1_loss(
                pred[..., 2].float(), 
                target[..., 2].float()
            )
            
            # 持续性损失
            continuity_loss = F.binary_cross_entropy_with_logits(
                pred[..., 3].float(), 
                target[..., 3].float()
            )
            
            # 组合损失
            total_loss = (0.3 * stepwise_prob_loss + 
                         0.3 * direction_loss + 
                         0.2 * interval_loss + 
                         0.2 * continuity_loss)
                         
            return torch.abs(total_loss)  # 确保返回非负值
        
        except Exception as e:
            logger.error(f"计算旋律轮廓损失时出错: {str(e)}")
            return torch.tensor(0.0, device=pred.device if isinstance(pred, torch.Tensor) else 'cuda')
    
    def compute_rhythm_pattern_loss(self, pred, target):
        """计算节奏模式损失"""
        try:
            if not (isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor)):
                return torch.tensor(0.0, device=pred.device if isinstance(pred, torch.Tensor) else 'cuda')
            
            # 使用 L1 损失代替 MSE
            density_loss = F.l1_loss(
                pred[..., 0].float(), 
                target[..., 0].float()
            )
            
            duration_loss = F.l1_loss(
                pred[..., 1].float(), 
                target[..., 1].float()
            )
            
            accent_loss = F.binary_cross_entropy_with_logits(
                pred[..., 2].float(), 
                target[..., 2].float()
            )
            
            total_loss = density_loss + duration_loss + 0.5 * accent_loss
            return torch.abs(total_loss)  # 确保返回非负值
        
        except Exception as e:
            logger.error(f"计算节奏模式损失时出错: {str(e)}")
            return torch.tensor(0.0, device=pred.device if isinstance(pred, torch.Tensor) else 'cuda')
    
    def compute_loss(self, predictions, targets):
        """计算自监督学习总损失
            
        Returns:
            dict: 包含total_loss和各组件损失的字典
        """
        device = next(iter(predictions.values())).device if isinstance(predictions, dict) and predictions else 'cuda'
        
        try:
            if not isinstance(predictions, dict) or not isinstance(targets, dict):
                logger.warning(f"预测值或目标值不是字典类型")
                result = {'total_loss': torch.tensor(0.01, device=device, requires_grad=True)}
                return result
            
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            loss_count = 0
            
            # 清空之前的损失记录
            result = {}
            
            # 获取权重
            ornament_weight = targets.get('ornament_weight', 0.0)
            nianzhi_weight = targets.get('nianzhi_weight', 0.5)
            contour_weight = targets.get('contour_weight', 0.3)
            
            # 设置装饰音权重为0，因为数据集中没有装饰音
            ornament_weight = 0.0
            
            # 计算撚指损失
            if nianzhi_weight > 0 and 'nianzhi_pred' in predictions and 'nianzhi_target' in targets:
                nianzhi_pred = predictions['nianzhi_pred']
                nianzhi_target = targets['nianzhi_target']
                
                # 记录转换前的形状
                logger.info(f"撚指损失计算前 - 预测值形状: {nianzhi_pred.shape if isinstance(nianzhi_pred, torch.Tensor) else 'None'}, "
                           f"目标值形状: {nianzhi_target.shape if isinstance(nianzhi_target, torch.Tensor) else 'None'}")
                
                # 检查是否有原始特征可用，增强特征获取逻辑
                original_features = predictions.get('node_features', None)
                if original_features is not None and isinstance(original_features, torch.Tensor):
                    logger.info(f"✅ 使用原始节点特征计算撚指损失，特征形状: {original_features.shape}")
                    use_original_features = True
                    
                    # 检查原始特征的维度与类型
                    if original_features.dim() == 2:
                        logger.info("原始特征是2D形式 [seq_len, hidden_dim]")
                    elif original_features.dim() == 3:
                        logger.info(f"原始特征是3D形式 [batch, seq_len, hidden_dim], 维度: {original_features.shape}")
                else:
                    logger.warning("❌ 找不到原始节点特征，将直接使用撚指预测结果")
                    use_original_features = False
                
                # 检查并修复形状不匹配
                if isinstance(nianzhi_pred, torch.Tensor) and isinstance(nianzhi_target, torch.Tensor):
                    # 修复目标维度
                    if nianzhi_target.dim() == 3 and nianzhi_target.shape[1] == 3 and nianzhi_target.shape[2] == 3:
                        logger.warning(f"检测到错误的目标维度: {nianzhi_target.shape}")
                        nianzhi_target = nianzhi_target[:, :, 0]  # [974, 3, 3] -> [974, 3]
                        logger.info(f"修复后目标形状: {nianzhi_target.shape}")
                    
                    # 如果预测是3D而目标是2D，添加批次维度
                    if nianzhi_pred.dim() == 3 and nianzhi_target.dim() == 2:
                        if nianzhi_pred.shape[1:] == nianzhi_target.shape:
                            logger.info("添加批次维度到目标")
                            nianzhi_target = nianzhi_target.unsqueeze(0)
                            logger.info(f"调整后 - 预测值: {nianzhi_pred.shape}, 目标值: {nianzhi_target.shape}")
                
                try:
                    # 计算撚指损失，优先使用原始特征
                    input_features = original_features if use_original_features else nianzhi_pred
                    logger.info(f"最终使用的输入特征形状: {input_features.shape}")
                    
                    nianzhi_loss_dict = self.compute_nianzhi_loss(
                        input_features,
                        nianzhi_target
                    )
                    
                    # 安全获取nianzhi_loss
                    if isinstance(nianzhi_loss_dict, dict) and 'nianzhi_loss' in nianzhi_loss_dict:
                        nianzhi_loss = nianzhi_loss_dict['nianzhi_loss']
                        
                        if torch.isfinite(nianzhi_loss) and nianzhi_loss > 0:
                            weighted_loss = nianzhi_weight * nianzhi_loss
                            total_loss = total_loss + weighted_loss
                            loss_count += 1
                            logger.info(f"撚指损失: {nianzhi_loss.item():.4f}, 权重: {nianzhi_weight}, 加权损失: {weighted_loss.item():.4f}")
                        
                        # 记录各个子损失到结果字典
                        for key, value in nianzhi_loss_dict.items():
                            if key.endswith('_loss') and torch.is_tensor(value):
                                result[key] = value
                                logger.info(f"{key}: {value.item():.4f}")
                    else:
                        logger.warning(f"撚指损失计算返回结果格式不正确: {type(nianzhi_loss_dict)}")
                except Exception as e:
                    logger.error(f"处理撚指损失时出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 计算旋律轮廓损失
            if contour_weight > 0 and 'features' in predictions and 'contour_target' in targets:
                try:
                    contour_loss = self.compute_melody_contour_loss(
                        predictions['features'],
                        targets['contour_target']
                    )
                    
                    if torch.isfinite(contour_loss) and contour_loss > 0:
                        total_loss = total_loss + contour_weight * contour_loss
                        loss_count += 1
                        logger.info(f"旋律轮廓损失: {contour_loss.item():.4f}")
                        # 记录旋律轮廓损失
                        result['contour_loss'] = contour_loss
                except Exception as e:
                    logger.error(f"计算旋律轮廓损失时出错: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # 计算装饰音损失（权重设置为0，不会影响总损失）
            if ornament_weight > 0 and 'features' in predictions and 'ornament_target' in targets:
                try:
                    ornament_loss = self.compute_ornament_loss(
                        predictions['features'],
                        targets['ornament_target'],
                        targets.get('ornament_weight', None)
                    )
                    
                    if torch.isfinite(ornament_loss) and ornament_loss > 0:
                        total_loss = total_loss + ornament_weight * ornament_loss
                        loss_count += 1
                        logger.info(f"装饰音损失: {ornament_loss.item():.4f}")
                        # 记录装饰音损失
                        result['ornament_loss'] = ornament_loss
                except Exception as e:
                    logger.error(f"计算装饰音损失时出错: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.info("装饰音损失未计算（权重为0或缺少相关数据）")
            
            # 如果没有有效的损失，返回一个小的正值
            if loss_count == 0:
                result['total_loss'] = torch.tensor(0.01, device=device, requires_grad=True)
                return result
            
            # 记录总损失
            result['total_loss'] = total_loss
            
            # 返回结果字典，包含总损失和各组件损失
            return result
        except Exception as e:
            logger.error(f"计算自监督损失时出错: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回一个包含默认损失的字典
            return {'total_loss': torch.tensor(0.01, device=device, requires_grad=True)} 

class NianzhiFeatureExtractor(nn.Module):
    """南音撚指特征提取器，增强版
    提取与撚指相关的音符特征，用于预测最适合撚指的音符位置和类型。
    支持不同类型的撚指（快速、标准、慢速）特征提取，提高撚指的表现力。
    """
    def __init__(self, config):
        super().__init__()
        self.input_dim = 4  # pitch, position, duration, velocity 
        self.hidden_dim = 256  # 改为固定值256，与注意力层匹配
        
        # 动态检测并处理输入维度
        self.input_projection = None
        
        # 特征提取网络 - 更复杂的多层网络
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 兼容性考虑，保留原有投影层
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # 撚指类型分类网络
        self.type_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3)  # 3种撚指类型: 快速, 标准, 慢速
        )
        
        # 撚指适用度评分网络
        self.suitability_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # 特征增强层
        self.feature_enhancer = nn.Sequential(
            nn.Linear(self.hidden_dim + 4, self.hidden_dim),  # 增加4维(类型+适用度)特征
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def forward(self, x):
        """增强的前向传播，提供撚指类型和适用度信息
        
        Args:
            x (torch.Tensor): [batch_size, seq_len, input_dim] 或 [batch_size, seq_len, hidden_dim]
        Returns:
            torch.Tensor: [batch_size, seq_len, hidden_dim]，增强的特征表示
        """
        try:
            # 获取特征维度
            feat_dim = x.size(-1)
            
            # 处理不同的输入维度
            if feat_dim == self.input_dim:
                # 标准输入维度，使用增强的特征提取器
                base_features = self.feature_extractor(x)
            elif feat_dim == 384:
                # 来自Transformer的特征，需要转换为隐藏维度
                if self.input_projection is None:
                    self.input_projection = nn.Sequential(
                        nn.Linear(384, self.hidden_dim),
                        nn.ReLU(),
                        nn.LayerNorm(self.hidden_dim)
                    ).to(x.device)
                base_features = self.input_projection(x)
            elif feat_dim == self.hidden_dim:
                # 已经是隐藏维度，直接使用
                base_features = x
            else:
                # 其他维度，提供通用处理
                # 创建临时投影层处理
                temp_projection = nn.Linear(feat_dim, self.hidden_dim).to(x.device)
                base_features = nn.LayerNorm(self.hidden_dim).to(x.device)(
                    nn.ReLU()(temp_projection(x))
                )
                
            # 预测撚指类型概率
            type_logits = self.type_classifier(base_features)
            type_probs = torch.softmax(type_logits, dim=-1)
            
            # 预测撚指适用度分数
            suitability_score = torch.sigmoid(self.suitability_scorer(base_features))
            
            # 将类型和适用度信息与原始特征融合
            batch_size = base_features.size(0)
            seq_len = base_features.size(1)
            
            # 合并类型和适用度特征
            type_feature = type_probs
            suitability_feature = suitability_score
            
            # 构建增强特征
            extra_features = torch.cat([type_feature, suitability_feature], dim=-1)
            
            # 随机记录一些特征进行调试
            if random.random() < 0.01:  # 1%概率记录
                idx = random.randint(0, seq_len-1) if seq_len > 1 else 0
                logger.info(f"撚指特征样本: 类型概率={type_probs[0,idx].detach().cpu().tolist()}, "
                           f"适用度={suitability_score[0,idx].item():.4f}")
            
            # 融合原始特征和撚指特征
            combined_features = torch.cat([base_features, extra_features], dim=-1)
            enhanced_features = self.feature_enhancer(combined_features)
            
            return enhanced_features
            
        except Exception as e:
            logger.error(f"特征提取失败: {str(e)}")
            logger.error(traceback.format_exc())
            # 返回一个形状正确的空张量，而不是None
            batch_size = x.size(0) if x.dim() > 1 else 1
            seq_len = x.size(1) if x.dim() > 2 else x.size(0)
            return torch.zeros(batch_size, seq_len, self.hidden_dim, device=x.device) 