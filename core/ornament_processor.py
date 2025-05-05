import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class OrnamentProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ornament_styles = config.get('ornament_styles', {'standard': {}})  # 确保至少有一个默认风格
        self.global_config = config.get('ornament_global', {})
        
        # 从配置获取关键维度
        # 隐藏维度：保持与预训练模型兼容(256)
        self.input_dim = config.get('input_dim', 256)  # 使用256匹配预训练模型
        self.hidden_dim = config.get('hidden_dim', 256)  # 使用256匹配预训练模型 
        self.output_dim = config.get('output_dim', 512)  # 输出维度保持512
        self.projection_dim = config.get('projection_dim', 512)  # 投影维度为512
        
        # 投影层：从input_dim到projection_dim (256->512)
        # 注意：检查点中的投影层维度是[256, 512]
        self.projection = nn.Linear(self.input_dim, self.projection_dim)
        
        # 风格编码器 (使用input_dim=256)
        self.style_encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3维风格向量
        )
        
        # 确保dropout参数是数值类型
        dropout_value = config.get('dropout', 0.1)
        # 如果是字典类型，使用默认值
        if isinstance(dropout_value, dict):
            logger.warning(f"接收到字典类型的dropout参数: {dropout_value}，使用默认值0.1")
            dropout_value = 0.1
        
        # 上下文编码器 (使用input_dim=256)
        use_bidirectional = config.get('bidirectional', True)
        rnn_hidden = self.input_dim * 2 if use_bidirectional else self.input_dim
        self.context_encoder = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.input_dim // 2,
            num_layers=config.get('num_layers', 2),
            dropout=float(dropout_value),  # 确保转换为浮点数
            bidirectional=use_bidirectional,
            batch_first=True
        )
        
        # 装饰音生成器 (从rnn_hidden到hidden_dim(256)，然后到output_dim(512))
        self.ornament_generator = nn.Sequential(
            nn.Linear(rnn_hidden, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.output_dim)  # 从256到512
        )
        
        # 特征归一化
        self.feature_norm = nn.LayerNorm(self.input_dim)
        
        # 初始化参数
        self.pitch_dim = config.get('pitch_dim', self.input_dim // 2)  # 使用input_dim而不是output_dim
        self.duration_dim = config.get('duration_dim', self.input_dim // 4)
        self.velocity_dim = config.get('velocity_dim', self.input_dim // 8)
        self.position_dim = config.get('position_dim', self.input_dim // 8)
        
        # 装饰音跟踪状态
        self._last_ornament_pos = -float('inf')
        self._consecutive_count = 0
        
        logger.info(f"OrnamentProcessor初始化完成，维度配置：input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, " +
                   f"output_dim={self.output_dim}, projection_dim={self.projection_dim}, dropout={dropout_value}")
        
    def forward(self, features, attention_mask=None):
        """处理装饰音
        
        Args:
            features: [batch_size, seq_len, input_dim] 或 [seq_len, input_dim]
            attention_mask: [batch_size, seq_len] 或 [seq_len]
            
        Returns:
            enhanced_features: [batch_size, seq_len, input_dim] 或 [seq_len, input_dim]
        """
        try:
            # 重置装饰音跟踪状态
            self._reset_ornament_state()
            
            # 确保输入是3维的
            if features.dim() == 2:
                features = features.unsqueeze(0)
                if attention_mask is not None:
                    attention_mask = attention_mask.unsqueeze(0)
            
            # 特征归一化
            features = self.feature_norm(features)
            
            batch_size, seq_len, _ = features.shape
            device = features.device
            
            # 1. 获取上下文特征
            context_features, _ = self.context_encoder(features)
            
            # 2. 预测装饰音风格
            style_logits = self.style_encoder(features)
            style_probs = F.softmax(style_logits, dim=-1)
            
            # 3. 处理每个位置
            enhanced_features = []
            for i in range(batch_size):
                seq_features = []
                self._reset_ornament_state()  # 每个序列重置状态
                
                for j in range(seq_len):
                    try:
                        if attention_mask is not None and not attention_mask[i, j]:
                            seq_features.append(features[i, j])
                            continue
                        
                        # 获取当前音符的装饰音风格
                        style_idx = torch.argmax(style_probs[i, j])
                        style_name = list(self.ornament_styles.keys())[min(style_idx.item(), len(self.ornament_styles)-1)]
                        style = self.ornament_styles[style_name]
                        
                        if not style.get('enabled', True):
                            seq_features.append(features[i, j])
                            continue
                        
                        # 检查是否应用装饰音
                        if self._should_apply_ornament(features[i], j, style):
                            self._update_ornament_state(j)
                            
                            # 获取上下文信息
                            context = torch.cat([
                                features[i, j],
                                context_features[i, j]
                            ], dim=-1)
                            
                            # 先使用投影层 (256->512)
                            projected = self.projection(features[i, j])
                            
                            # 生成装饰音特征 (生成器: 768->384->512)
                            ornament_feature = self.ornament_generator(context)
                            
                            # 返回到384维度
                            # 注意：由于生成器输出现在是512维，我们需要处理回384维
                            processed_feature = self._apply_ornament_params(
                                ornament_feature,
                                style,
                                features[i, j]
                            )
                            
                            seq_features.append(processed_feature)
                        else:
                            self._consecutive_count = 0
                            seq_features.append(features[i, j])
                            
                    except Exception as e:
                        logger.error(f"处理位置 {j} 时出错: {str(e)}")
                        seq_features.append(features[i, j])  # 出错时使用原始特征
                
                enhanced_features.append(torch.stack(seq_features))
            
            result = torch.stack(enhanced_features)
            
            # 如果输入是2维的，去掉批次维度
            if features.size(0) == 1:
                result = result.squeeze(0)
            
            return result
            
        except Exception as e:
            logger.error(f"装饰音处理失败: {str(e)}")
            return features
    
    def _apply_ornament_params(self, feature, style, original_feature):
        """应用装饰音参数，确保特征维度的正确性和数值的有效性"""
        try:
            # 特征分段 - 现在处理512维特征，但需要返回256维
            # 我们需要先将512维特征映射回256维
            if feature.shape[0] == self.output_dim:  # 如果是512维
                # 创建256维的临时特征，使用512维特征的部分来填充
                temp_feature = torch.zeros(self.input_dim, device=feature.device)
                # 使用前面部分填充
                temp_feature[:self.input_dim] = feature[:self.input_dim]
                feature = temp_feature
                logger.debug(f"调整特征维度：从{self.output_dim}到{self.input_dim}")
            
            # 特征分段 - 使用256维特征进行分段
            pitch_feature = feature[:self.pitch_dim].clamp(0, 1)
            duration_feature = feature[self.pitch_dim:self.pitch_dim+self.duration_dim]
            velocity_feature = feature[self.pitch_dim+self.duration_dim:self.pitch_dim+self.duration_dim+self.velocity_dim]
            position_feature = feature[-self.position_dim:]
            
            # 1. 应用风格参数
            duration_factor = style.get('duration_factor', 0.3)
            velocity_factor = style.get('velocity_factor', 0.7)
            position_shift = style.get('position_shift', 0.05)
            
            # 2. 调整特征
            duration_feature = duration_feature * duration_factor
            velocity_feature = velocity_feature * velocity_factor
            
            # 应用位置偏移
            position_direction = style.get('direction', 'up')
            shift_sign = 1 if position_direction == 'up' else -1
            position_feature = position_feature + shift_sign * position_shift
            
            # 3. 合并特征 - 确保返回的是256维特征
            combined_feature = torch.cat([
                pitch_feature,
                duration_feature,
                velocity_feature,
                position_feature
            ])
            
            # 确保返回的特征是256维
            if combined_feature.shape[0] != self.input_dim:
                logger.warning(f"特征维度不匹配，调整为原始特征: {combined_feature.shape[0]} vs {self.input_dim}")
                return original_feature
                
            return combined_feature
            
        except Exception as e:
            logger.error(f"应用装饰音参数时出错: {str(e)}")
            return original_feature
    
    def _should_apply_ornament(self, seq_features, pos, style):
        """判断是否应该在当前位置应用装饰音"""
        try:
            # 基础概率检查
            if torch.rand(1).item() >= style.get('probability', 0.5):
                return False
            
            # 检查最小间隔
            min_interval = self.global_config.get('min_interval', 2)
            if pos > 0 and pos - self._last_ornament_pos < min_interval:
                return False
            
            # 检查最大连续装饰音数
            max_consecutive = self.global_config.get('max_consecutive', 2)
            if self._consecutive_count >= max_consecutive:
                return False
            
            # 检查乐句位置
            if style.get('condition') == 'phrase_beginning':
                # TODO: 实现乐句开始检测
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"装饰音条件检查失败: {str(e)}")
            return False
    
    def _reset_ornament_state(self):
        """重置装饰音状态"""
        self._last_ornament_pos = -float('inf')
        self._consecutive_count = 0
        
    def _update_ornament_state(self, pos):
        """更新装饰音状态
        
        Args:
            pos: 当前位置
        """
        self._last_ornament_pos = pos
        self._consecutive_count += 1 