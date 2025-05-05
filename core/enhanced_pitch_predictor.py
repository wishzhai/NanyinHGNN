import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

class HierarchicalDecoder(nn.Module):
    """改进的分层解码器"""
    
    def __init__(self, hidden_dim, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 简化的时值预测网络
        self.duration_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 32)  # 直接预测32个时值类别
        )
        
        # 时值辅助预测器 - 使用更简单的结构
        self.duration_aux = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )
        
        # 添加时值特征提取器
        self.duration_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # 力度预测网络 - 增强特征提取
        self.velocity_net = nn.ModuleList([
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 128)  # 128个力度等级
        ])
        
        # 力度辅助预测器
        self.velocity_aux = nn.Linear(hidden_dim, 128)
        
        # 音高预测网络保持不变
        self.pitch_net = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 128, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 88)
        )
        
    def forward(self, x):
        # 主时值预测
        duration_main = self.duration_net(x)
        # 辅助时值预测
        duration_aux = self.duration_aux(x)
        # 组合时值预测
        duration_logits = duration_main + 0.1 * duration_aux
        duration_probs = F.softmax(duration_logits, dim=-1)
        
        # 主力度预测
        velocity_input = torch.cat([x, duration_probs], dim=-1)
        velocity_features = velocity_input
        for i, layer in enumerate(self.velocity_net):
            velocity_features = layer(velocity_features)
            if i == 3:  # 在中间层添加残差连接
                velocity_features = velocity_features + velocity_input[:, :self.hidden_dim//2]
        velocity_main = velocity_features
        
        # 辅助力度预测
        velocity_aux = self.velocity_aux(x)
        # 组合力度预测
        velocity_logits = velocity_main + 0.1 * velocity_aux
        velocity_probs = F.softmax(velocity_logits, dim=-1)
        
        # 音高预测
        pitch_input = torch.cat([x, duration_probs, velocity_probs], dim=-1)
        pitch_logits = self.pitch_net(pitch_input)
        
        return duration_logits, duration_aux, velocity_logits, velocity_aux, pitch_logits

class MultiScaleHybridAttention(nn.Module):
    """优化的多尺度混合注意力机制"""
    
    def __init__(self, embed_dim, num_heads=6, dropout=0.1, local_window=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.local_window = local_window
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 全局注意力
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 为了与LoRA兼容，添加q_proj, k_proj, v_proj属性
        self.q_proj = self.global_attention.q_proj
        self.k_proj = self.global_attention.k_proj
        self.v_proj = self.global_attention.v_proj
        
        # 局部感知 - 使用多尺度卷积
        self.local_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k//2, groups=embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.SiLU()
            ) for k in [3, 7, 11]  # 多个感受野
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 相对位置编码
        self.rel_pos_encoding = nn.Parameter(torch.zeros(2 * local_window - 1, self.head_dim))
        self._reset_parameters()
        
    def _reset_parameters(self):
        """改进的参数初始化"""
        nn.init.trunc_normal_(self.rel_pos_encoding, std=0.02)
        
    def forward(self, x, attention_mask=None):
        batch_size, seq_len, _ = x.shape
        
        # 全局注意力
        global_out, attn_weights = self.global_attention(x, x, x, attn_mask=attention_mask)
        
        # 多尺度局部特征
        x_conv = x.transpose(1, 2)
        local_outs = []
        for conv in self.local_convs:
            local_outs.append(conv(x_conv).transpose(1, 2))
        
        # 特征融合
        all_features = torch.cat([global_out] + local_outs, dim=-1)
        fused_features = self.fusion(all_features)
        
        # 添加相对位置信息
        rel_pos = self._get_relative_positions(seq_len)
        if attention_mask is not None:
            rel_pos = rel_pos.masked_fill(attention_mask.unsqueeze(-1) == 0, 0)
        
        # 残差连接和输出
        output = self.output_projection(fused_features + rel_pos)
        output = output + x
        
        return output, attn_weights
        
    def _get_relative_positions(self, length):
        """获取相对位置编码"""
        positions = torch.arange(length, device=self.rel_pos_encoding.device)
        rel_pos_mat = positions.unsqueeze(1) - positions.unsqueeze(0)
        rel_pos_mat = rel_pos_mat + self.local_window - 1
        return self.rel_pos_encoding[rel_pos_mat]

class EnhancedPitchPredictor(nn.Module):
    """改进的音高预测器"""
    
    def __init__(self, config):
        super().__init__()
        # 配置参数
        model_config = config.get('model', {})
        # 使用256维度以匹配预训练模型
        self.hidden_dim = 256
        logger.info(f"初始化EnhancedPitchPredictor，使用hidden_dim={self.hidden_dim}")
        self.num_heads = 8
        self.dropout = 0.2
        self.local_window = 12
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout)
        )
        
        # 注意力层
        self.attention_layers = nn.ModuleList([
            MultiScaleHybridAttention(
                embed_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                local_window=self.local_window
            ) for _ in range(2)
        ])
        
        # 层标准化
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(2)
        ])
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(self.dropout)
        )
        
        # 分层解码器
        self.hierarchical_decoder = HierarchicalDecoder(self.hidden_dim, self.dropout)
        
        # 辅助任务: 节奏模式预测
        self.rhythm_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 16)  # 16种节奏模式
        )
        
        # 辅助任务: 力度轮廓预测
        self.dynamics_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 8)  # 8种力度轮廓
        )
        
        # 调式感知层
        self.mode_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # 课程学习状态
        self.curriculum_step = 0
        
        # 更新课程学习参数
        steps_per_epoch = 51
        self.total_epochs = 300
        
        # 课程学习阶段
        self.warmup_epochs = 30
        self.duration_epochs = 100
        self.velocity_epochs = 200
        
        # 转换为步数
        self.warmup_steps = self.warmup_epochs * steps_per_epoch
        self.duration_steps = self.duration_epochs * steps_per_epoch
        self.velocity_steps = self.velocity_epochs * steps_per_epoch
        self.total_steps = self.total_epochs * steps_per_epoch
        
        # 初始化课程权重
        self.curriculum_weights = {
            'duration': 0.0,
            'velocity': 0.0,
            'pitch': 0.0
        }
        
        # 学习率相关参数
        self.initial_lr = 5e-5
        self.max_lr = 3e-4
        self.min_lr = 5e-6
        
        # 初始化参数
        self._init_weights()
        
        logger.info("EnhancedPitchPredictor初始化完成，所有组件使用256维度")
        
    def _init_weights(self):
        """改进的参数初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def update_curriculum(self, step):
        """改进的课程学习策略，使用更平滑的过渡"""
        self.curriculum_step = step
        
        # 预热阶段
        if step < self.warmup_steps:
            ratio = step / self.warmup_steps
            self.curriculum_weights = {
                'duration': ratio * 0.5,  # 逐渐增加时值权重
                'velocity': 0.0,
                'pitch': 0.0
            }
            
        # 时值学习阶段
        elif step < self.duration_steps:
            progress = (step - self.warmup_steps) / (self.duration_steps - self.warmup_steps)
            self.curriculum_weights = {
                'duration': min(1.0, 0.5 + progress * 0.5),  # 平滑增加到1.0
                'velocity': 0.0,
                'pitch': 0.0
            }
            
        # 力度学习阶段
        elif step < self.velocity_steps:
            progress = (step - self.duration_steps) / (self.velocity_steps - self.duration_steps)
            self.curriculum_weights = {
                'duration': 1.0,
                'velocity': progress * 0.8,  # 平滑增加力度权重
                'pitch': 0.0
            }
            
        # 全任务学习阶段
        else:
            progress = min(1.0, (step - self.velocity_steps) / (self.total_steps - self.velocity_steps))
            self.curriculum_weights = {
                'duration': 1.0,
                'velocity': 0.8,
                'pitch': progress * 0.6  # 平滑增加音高权重
            }
    
    def get_lr_scale(self, step):
        """获取学习率缩放因子"""
        if step < self.warmup_steps:
            # 线性预热
            return step / self.warmup_steps
        
        # 使用余弦退火
        step = step - self.warmup_steps
        total_steps = self.total_steps - self.warmup_steps
        progress = step / total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
    
    def forward(self, x, attention_mask=None):
        """改进的前向传播"""
        try:
            # 输入投影
            x = self.input_projection(x)
            
            # 注意力掩码处理
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                attention_mask = (1.0 - attention_mask) * -10000.0
            
            # 多层注意力处理
            attention_weights = []
            for i, (attn_layer, norm_layer) in enumerate(zip(self.attention_layers, self.layer_norms)):
                x_norm = norm_layer(x)
                attn_out, weights = attn_layer(x_norm, attention_mask)
                x = x + attn_out
                attention_weights.append(weights)
            
            # 前馈网络
            x = x + self.ffn(x)
            
            # 调式感知
            mode_features = self.mode_layer(x)
            x = x + mode_features
            
            # 辅助任务预测
            rhythm_logits = self.rhythm_predictor(x)
            dynamics_logits = self.dynamics_predictor(x)
            
            # 分层预测主要任务
            duration_logits, duration_aux, velocity_logits, velocity_aux, pitch_logits = self.hierarchical_decoder(x)
            
            # 课程学习策略
            if self.training:
                # 根据课程权重调整预测
                if self.curriculum_weights['velocity'] == 0:
                    velocity_logits = velocity_logits.detach()
                    velocity_aux = velocity_aux.detach()
                if self.curriculum_weights['pitch'] == 0:
                    pitch_logits = pitch_logits.detach()
            
            return {
                'pitch_logits': pitch_logits,
                'duration_logits': duration_logits,
                'duration_aux': duration_aux,
                'velocity_logits': velocity_logits,
                'velocity_aux': velocity_aux,
                'rhythm_logits': rhythm_logits,
                'dynamics_logits': dynamics_logits,
                'attention_weights': attention_weights
            }
            
        except Exception as e:
            logger.error(f"音高预测器前向传播失败: {str(e)}")
            device = x.device if isinstance(x, torch.Tensor) else torch.device('cuda')
            return {
                'pitch_logits': torch.zeros((1, 88), device=device),
                'duration_logits': torch.zeros((1, 32), device=device),
                'duration_aux': torch.zeros((1, 32), device=device),
                'velocity_logits': torch.zeros((1, 128), device=device),
                'velocity_aux': torch.zeros((1, 128), device=device),
                'rhythm_logits': torch.zeros((1, 16), device=device),
                'dynamics_logits': torch.zeros((1, 8), device=device),
                'attention_weights': None
            }
    
    def compute_loss(self, outputs, targets, weights=None):
        """改进的损失计算，添加自适应权重"""
        device = outputs['pitch_logits'].device
        total_loss = torch.tensor(0.0, device=device)
        losses = {}
        
        # 更激进的权重设置
        if weights is None:
            weights = {
                'duration': 8.0,  # 进一步增加时值权重
                'velocity': 3.0,  # 保持力度权重
                'pitch': 1.0,
                'rhythm': 0.5,  # 增加节奏权重
                'dynamics': 0.3
            }
        
        # 时值损失
        if 'duration_target' in targets:
            # 主时值损失
            duration_main_loss = F.cross_entropy(
                outputs['duration_logits'].view(-1, 32),
                targets['duration_target'].view(-1),
                reduction='none'
            )
            
            # 添加样本重要性权重
            if 'duration_weights' in targets:
                duration_main_loss = duration_main_loss * targets['duration_weights'].view(-1)
            
            duration_main_loss = duration_main_loss.mean()
            
            # 辅助时值损失
            duration_aux_loss = F.cross_entropy(
                outputs['duration_aux'].view(-1, 32),
                targets['duration_target'].view(-1)
            )
            
            # 添加L1正则化
            l1_reg = 0.01 * (
                torch.norm(outputs['duration_logits'], 1) +
                torch.norm(outputs['duration_aux'], 1)
            )
            
            # 组合损失
            duration_loss = duration_main_loss + 0.3 * duration_aux_loss + l1_reg
            losses['duration'] = duration_loss
            total_loss += weights['duration'] * self.curriculum_weights['duration'] * duration_loss
        
        # 力度损失
        if 'velocity_target' in targets:
            velocity_main_loss = F.cross_entropy(
                outputs['velocity_logits'].view(-1, 128),
                targets['velocity_target'].view(-1)
            )
            velocity_aux_loss = F.cross_entropy(
                outputs['velocity_aux'].view(-1, 128),
                targets['velocity_target'].view(-1)
            )
            velocity_loss = velocity_main_loss + 0.2 * velocity_aux_loss
            
            # 添加平滑损失
            velocity_smooth_loss = F.smooth_l1_loss(
                outputs['velocity_logits'][:, 1:],
                outputs['velocity_logits'][:, :-1]
            )
            
            losses['velocity'] = velocity_loss + 0.1 * velocity_smooth_loss
            total_loss += weights['velocity'] * self.curriculum_weights['velocity'] * losses['velocity']
        
        # 音高损失
        if 'pitch_target' in targets:
            pitch_loss = F.cross_entropy(
                outputs['pitch_logits'].view(-1, 88),
                targets['pitch_target'].view(-1)
            )
            losses['pitch'] = pitch_loss
            total_loss += weights['pitch'] * self.curriculum_weights['pitch'] * pitch_loss
        
        # 辅助任务损失
        if 'rhythm_target' in targets:
            rhythm_loss = F.cross_entropy(
                outputs['rhythm_logits'].view(-1, 16),
                targets['rhythm_target'].view(-1)
            )
            losses['rhythm'] = rhythm_loss
            total_loss += weights['rhythm'] * rhythm_loss
        
        if 'dynamics_target' in targets:
            dynamics_loss = F.cross_entropy(
                outputs['dynamics_logits'].view(-1, 8),
                targets['dynamics_target'].view(-1)
            )
            losses['dynamics'] = dynamics_loss
            total_loss += weights['dynamics'] * dynamics_loss
        
        return total_loss, losses

    def compute_pitch_smoothness_loss(self, pitch_logits):
        """改进的音高平滑损失"""
        if pitch_logits.dim() == 2:
            pitch_logits = pitch_logits.unsqueeze(0)
        
        # 使用Huber损失
        pitch_diffs = pitch_logits[:, 1:] - pitch_logits[:, :-1]
        smoothness_loss = F.huber_loss(pitch_diffs, torch.zeros_like(pitch_diffs), reduction='mean', delta=1.0)
        
        return smoothness_loss

    def compute_mode_specific_range_loss(self, pitch_logits, mode):
        """改进的调式约束损失"""
        try:
            # 定义各调式的音域范围
            mode_ranges = {
                'wukong': {
                    'upper': list(range(74, 96)),  # 上音域
                    'lower': list(range(48, 74))   # 下音域
                },
                'sikong': list(range(48, 85)),
                'wukong_siyi': list(range(48, 85)),
                'beisi': list(range(48, 80))
            }
            
            if mode not in mode_ranges:
                return torch.tensor(0.0, device=pitch_logits.device)
            
            # 计算音高概率分布
            pitch_probs = F.softmax(pitch_logits, dim=-1)
            
            if mode == 'wukong':
                # 五空调特殊处理
                upper_range = mode_ranges['wukong']['upper']
                lower_range = mode_ranges['wukong']['lower']
                
                # 计算上下音域的概率
                upper_probs = pitch_probs[:, [i-21 for i in upper_range if 21 <= i < 109]].sum(dim=-1)
                lower_probs = pitch_probs[:, [i-21 for i in lower_range if 21 <= i < 109]].sum(dim=-1)
                
                # 使用focal loss
                alpha = 0.25
                gamma = 2.0
                upper_loss = -alpha * ((1 - upper_probs) ** gamma) * torch.log(upper_probs + 1e-8)
                lower_loss = -alpha * ((1 - lower_probs) ** gamma) * torch.log(lower_probs + 1e-8)
                
                return (upper_loss + lower_loss).mean()
            else:
                # 其他调式的处理
                valid_range = mode_ranges[mode]
                valid_indices = [i-21 for i in valid_range if 21 <= i < 109]
                valid_probs = pitch_probs[:, valid_indices].sum(dim=-1)
                
                # 使用focal loss
                alpha = 0.25
                gamma = 2.0
                range_loss = -alpha * ((1 - valid_probs) ** gamma) * torch.log(valid_probs + 1e-8)
                return range_loss.mean()
            
        except Exception as e:
            logger.error(f"计算音域约束损失失败: {str(e)}")
            return torch.tensor(0.0, device=pitch_logits.device) 