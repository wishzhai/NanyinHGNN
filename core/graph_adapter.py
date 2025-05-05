import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import logging

logger = logging.getLogger(__name__)

class AdapterLayer(nn.Module):
    """
    通用适配器层实现，添加在原始层之后
    使用瓶颈架构减少参数量
    """
    def __init__(self, in_features, bottleneck_dim, activation=F.gelu, skip_connection=True):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_dim = bottleneck_dim
        self.activation = activation
        self.skip_connection = skip_connection
        
        # 定义适配器层
        self.down_proj = nn.Linear(in_features, bottleneck_dim)
        self.up_proj = nn.Linear(bottleneck_dim, in_features)
        
        # 初始化参数
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化适配器参数，默认将up_proj权重置为接近0"""
        nn.init.normal_(self.down_proj.weight, std=0.01)
        nn.init.normal_(self.up_proj.weight, std=0.01)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x):
        """前向传播"""
        # 保存输入用于跳跃连接
        residual = x
        
        # 瓶颈结构：降维->非线性->升维
        out = self.down_proj(x)
        out = self.activation(out)
        out = self.up_proj(out)
        
        # 应用跳跃连接
        if self.skip_connection:
            out = out + residual
            
        return out

class GraphConvAdapter(nn.Module):
    """
    图卷积层的适配器实现
    """
    def __init__(self, gnn_layer, bottleneck_dim, activation=F.gelu, skip_connection=True):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.skip_connection = skip_connection
        
        # 获取GNN层的输出特征维度
        if hasattr(gnn_layer, 'out_feats'):
            out_feats = gnn_layer.out_feats
        elif hasattr(gnn_layer, 'out_dim'):
            out_feats = gnn_layer.out_dim
        else:
            # 默认情况下使用一个合理的值
            logger.warning(f"无法确定GNN层 {type(gnn_layer).__name__} 的输出维度，使用默认值512")
            out_feats = 512
            
        # 创建适配器
        self.adapter = AdapterLayer(
            in_features=out_feats,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            skip_connection=skip_connection
        )
        
        # 冻结原始GNN层参数
        for param in self.gnn_layer.parameters():
            param.requires_grad = False
    
    def forward(self, graph, feat, edge_weight=None):
        """前向传播，适配各种GNN层的接口"""
        # 检查是否需要edge_weight
        if edge_weight is not None and hasattr(self.gnn_layer, 'needs_edge_weight') and self.gnn_layer.needs_edge_weight:
            out = self.gnn_layer(graph, feat, edge_weight)
        else:
            out = self.gnn_layer(graph, feat)
            
        # 应用适配器
        out = self.adapter(out)
        return out

class MessageFunctionAdapter(nn.Module):
    """
    消息函数的适配器实现
    """
    def __init__(self, msg_func, in_feats, bottleneck_dim, activation=F.gelu, skip_connection=True):
        super().__init__()
        self.msg_func = msg_func
        self.in_feats = in_feats
        
        # 创建适配器
        self.adapter = AdapterLayer(
            in_features=in_feats,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            skip_connection=skip_connection
        )
        
        # 冻结原始消息函数参数
        if hasattr(msg_func, 'parameters'):
            for param in self.msg_func.parameters():
                param.requires_grad = False
    
    def forward(self, edges):
        """前向传播，处理消息"""
        # 应用原始消息函数
        msg = self.msg_func(edges)
        
        # 提取消息特征
        if isinstance(msg, dict) and 'msg' in msg:
            orig_msg = msg['msg']
            # 应用适配器
            adapted_msg = self.adapter(orig_msg)
            # 更新消息
            msg['msg'] = adapted_msg
            return msg
        else:
            # 如果消息不是预期格式，直接返回
            logger.warning("消息格式不符合预期，无法应用适配器")
            return msg

class AttentionAdapter(nn.Module):
    """
    注意力机制的适配器实现
    """
    def __init__(self, attn_module, bottleneck_dim, activation=F.gelu, skip_connection=True):
        super().__init__()
        self.attn_module = attn_module
        self.skip_connection = skip_connection
        
        # 获取注意力模块的输出维度
        if hasattr(attn_module, 'out_dim'):
            out_feats = attn_module.out_dim
        elif hasattr(attn_module, 'out_feats'):
            out_feats = attn_module.out_feats
        elif hasattr(attn_module, 'embed_dim'):
            out_feats = attn_module.embed_dim
        else:
            # 默认情况下使用一个合理的值
            logger.warning(f"无法确定注意力模块 {type(attn_module).__name__} 的输出维度，使用默认值512")
            out_feats = 512
            
        # 创建适配器
        self.adapter = AdapterLayer(
            in_features=out_feats,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            skip_connection=skip_connection
        )
        
        # 冻结原始注意力模块参数
        for param in self.attn_module.parameters():
            param.requires_grad = False
    
    def forward(self, *args, **kwargs):
        """前向传播，处理注意力"""
        # 应用原始注意力模块
        out = self.attn_module(*args, **kwargs)
        
        # 检查输出类型
        if isinstance(out, tuple):
            attn_out = out[0]
            # 应用适配器
            adapted_out = self.adapter(attn_out)
            # 返回修改后的输出元组
            return (adapted_out,) + out[1:]
        else:
            # 直接应用适配器
            return self.adapter(out)

class GraphAdapterManager:
    """
    图适配器管理器，用于向模型添加各种适配器
    """
    def __init__(self, config):
        self.config = config
        self.adapter_config = config.get('graph_adapter', {})
        self.bottleneck_dim = self.adapter_config.get('bottleneck_dim', 64)
        self.skip_connection = self.adapter_config.get('skip_connection', True)
        self.apply_to = self.adapter_config.get('apply_to', ['graph_conv', 'message_passing', 'attention'])
        
        # 记录已添加的适配器
        self.adapted_modules = {}
    
    def _is_graph_conv_module(self, module):
        """检查是否为图卷积模块"""
        return (
            isinstance(module, dglnn.GraphConv) or
            isinstance(module, dglnn.GATConv) or
            isinstance(module, dglnn.SAGEConv) or
            isinstance(module, dglnn.GINConv) or
            'GraphConv' in module.__class__.__name__ or
            'GATConv' in module.__class__.__name__ or
            'SAGEConv' in module.__class__.__name__
        )
    
    def _is_message_passing_module(self, module):
        """检查是否为消息传递模块"""
        return (
            'MessagePassing' in module.__class__.__name__ or
            hasattr(module, 'message_func') or
            hasattr(module, 'msg_fn')
        )
    
    def _is_attention_module(self, module):
        """检查是否为注意力模块"""
        return (
            isinstance(module, nn.MultiheadAttention) or
            'Attention' in module.__class__.__name__ or
            'MultiHead' in module.__class__.__name__
        )
    
    def apply_adapters(self, model):
        """向模型应用适配器"""
        adapted_count = 0
        
        # 遍历模型的所有模块
        for name, module in model.named_modules():
            # 跳过已适配的模块
            if id(module) in self.adapted_modules:
                continue
                
            # 根据配置和模块类型应用适配器
            if 'graph_conv' in self.apply_to and self._is_graph_conv_module(module):
                try:
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    parent = model if parent_name == '' else getattr(model, parent_name)
                    
                    # 创建图卷积适配器
                    adapted_module = GraphConvAdapter(
                        module,
                        bottleneck_dim=self.bottleneck_dim,
                        skip_connection=self.skip_connection
                    )
                    
                    # 替换原始模块
                    if '.' in name:
                        child_name = name.split('.')[-1]
                        setattr(parent, child_name, adapted_module)
                    else:
                        setattr(model, name, adapted_module)
                        
                    # 记录已适配的模块
                    self.adapted_modules[id(adapted_module)] = name
                    adapted_count += 1
                    logger.info(f"已将适配器应用到图卷积模块: {name}")
                except Exception as e:
                    logger.error(f"应用适配器到图卷积模块 {name} 时出错: {str(e)}")
                    
            elif 'message_passing' in self.apply_to and self._is_message_passing_module(module):
                try:
                    # 提取消息函数
                    if hasattr(module, 'message_func'):
                        msg_func = module.message_func
                    elif hasattr(module, 'msg_fn'):
                        msg_func = module.msg_fn
                    else:
                        continue
                        
                    # 推断输入特征维度
                    in_feats = 512  # 默认值
                    if hasattr(module, 'in_feats'):
                        in_feats = module.in_feats
                    elif hasattr(module, 'in_dim'):
                        in_feats = module.in_dim
                        
                    # 创建消息函数适配器
                    msg_adapter = MessageFunctionAdapter(
                        msg_func,
                        in_feats=in_feats,
                        bottleneck_dim=self.bottleneck_dim,
                        skip_connection=self.skip_connection
                    )
                    
                    # 替换原始消息函数
                    if hasattr(module, 'message_func'):
                        module.message_func = msg_adapter
                    elif hasattr(module, 'msg_fn'):
                        module.msg_fn = msg_adapter
                        
                    # 记录已适配的模块
                    self.adapted_modules[id(msg_adapter)] = f"{name}.message_func"
                    adapted_count += 1
                    logger.info(f"已将适配器应用到消息传递模块: {name}")
                except Exception as e:
                    logger.error(f"应用适配器到消息传递模块 {name} 时出错: {str(e)}")
                    
            elif 'attention' in self.apply_to and self._is_attention_module(module):
                try:
                    parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
                    parent = model if parent_name == '' else getattr(model, parent_name)
                    
                    # 创建注意力适配器
                    adapted_module = AttentionAdapter(
                        module,
                        bottleneck_dim=self.bottleneck_dim,
                        skip_connection=self.skip_connection
                    )
                    
                    # 替换原始模块
                    if '.' in name:
                        child_name = name.split('.')[-1]
                        setattr(parent, child_name, adapted_module)
                    else:
                        setattr(model, name, adapted_module)
                        
                    # 记录已适配的模块
                    self.adapted_modules[id(adapted_module)] = name
                    adapted_count += 1
                    logger.info(f"已将适配器应用到注意力模块: {name}")
                except Exception as e:
                    logger.error(f"应用适配器到注意力模块 {name} 时出错: {str(e)}")
        
        logger.info(f"共应用了 {adapted_count} 个适配器")
        return adapted_count 