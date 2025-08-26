"""
MambaOut models for image classification.
Some implementations are modified from:
timm (https://github.com/rwightman/pytorch-image-models),
MetaFormer (https://github.com/sail-sg/metaformer),
InceptionNeXt (https://github.com/sail-sg/inceptionnext)
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.nn import init


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 3, 'input_size': (1, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


class StemLayer(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """

    def __init__(self,
                 in_channels=1,
                 out_channels=64,
                 act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-7)):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               out_channels // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
                # C: 3 --> 48
                ## in_channels = 3, out_channels = 48
                ## output_height = (input_height -3 + 2)/2 + 1
        self.norm1 = norm_layer(out_channels // 2) #  channels = 48
        self.act = act_layer()  # LayerNorm(eps=1e-6)  # channels = 48
        self.conv2 = nn.Conv2d(out_channels // 2,
                               out_channels,
                               kernel_size=3,
                               stride=2,
                               padding=1)
                # C: 48 --> 64
                ## in_channels = 48, out_channels = 64
                ## out_height = (output - 3 + 2)/2 + 1
        self.norm2 = norm_layer(out_channels)  # channels = 64

    def forward(self, x):

        x = self.conv1(x) # (32, 3, 224, 224) --> (32, 48, 112, 112)✅ 1
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) --> (B, H, W, C) == (32, 112, 112, 48) ✅ 2
        x = self.norm1(x)  # (32, 112, 112, 48) --> (32, 112, 112, 48) ✅ 3
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) --> (B, C, H, W)  

        x = self.act(x)  
        ### 类比：类似于将图像展平为 H*W 个网格点，每个点有一个 C 维特征向量，然后对这些向量独立归一化。
        x = self.conv2(x)  # (32, 48, 112, 112)  --> (32, 64, 56, 56) ✅ 5
        ## output_height = (112 -3 +2)/2 + 1 = (55.5) + 1 = 56
        x = x.permute(0, 2, 3, 1)  # (B, C, H ,W) --> (B, H, W, C) == (32, 56, 56, 64) ✅ 6
        x = self.norm2(x)  # (32, 56, 56, 64) --> (32, 56, 56, 64) ✅ 6
        return x  # (32, 56, 56, 64) (B, H, W, C)


class DownsampleLayer(nn.Module):
    r""" Code modified from InternImage:
        https://github.com/OpenGVLab/InternImage
    """


    def __init__(self, in_channels=64, out_channels=96, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2,
                              padding=1)
            # in_channels=64, out_channels=96
        self.norm = norm_layer(out_channels)

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # ✅ 1

        x = self.norm(x)  # (32, 28, 28, 96) --> (32, 28, 28, 96)  # ✅ 2
        return x

class SEAttention(nn.Module):

    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # 在空间维度上,将H×W压缩为1×1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 包含两层全连接,先降维,后升维。最后接一个sigmoid函数
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, 2*channel, bias=False),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x1, x2):

        x1 = x1.permute(0, 3, 1, 2)  # tensor:(32, 576, 7, 7) (10, 288, 7, 7)
        x2 = x2.permute(0, 3, 1, 2)  # tensor:(32, 576, 7, 7)
        # (B,C,H,W)
        B1, C1, H1, W1 = x1.size() # 32, 576, 7, 7
        B1, C1, H1, W1 = x1.size() # 32, 576, 7, 7



        x = x1 + x2  # (32, 576, 7, 7) tensor值与值简单相加
        # Squeeze: (B,C,H,W)-->avg_pool-->(B,C,1,1)-->view-->(B,C)
        y = self.avg_pool(x).view(B1, C1)  # y tensor(32, 576) (10, 288)
        # Excitation: (B,C)-->fc-->(B,2C)-->(B, 2C, 1, 1)
        y = self.fc(y) # (10, 576)
        # print(f"y.shape after fc: {y.shape}")  # Debugging line
        y = y.view(B1, 2*C1, 1, 1)
        # split
        weight1 = torch.sigmoid(y[:,:C1,:,:]) # (B,C,1,1)
        weight2 = torch.sigmoid(y[:, C1:, :, :]) # (B,C,1,1)
        # scale: (B,C,H,W) * (B,C,1,1) == (B,C,H,W)
        out = x1 * weight1 + x2 * weight2
        out = out.permute(0, 2, 3, 1)
        return out

class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=3, act_layer=nn.GELU, mlp_ratio=2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)  # (4 * )
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 可以调整标准差值
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x

class GatedCNNBlock(nn.Module):
    r""" Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args: 
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and 
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """
    def __init__(self, dim, expansion_ratio=8/3, kernel_size=7, conv_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm,eps=1e-6), 
                 act_layer=nn.GELU,
                 drop_path=0.,
                 **kwargs):
        super().__init__()
        self.norm = norm_layer(dim)
        hidden = int(expansion_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.act = act_layer()
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.conv = nn.Conv2d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size//2, groups=conv_channels)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)  # 调整标准差以适应您的需求
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, x):
        shortcut = x # [B, H, W, C] tensor(32, 56, 56, 64)
        x = self.norm(x)  # tensor(32, 56, 56, 64)
        g, i, c = torch.split(self.fc1(x), self.split_indices, dim=-1)

        c = c.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W] || (32, 56, 56, 64) --> (32, 64, 56, 56)
        c = self.conv(c)
        c = c.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.fc2(self.act(g) * torch.cat((i, c), dim=-1))
        x = self.drop_path(x)
        return x + shortcut

# 钩子函数：用来保存每一层的输出
def hook_fn(module, input, output):
    print(f"Layer: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")
    print("-" * 50)


# fixme 这里是四个下采样层的调用顺序
DOWNSAMPLE_LAYERS_FOUR_STAGES = [StemLayer] + [DownsampleLayer]*3

#todo 所以，那现在前面就需要设置为多头输入

class MambaOut(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [3, 3, 9, 3].
        dims (int): Feature dimension at each stage. Default: [64, 192, 384, 576].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
        head_dropout (float): dropout for MLP classifier. Default: 0.
    """
    def __init__(self, in_chans=1, num_classes=3,
                 depths=[3, 3, 9, 3],
                 dims=[48, 64, 192, 288],
                 reduction=8,
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 seattention_layer=SEAttention(),
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 act_layer=nn.GELU,
                 conv_ratio=1.0,
                 kernel_size=7,
                 drop_path_rate=0.,
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=MlpHead,
                 head_dropout=0.0, 
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):  # 判断depth 是不是 list/ tuple

            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]  # 同样，判断dims是不是这两类

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):  # 判断downsample_layers是不是这两类

            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims # down_dims = [3] + [64, 192, 384, 576] = [3, 64, 192, 384, 576]
        self.downsample_layers1 = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)])  # 下采样层1的构建

        self.downsample_layers2 = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)])  # 下采样层2的构建
        self.seattention_layer = SEAttention(channel=dims[-1])  # SEAttention/特征融合层构建


        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stages = nn.ModuleList()
        cur = 0 # 0+3 3+3 6+9 9+3
        for i in range(num_stage):
            stage = nn.Sequential(
                *[GatedCNNBlock(dim=dims[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                 kernel_size=kernel_size,
                conv_ratio=conv_ratio,
                drop_path=dp_rates[cur + j],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        #fixme 3.25
        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def extract_features(self, x1, x2):
        # 前向传播到SEAttention层之前
        for i in range(self.num_stage):
            x1 = self.downsample_layers1[i](x1)  # x1 tensor(32, 3, 224, 224) --> (32, 56, 56, 64)
            x1 = self.stages[i](x1)  # (32, 56, 56, 64)
            x2 = self.downsample_layers2[i](x2)  # x2 tensor(32, 3, 224, 224) --> (32, 56, 56, 64)
            x2 = self.stages[i](x2)

        # 进行 SEAttention 特征融合

        fused_features = self.seattention_layer(x1, x2)

        # 获取SEAttention层输入作为特征（可选：也可在SEAttention之后提取）
        return fused_features.permute(0, 3, 1, 2).flatten(1)  # [B, C, H, W] -> [B, C*H*W]

    def forward_features(self, x1, x2):  
        # 检查 x1 和 x2 的批次大小是否匹配
        if x1.size(0) != x2.size(0):
            raise ValueError(f"Input sizes do not match: x1: {x1.size()}, x2: {x2.size()}")
            return None

        for i in range(self.num_stage):
            x1 = self.downsample_layers1[i](x1)  # x1 tensor(32, 3, 224, 224) --> (32, 56, 56, 64)
            x1 = self.stages[i](x1)  # (32, 56, 56, 64)
            x2 = self.downsample_layers2[i](x2)  # x2 tensor(32, 3, 224, 224) --> (32, 56, 56, 64)
            x2 = self.stages[i](x2)

        # 进行 SEAttention 特征融合

        fused_features = self.seattention_layer(x1, x2)

        return self.norm(fused_features.mean([1, 2])) # (B, H, W, C) -> (B, C)

    def forward(self, x1, x2):
        x = self.forward_features(x1, x2)
        if x is None:
            return None
        x = self.head(x)
        return x



###############################################################################
# a series of MambaOut models
# @register_model
# def mambaout_femto(pretrained=False, **kwargs):
#     model = MambaOut(
#         depths=[3, 3, 9, 3],
#         dims=[48, 64, 192, 288],
#         **kwargs)
#     #todo
#     model.default_cfg = default_cfgs['mambaout_femto']
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(
#             url= model.default_cfg['url'], map_location="cpu", check_hash=True)
#         model.load_state_dict(state_dict)
#     return model


# # Kobe Memorial Version with 24 Gated CNN blocks
# @register_model
# def mambaout_kobe(pretrained=False, **kwargs):
#     model = MambaOut(
#         depths=[3, 3, 15, 3],
#         dims=[48, 64, 192, 288],
#         **kwargs)
#     model.default_cfg = default_cfgs['mambaout_kobe']
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(
#             url= model.default_cfg['url'], map_location="cpu", check_hash=True)
#         model.load_state_dict(state_dict)
#     return model


# @register_model
# def mambaout_tiny(pretrained=False, **kwargs):
#     model = MambaOut(
#         depths=[3, 3, 9, 3],
#         dims=[64, 192, 384, 576],
#         **kwargs)
#     model.default_cfg = default_cfgs['mambaout_tiny']
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(
#             url= model.default_cfg['url'], map_location="cpu", check_hash=True)
#         model.load_state_dict(state_dict)
#     return model
#
#
# @register_model
# def mambaout_small(pretrained=False, **kwargs):
#     model = MambaOut(
#         depths=[3, 4, 27, 3],
#         dims=[64, 192, 384, 576],
#         **kwargs)
#     model.default_cfg = default_cfgs['mambaout_small']
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(
#             url= model.default_cfg['url'], map_location="cpu", check_hash=True)
#         model.load_state_dict(state_dict)
#     return model
#
#
# @register_model
# def mambaout_base(pretrained=False, **kwargs):
#     model = MambaOut(
#         depths=[3, 4, 27, 3],
#         dims=[96, 256, 512, 768],
#         **kwargs)
#     model.default_cfg = default_cfgs['mambaout_base']
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(
#             url= model.default_cfg['url'], map_location="cpu", check_hash=True)
#         model.load_state_dict(state_dict)
#     return model