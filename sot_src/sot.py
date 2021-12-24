import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.models.senet import SEModule
from timm.models.inception_resnet_v2 import BasicConv2d
from timm.models.densenet import DenseBlock
from timm.models.densenet import DenseTransition

import numpy as np
from torch.nn import functional as F
from .model import Classifier
from .model import TokenEmbed
from .model import ViTBlock, get_sinusoid_encoding


visualTokenConfig = dict(
    type='DenseNet',
    token_dim = 64,
    large_output = False,
)

ViTConfig = dict(
    embed_dim=384, 
    depth=14, 
    num_heads=6, 
    mlp_ratio=3.,
    qkv_bias=False,
    qk_scale=None,
    attn_drop=0.,
    norm_layer=nn.LayerNorm,
    act_layer=nn.GELU,
)

representationConfig = dict(
    type='MGCrP',
    fusion_type='sum_fc',
    args=dict(
        dim=384,
        num_heads=6,
        wr_dim=24,
        normalization=dict(
            type='svPN',
            alpha=0.5,
            iterNum=1,
            svNum=1,
            regular=nn.Dropout(0.5),
            input_dim=24,
        )
    ),
)



class SoT(nn.Module):
    def __init__(self,
            img_size=224, 
            in_chans=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            visualTokenConfig=visualTokenConfig,
            ViTConfig=ViTConfig, 
            representationConfig=representationConfig):
        super(SoT, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = ViTConfig['embed_dim']
        self.depth = ViTConfig['depth']
        ViTConfig.pop('depth')
        #-------------
        # Build SO-ViT
        #-------------
        self.visual_tokens = TokenEmbed(img_size=img_size, in_chans=in_chans, embed_dim=self.embed_dim, visualTokenConfig=visualTokenConfig)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)] 
        self.blocks = nn.ModuleList([ViTBlock(drop=drop_rate, drop_path=dpr[i], **ViTConfig) for i in range(self.depth)])
        self.classifier = Classifier(num_classes=num_classes, input_dim=self.embed_dim, representationConfig=representationConfig)
        #-------------------------------------------
        # Prepare Class Token and Position Embedding
        #-------------------------------------------
        num_patches = self.visual_tokens.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=self.embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.embed_dim)
        self.apply(self._init_weights)
        trunc_normal_(self.cls_token, std=.02)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        B = x.shape[0]
        x = self.visual_tokens(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x


@register_model
def SoT_Tiny(pretrained=False, **kwargs):
    ViTConfig['embed_dim']=240
    ViTConfig['depth']=12
    ViTConfig['num_heads']=4
    ViTConfig['mlp_ratio']=2.5
    representationConfig['args']['dim']=240
    representationConfig['args']['num_heads']=6
    representationConfig['args']['wr_dim']=14
    representationConfig['args']['normalization']['input_dim']=14
    representationConfig['args']['normalization']['regular']=None
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = SoT(visualTokenConfig=visualTokenConfig, ViTConfig=ViTConfig, representationConfig=representationConfig, **kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def SoT_Small(pretrained=False, **kwargs):
    ViTConfig['embed_dim']=384
    ViTConfig['depth']=14
    ViTConfig['num_heads']=6
    ViTConfig['mlp_ratio']=3.5
    representationConfig['args']['dim']=384
    representationConfig['args']['num_heads']=6
    representationConfig['args']['wr_dim']=24
    representationConfig['args']['normalization']['input_dim']=24
    representationConfig['args']['normalization']['regular']=nn.Dropout(0.5)
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = SoT(visualTokenConfig=visualTokenConfig, ViTConfig=ViTConfig, representationConfig=representationConfig, **kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model

@register_model
def SoT_Base(pretrained=False, **kwargs):
    ViTConfig['embed_dim']=528
    ViTConfig['depth']=24
    ViTConfig['num_heads']=8
    ViTConfig['mlp_ratio']=3
    representationConfig['args']['dim']=528
    representationConfig['args']['num_heads']=6
    representationConfig['args']['wr_dim']=38
    representationConfig['args']['normalization']['input_dim']=38
    representationConfig['args']['normalization']['regular']=nn.Dropout(0.7)
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = SoT(visualTokenConfig=visualTokenConfig, ViTConfig=ViTConfig, representationConfig=representationConfig, **kwargs)
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model