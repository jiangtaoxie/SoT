import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from ..normalization import MPN
from ..normalization import svPN
from ..representation import MultiHeadPooling, Covariance


class Classifier(nn.Module):
    def __init__(self,num_classes=1000, input_dim=384, representationConfig={}):
        super(Classifier, self).__init__()
        self.fusion_type = representationConfig['fusion_type']
        self.re_type = representationConfig['type']
        if self.re_type == 'MGCrP':
            self.representation = MultiHeadPooling(**representationConfig['args'])
            output_dim = self.representation.output_dim
            if self.fusion_type == 'concat':
                output_dim = input_dim + output_dim 
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'GAP':
            self.representation = nn.AdaptiveAvgPool1d(1)
            output_dim = input_dim
            if self.fusion_type == 'concat':
                output_dim = input_dim*2
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'GCP':
            cov_dim = int(pow(2 * representationConfig['args']['num_heads'] * representationConfig['args']['qkv_dim'] ** 2, 0.5))
            self.dr = nn.Linear(input_dim, cov_dim)
            self.representation = Covariance(remove_mean=True)
            self.normalization = MPN(input_dim=cov_dim)
            if self.fusion_type == 'concat':
                output_dim = input_dim + int(cov_dim * (cov_dim + 1) / 2)
            else:
                output_dim = int(cov_dim * (cov_dim + 1) / 2)
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'fc':
            self.dim = nn.Linear(input_dim, 1280)
            input_dim = 1280

        self.cls_fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):  
        cls_token = x[:,0] # B*1*C
        vis_token = x[:,1:] # B*N*C
        if self.re_type is not None:
            if self.re_type == 'GAP':
                if self.fusion_type != 'aggre_all':
                    x = vis_token
                    cls_head = self.cls_fc(cls_token)                  
                x = x.transpose(-1, -2)
                x = self.representation(x)
                x = x.view(x.size(0), -1)
            if self.re_type == 'GCP':
                if self.fusion_type != 'aggre_all':
                    x = vis_token
                    cls_head = self.cls_fc(cls_token)
                x = self.dr(x)
                x = self.representation(x)
                x = self.normalization(x)
                x = x.view(x.size(0), -1)
            if self.re_type == 'MGCrP':
                if self.fusion_type != 'aggre_all':
                    x = vis_token
                    cls_head = self.cls_fc(cls_token)
                x = self.representation(x)
                x = x.view(x.size(0), -1)
            if self.re_type == 'fc':
                x = cls_token
                x = self.dim(x)
                x = self.cls_fc(x)
                return x
            if self.fusion_type == 'aggre_all':
                head = self.visual_fc(x)
                return head
            elif self.fusion_type == 'concat':
                x = torch.cat([cls_token, x], dim=1)
                head = self.visual_fc(x)
                return head
            elif self.fusion_type == 'sum_fc':
                vis_head = self.visual_fc(x)
                return vis_head + cls_head
            elif self.fusion_type == 'sum_loss':
                vis_head = self.visual_fc(x)
                return [cls_head, vis_head]
            return x
        else:
            head = self.cls_fc(cls_token)
            return head
            

class OnlyVisualTokensClassifier(nn.Module):
    def __init__(self,num_classes=1000, input_dim=384, representationConfig={}):
        super(OnlyVisualTokensClassifier, self).__init__()
        self.fusion_type = representationConfig['fusion_type'] # unuse argument
        self.re_type = representationConfig['type']
        if self.re_type == 'MGCrP':
            self.representation = MultiHeadPooling(**representationConfig['args'])
            output_dim = self.representation.output_dim
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'GAP':
            self.representation = nn.AdaptiveAvgPool1d(1)
            output_dim = input_dim
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'GCP':
            cov_dim = int(pow(2 * representationConfig['args']['num_heads'] * representationConfig['args']['qkv_dim'] ** 2, 0.5))
            self.dr = nn.Linear(input_dim, cov_dim)
            self.representation = Covariance(remove_mean=True)
            self.normalization = MPN(input_dim=cov_dim)
            output_dim = int(cov_dim * (cov_dim + 1) / 2)
            self.visual_fc = nn.Linear(output_dim, num_classes) if num_classes > 0 else nn.Identity()
        elif self.re_type == 'fc':
            self.dim = nn.Linear(input_dim, 1280)
            input_dim = 1280


    def forward(self, x):
        if self.re_type == 'GAP':
            x = x.transpose(-1, -2)
            x = self.representation(x)
            x = x.view(x.size(0), -1)
        if self.re_type == 'GCP':
            x = self.dr(x)
            x = self.representation(x)
            x = self.normalization(x)
            x = x.view(x.size(0), -1)
        if self.re_type == 'MGCrP':
            x = self.representation(x)
            x = x.view(x.size(0), -1)
        if self.re_type == 'fc':
            x = self.dim(x)
        output = self.visual_fc(x)
        return output               
            