import os
import sys
import math
import numpy as np


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.backbone.resnet import resnet50
from public.detection.models.head import FCOSClsRegCntHead
from public.detection.models.anchor import FCOSPositions

import torch
import torch.nn as nn
import torch.nn.functional as F




# assert input annotations are[x_min,y_min,x_max,y_max]
class FCOS(nn.Module):
    def __init__(self, config=None):
        super(FCOS, self).__init__()
        self.backbone=resnet50(pretrained=config.pretrained,if_include_top=False)
        if config.use_yolof:
            self.fpn=YolofDC5FPN(C5_inplanes=config.C5_inplanes)
        else:
            self.fpn=FPN(config.fpn_out_channels,use_p5=config.use_p5)
        self.head=ClsCntRegHead(config.fpn_out_channels,config.class_num,
                                config.use_GN_head,config.cnt_on_reg,config.prior)

        self.num_classes = config.class_num
        self.planes = config.fpn_out_channels

        self.clsregcnt_head = FCOSClsRegCntHead(self.planes,
                                                self.num_classes,
                                                num_layers=4,
                                                prior=0.01,
                                                use_gn=True,
                                                cnt_on_reg=True)

        self.strides = torch.tensor(config.strides, dtype=torch.float)
        self.positions = FCOSPositions(self.strides)

        self.scales = nn.Parameter(
            torch.tensor(config.scales, dtype=torch.float32))
        
    
    def train(self,mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)
        def freeze_bn(module):
            if isinstance(module,nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad=False
        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self, inputs):
        self.batch_size, _, _, _ = inputs.shape
        device = inputs.device
        [C3, C4, C5] = self.backbone(inputs)

        del inputs

        features = self.fpn([C3, C4, C5])

        del C3, C4, C5

        self.fpn_feature_sizes = []
        cls_heads, reg_heads, center_heads = [], [], []
        for feature, scale in zip(features, self.scales):
            self.fpn_feature_sizes.append([feature.shape[3], feature.shape[2]])

            cls_outs, reg_outs, center_outs = self.clsregcnt_head(feature)

            # [N,num_classes,H,W] -> [N,H,W,num_classes]
            cls_outs = cls_outs.permute(0, 2, 3, 1).contiguous()
            cls_heads.append(cls_outs)
            # [N,4,H,W] -> [N,H,W,4]
            reg_outs = reg_outs.permute(0, 2, 3, 1).contiguous()
            reg_outs = reg_outs * torch.exp(scale)
            reg_heads.append(reg_outs)
            # [N,1,H,W] -> [N,H,W,1]
            center_outs = center_outs.permute(0, 2, 3, 1).contiguous()
            center_heads.append(center_outs)

        del features

        self.fpn_feature_sizes = torch.tensor(
            self.fpn_feature_sizes).to(device)

        batch_positions = self.positions(self.batch_size,
                                         self.fpn_feature_sizes)

        # if input size:[B,3,640,640]
        # features shape:[[B, 256, 80, 80],[B, 256, 40, 40],[B, 256, 20, 20],[B, 256, 10, 10],[B, 256, 5, 5]]
        # cls_heads shape:[[B, 80, 80, 80],[B, 40, 40, 80],[B, 20, 20, 80],[B, 10, 10, 80],[B, 5, 5, 80]]
        # reg_heads shape:[[B, 80, 80, 4],[B, 40, 40, 4],[B, 20, 20, 4],[B, 10, 10, 4],[B, 5, 5, 4]]
        # center_heads shape:[[B, 80, 80, 1],[B, 40, 40, 1],[B, 20, 20, 1],[B, 10, 10, 1],[B, 5, 5, 1]]
        # batch_positions shape:[[B, 80, 80, 2],[B, 40, 40, 2],[B, 20, 20, 2],[B, 10, 10, 2],[B, 5, 5, 2]]

        return cls_heads, reg_heads, center_heads, batch_positions


def _fcos(arch, pretrained, **kwargs):
    model = FCOS(arch, **kwargs)
    # only load state_dict()
    if pretrained:
        pretrained_models = torch.load(model_urls[arch + "_fcos"],
                                       map_location=torch.device('cpu'))
        # del pretrained_models['cls_head.cls_head.8.weight']
        # del pretrained_models['cls_head.cls_head.8.bias']

        # only load state_dict()
        model.load_state_dict(pretrained_models, strict=False)

    return model


def resnet18_fcos(pretrained=False, **kwargs):
    return _fcos('resnet18', pretrained, **kwargs)


def resnet34_fcos(pretrained=False, **kwargs):
    return _fcos('resnet34', pretrained, **kwargs)


def resnet50_fcos(pretrained=False, **kwargs):
    return _fcos('resnet50', pretrained, **kwargs)


def resnet101_fcos(pretrained=False, **kwargs):
    return _fcos('resnet101', pretrained, **kwargs)


def resnet152_fcos(pretrained=False, **kwargs):
    return _fcos('resnet152', pretrained, **kwargs)


if __name__ == '__main__':
    net = FCOS(resnet_type="resnet50")
    image_h, image_w = 600, 600
    cls_heads, reg_heads, center_heads, batch_positions = net(
        torch.autograd.Variable(torch.randn(3, 3, image_h, image_w)))
    annotations = torch.FloatTensor([[[113, 120, 183, 255, 5],
                                      [13, 45, 175, 210, 2]],
                                     [[11, 18, 223, 225, 1],
                                      [-1, -1, -1, -1, -1]],
                                     [[-1, -1, -1, -1, -1],
                                      [-1, -1, -1, -1, -1]]])

    print("1111", cls_heads[0].shape, reg_heads[0].shape,
          center_heads[0].shape, batch_positions[0].shape)