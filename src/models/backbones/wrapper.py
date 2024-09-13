import os
from functools import reduce

import torch
import torch.nn as nn

from .mobilenetv2 import MobileNetV2


class BaseBackbone(nn.Module):
    """ Superclass of Replaceable Backbone Model for Semantic Estimation
    """

    def __init__(self, in_channels):
        super(BaseBackbone, self).__init__()
        self.in_channels = in_channels

        self.model = None
        self.enc_channels = []

    def forward(self, x):
        raise NotImplementedError

    def load_pretrained_ckpt(self):
        raise NotImplementedError


class MobileNetV2Backbone(BaseBackbone):
    """ MobileNetV2 Backbone 
    """

    def __init__(self, in_channels,cfg = None):
        super(MobileNetV2Backbone, self).__init__(in_channels)
        self.model = MobileNetV2(self.in_channels, alpha=1.0, expansion=6, num_classes=None,cfg=cfg)
        if cfg == None:
            self.enc_channels = [16, 24, 32, 96, 1280]
        else:
            self.enc_channels = [cfg[2], cfg[8], cfg[17], cfg[38], cfg[-1]]


    def forward(self, x):
        # x = reduce(lambda x, n: self.model.features[n](x), list(range(0, 2)), x)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        enc2x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(2, 4)), x)
        x = self.model.features[2](x)
        x = self.model.features[3](x)
        enc4x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(4, 7)), x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = self.model.features[6](x)
        enc8x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(7, 14)), x)
        x = self.model.features[7](x)
        x = self.model.features[8](x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = self.model.features[11](x)
        x = self.model.features[12](x)
        x = self.model.features[13](x)
        enc16x = x

        # x = reduce(lambda x, n: self.model.features[n](x), list(range(14, 19)), x)
        x = self.model.features[14](x)
        x = self.model.features[15](x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = self.model.features[18](x)
        enc32x = x
        return [enc2x, enc4x, enc8x, enc16x, enc32x]

    def load_pretrained_ckpt(self):
        # the pre-trained model is provided by https://github.com/thuyngch/Human-Segmentation-PyTorch 
        
        # the original model
        # ckpt_path = './pretrained/mobilenetv2_human_seg.ckpt'
        # the first prue
        # ckpt_path = './src/convert_new_v2/first_pruning_27_0.0134_v2_0.5.pth'
        ckpt_path = './src/convert_new_v2_gai/first_pruning_34_0.01456_v2_0.5.pth'
        # the second prue
        # ckpt_path = "./src/convert_new_v2/second_pruning_24_0.0167_v2_0.5.pth"
        
        # first pruning
        # ckpt_path = './src/convert_new_v2/first_pruning_18_0.016913_v2_0.5.pth'
        if not os.path.exists(ckpt_path):
            print('cannot find the pretrained mobilenetv2 backbone')
            exit()
        
        ckpt = torch.load(ckpt_path)

        self.model.load_state_dict(ckpt)
        # print(self.model.state_dict())
