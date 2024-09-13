import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones import SUPPORTED_BACKBONES


#------------------------------------------------------------------------------
#  MODNet Basic Modules
#------------------------------------------------------------------------------
# count = [-1]
class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        # # original
        # self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        # self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

        # the first modify(bn:all in:half)
        self.bnorm = nn.BatchNorm2d(in_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels,affine=False)
        
        # # the second modify(bn:all in:all)
        # self.bnorm = nn.BatchNorm2d(in_channels, affine=True)
        # self.inorm = nn.InstanceNorm2d(in_channels,affine=False)
        
    def forward(self, x):
        # # original
        # bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        # in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        # return torch.cat((bn_x, in_x), 1)

        # the first modify(bn:all in:half)
        bn_x = self.bnorm(x)
        # in_x = self.inorm(bn_x[:, self.bnorm_channels:, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())
        bn_x = bn_x[:, :self.bnorm_channels, ...].contiguous()
        return torch.cat((bn_x,in_x), 1)
        
        # # the second modify(bn:all in:all) TOOD
        # bn_x = self.bnorm(x)
        # in_x = self.inorm(x)
        # bn_x = bn_x[:, :self.bnorm_channels, ...].contiguous()
        # in_x = in_x[:, self.bnorm_channels:, ...].contiguous()
        # return torch.cat((bn_x,in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    """ Convolution + IBNorm + ReLu
    """

    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True, 
                 with_ibn=True, with_relu=True):
        super(Conv2dIBNormRelu, self).__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride=stride, padding=padding, dilation=dilation, 
                      groups=groups, bias=bias)
        ]

        if with_ibn:       
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) 


class SEBlock(nn.Module):
    """ SE Block Proposed in https://arxiv.org/pdf/1709.01507.pdf 
    """

    def __init__(self, in_channels, out_channels, reduction=1):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)

        return x * w.expand_as(x)


#------------------------------------------------------------------------------
#  MODNet Branches
#------------------------------------------------------------------------------

class LRBranch(nn.Module):
    """ Low Resolution Branch of MODNet
    """

    def __init__(self, backbone, cfg = None,count=None):
        super(LRBranch, self).__init__()
        enc_channels = backbone.enc_channels
        if cfg == None:
            self.backbone = backbone
            self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
            self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
            self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
            self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)
        else:
            self.backbone = backbone
            self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
            self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], cfg[count[-1]+1], 5, stride=1, padding=2)
            self.conv_lr8x = Conv2dIBNormRelu(cfg[count[-1]+1], cfg[count[-1]+2], 5, stride=1, padding=2)
            self.conv_lr = Conv2dIBNormRelu(cfg[count[-1]+2], 1, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)
            count[-1]+=2
    def forward(self, img, inference):
        enc_features = self.backbone.forward(img)
        enc2x, enc4x, enc32x = enc_features[0], enc_features[1], enc_features[4]

        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)

        pred_semantic = None
        if not inference:
            lr = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)

        return pred_semantic, lr8x, [enc2x, enc4x] 


class HRBranch(nn.Module):
    """ High Resolution Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels, cfg = None,count=None):
        super(HRBranch, self).__init__()
        if cfg == None:
            self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
            self.conv_enc2x = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)

            self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
            self.conv_enc4x = Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1)

            self.conv_hr4x = nn.Sequential(
                Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
                Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
                Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
            )

            self.conv_hr2x = nn.Sequential(
                Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
                Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
                Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
                Conv2dIBNormRelu(hr_channels, hr_channels, 3, stride=1, padding=1),
            )

            self.conv_hr = nn.Sequential(
                Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
                Conv2dIBNormRelu(hr_channels, 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
            )
        else:
            self.tohr_enc2x = Conv2dIBNormRelu(enc_channels[0], cfg[count[-1]+1], 1, stride=1, padding=0)
            self.conv_enc2x = Conv2dIBNormRelu(cfg[count[-1]+1] + 3, cfg[count[-1]+2], 3, stride=2, padding=1)

            self.tohr_enc4x = Conv2dIBNormRelu(enc_channels[1], cfg[count[-1]+3], 1, stride=1, padding=0)
            self.conv_enc4x = Conv2dIBNormRelu(cfg[count[-1]+2]+cfg[count[-1]+3], cfg[count[-1]+4], 3, stride=1, padding=1)

            self.conv_hr4x = nn.Sequential(
                Conv2dIBNormRelu(cfg[count[-1]+4]+cfg[1] + 3, cfg[count[-1]+5], 3, stride=1, padding=1),
                Conv2dIBNormRelu(cfg[count[-1]+5], cfg[count[-1]+6], 3, stride=1, padding=1),
                Conv2dIBNormRelu(cfg[count[-1]+6], cfg[count[-1]+7], 3, stride=1, padding=1),
            )

            self.conv_hr2x = nn.Sequential(
                Conv2dIBNormRelu(cfg[count[-1]+7]+cfg[count[-1]+1], cfg[count[-1]+8], 3, stride=1, padding=1),
                Conv2dIBNormRelu(cfg[count[-1]+8], cfg[count[-1]+9], 3, stride=1, padding=1),
                Conv2dIBNormRelu(cfg[count[-1]+9], cfg[count[-1]+10], 3, stride=1, padding=1),
                Conv2dIBNormRelu(cfg[count[-1]+10], cfg[count[-1]+11], 3, stride=1, padding=1),
            )

            self.conv_hr = nn.Sequential(
                Conv2dIBNormRelu(cfg[count[-1]+11] + 3, cfg[count[-1]+12], 3, stride=1, padding=1),
                Conv2dIBNormRelu(cfg[count[-1]+12], 1, kernel_size=1, stride=1, padding=0, with_ibn=False, with_relu=False),
            )
            count[-1]+=12


    def forward(self, img, enc2x, enc4x, lr8x, inference):
        img2x = F.interpolate(img, scale_factor=1/2, mode='bilinear', align_corners=False)
        img4x = F.interpolate(img, scale_factor=1/4, mode='bilinear', align_corners=False)

        enc2x = self.tohr_enc2x(enc2x)
        hr4x = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))

        enc4x = self.tohr_enc4x(enc4x)
        hr4x = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))

        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))

        hr2x = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))

        pred_detail = None
        if not inference:
            hr = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)
            hr = self.conv_hr(torch.cat((hr, img), dim=1))
            pred_detail = torch.sigmoid(hr)

        return pred_detail, hr2x


class FusionBranch(nn.Module):
    """ Fusion Branch of MODNet
    """

    def __init__(self, hr_channels, enc_channels,cfg = None,count=None):
        super(FusionBranch, self).__init__()
        if cfg == None:
            self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
            
            self.conv_f2x = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
            self.conv_f = nn.Sequential(
                Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
                Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
            )
        else:
            self.conv_lr4x = Conv2dIBNormRelu(cfg[1], cfg[count[-1]+1], 5, stride=1, padding=2)
            
            self.conv_f2x = Conv2dIBNormRelu(cfg[count[-1]+1]+ cfg[12], cfg[count[-1]+2], 3, stride=1, padding=1)
            self.conv_f = nn.Sequential(
                Conv2dIBNormRelu(cfg[count[-1]+2] + 3, cfg[count[-1]+3], 3, stride=1, padding=1),
                Conv2dIBNormRelu(cfg[count[-1]+3], 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
            )
            count[-1]+=3

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        lr2x = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)

        f2x = self.conv_f2x(torch.cat((lr2x, hr2x), dim=1))
        f = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        f = self.conv_f(torch.cat((f, img), dim=1))
        pred_matte = torch.sigmoid(f)

        return pred_matte


#------------------------------------------------------------------------------
#  MODNet
#------------------------------------------------------------------------------

class MODNet(nn.Module):
    """ Architecture of MODNet
    """

    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2', backbone_pretrained=True, cfg=None,cfg1=None):
        super(MODNet, self).__init__()
        count = [-1]
        self.in_channels = in_channels
        self.hr_channels = hr_channels
        self.backbone_arch = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = SUPPORTED_BACKBONES[self.backbone_arch](self.in_channels,cfg=cfg)

        self.lr_branch = LRBranch(self.backbone,cfg=cfg1,count=count)
        self.hr_branch = HRBranch(self.hr_channels, self.backbone.enc_channels,cfg=cfg1,count=count)
        self.f_branch = FusionBranch(self.hr_channels, self.backbone.enc_channels,cfg=cfg1,count=count)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self._init_conv(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                self._init_norm(m)

        if self.backbone_pretrained:
            self.backbone.load_pretrained_ckpt()                

    def forward(self, img, inference=False):
        pred_semantic, lr8x, [enc2x, enc4x] = self.lr_branch(img, inference)
        pred_detail, hr2x = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte = self.f_branch(img, lr8x, hr2x)

        return pred_semantic, pred_detail, pred_matte
    
    def freeze_norm(self):
        norm_types = [nn.BatchNorm2d, nn.InstanceNorm2d]
        for m in self.modules():
            for n in norm_types:
                if isinstance(m, n):
                    m.eval()
                    continue

    def _init_conv(self, conv):
        nn.init.kaiming_uniform_(
            conv.weight, a=0, mode='fan_in', nonlinearity='relu')
        if conv.bias is not None:
            nn.init.constant_(conv.bias, 0)

    def _init_norm(self, norm):
        if norm.weight is not None:
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
            
from nni.compression.pytorch.utils import count_flops_params
numoy_array = torch.rand(1,3,512,512)
# # # cfg = [32, 32, 16, 82, 82, 24, 113, 113, , 109, 109, 32, 102, 102, 32, 74, 74, 32, 108, 108, 64, 165, 165, 64, 94, 94, 64, 71, 71, 64, 145, 145, 96, 108, 108, 96, 104, 104, 96, 135, 135, 160, 195, 195, 160, 158, 158, 160, 208, 208, 320, 1280]
# # cfg = [32, 32, 16, 93, 93, 24, 131, 131, 24, 128, 128, 32, 138, 138, 32, 134, 134, 32, 138, 138, 64, 243, 243, 64, 172, 172, 64, 117, 117, 64, 214, 214, 96, 181, 181, 96, 167, 167, 96, 237, 237, 160, 308, 308, 160, 284, 284, 160, 556, 556, 320, 1280]

# cfg = None
# cfg1 = None
cfg = [32, 32, 16, 93, 93, 24, 131, 131, 24, 128, 128, 32, 138, 138, 32, 134, 134, 32, 138, 138, 64, 243, 243, 64, 172, 172, 64, 117, 117, 64, 214, 214, 96, 181, 181, 96, 167, 167, 96, 237, 237, 160, 308, 308, 160, 284, 284, 160, 556, 556, 320, 1280]
cfg1 = [30, 15, 5, 2, 15, 13, 31, 32, 16, 19, 11, 10, 11, 16, 15, 11, 8]
flops,param,_ = count_flops_params(MODNet(3,cfg = cfg, cfg1=cfg1,backbone_pretrained=False),numoy_array,verbose=True)
