import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)
from src.models.modnet import MODNet
import torch
import torch.nn as nn
import numpy as np
from src.models.backbones.mobilenetv2 import InvertedResidual, MobileNetV2
from src.models.modnet import LRBranch,HRBranch,FusionBranch


# cfg = None
# cfg1 = None
cfg = [32, 32, 16, 94, 94, 24, 133, 133, 24, 132, 132, 32, 158, 158, 32, 148, 148, 32, 160, 160, 64, 289, 289, 64, 270, 270, 64, 235, 235, 64, 290, 290, 96, 375, 375, 96, 365, 365, 96, 391, 391, 160, 647, 647, 160, 615, 615, 160, 660, 660, 320, 1280]
cfg1 = [45, 16, 16, 16, 32, 32, 32, 16, 32, 16, 16, 16, 16, 13, 8]

ratio_mobilenetv2 = 0.4
ratio_modnet_other = 0.5

# first pruning and the best accuracy of the model
pruned_model_save = "./src/convert_new/1pruning_31_0.01181_{}_{}.pth".format(ratio_mobilenetv2,ratio_modnet_other)
pruned_model_save_v2 = "./src/convert_new_v2/1pruning_31_0.01181_v2_{}.pth".format(ratio_mobilenetv2)
# pruned_model_save = "./src/convert_new/1pruning_31_0.01181_0.6_0.5.pth"
# pruned_model_save_v2 = "./src/convert_new_v2/1pruning_31_0.01181_v2_0.6.pth"
# model_path = "./the_another_model/2_the_again_train_with_l1/new_trimap_0016_lr0.001.pth"
# model_path = "./model_save/model_M_1pruning_train/new_trimap_0019_lr0.001.pth"
model_path = "./model_save/model_15M_2pruning_train/new_trimap_0031_lr1e-05.pth"
weight = torch.load(model_path)
model = torch.nn.DataParallel(MODNet(backbone_pretrained=False,cfg=cfg,cfg1=cfg1))
model.cuda()
model.load_state_dict(weight)

# ratio_mobilenetv2 = 0.6
# ratio_modnet_other = 0.5

# 针对mobilenetv2部分的处理，主要是对残差模块，因为mobilenetv2是加入了残差模块，所以对v2每个模块最后一层不进行处理
total = 0
highest_thre = []
for m in model.modules():
    if isinstance(m, MobileNetV2):
        total += m.features[0][1].weight.data.shape[0]
        highest_thre.append(m.features[0][1].weight.data.abs().max().item())
        for m1 in m.modules():
            if isinstance(m1, InvertedResidual):
                if len(m1.conv) > 5:
                    total += m1.conv[1].weight.data.shape[0]
                    highest_thre.append(m1.conv[1].weight.data.abs().max().item())
                    total += m1.conv[4].weight.data.shape[0]
                    highest_thre.append(m1.conv[4].weight.data.abs().max().item())
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, MobileNetV2):
        size = m.features[0][1].weight.data.shape[0]
        bn[index:(index +size)] = m.features[0][1].weight.data.abs().clone()
        index+=size
        for m1 in m.modules():
            if isinstance(m1, InvertedResidual):
                if len(m1.conv)>5:
                    for i in range(len(m1.conv)-1):
                        if isinstance(m1.conv[i], nn.BatchNorm2d):
                            size = m1.conv[i].weight.data.shape[0]
                            bn[index:(index+size)] = m1.conv[i].weight.data.abs().clone()
                            index+=size
# 对于mobilenetv2部分选取阈值
y, i = torch.sort(bn)
thre_index = int(total*ratio_mobilenetv2)
thre = y[thre_index]
highest_thre = min(highest_thre)
if thre>highest_thre:
    thre = highest_thre- 1e-5
    print("The thre of mobilenetv2 use highest_thre")

# mobilenetv2部分进行mask的确定
pruned = 0
cfg = []
cfg_mask = []
idx = 0
for m in model.modules():
    if isinstance(m, MobileNetV2):
        weight_copy = m.features[0][1].weight.data.clone()
        mask = weight_copy.abs().gt(thre).float().cuda()
        pruned = pruned + (mask.shape[0] - torch.sum(mask))
        m.features[0][1].weight.data.mul_(mask)
        m.features[0][1].bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())

        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        for m1 in m.modules():
            if isinstance(m1, InvertedResidual):
                if len(m1.conv)>5:
                    weight_copy = m1.conv[1].weight.data.clone()
                    mask = weight_copy.abs().gt(thre).float().cuda()
                    # pruned = pruned + mask.shape[0] - torch.sum(mask)
                    pruned = pruned + (mask.shape[0] - torch.sum(mask))*2
                    # first conv + bn
                    m1.conv[1].weight.data.mul_(mask)
                    m1.conv[1].bias.data.mul_(mask)
                    # second conv + bn
                    m1.conv[4].weight.data.mul_(mask)
                    m1.conv[4].bias.data.mul_(mask)
                    cfg.append(int(torch.sum(mask)))
                    cfg_mask.append(mask.clone())
                    cfg.append(int(torch.sum(mask)))
                    cfg_mask.append(mask.clone())
                    cfg.append(m1.conv[7].weight.data.shape[0])
                    cfg_mask.append(torch.ones(m1.conv[7].weight.data.shape[0]))

                    # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(idx, mask.shape[0], int(torch.sum(mask))))
                    idx += 1
                else:
                    idx+=1
                    # cfg.append(m1.conv[1].weight.data.shape[0])
                    # cfg_mask.append(torch.ones(m1.conv[1].weight.data.shape[0]))
                    cfg.append(m1.conv[4].weight.data.shape[0])
                    cfg_mask.append(torch.ones(m1.conv[4].weight.data.shape[0]))
                    # cfg.append()
        cfg.append(m.features[18][1].weight.data.shape[0])
        cfg_mask.append(torch.ones(m.features[18][1].weight.data.shape[0]))

# modnet其余部分需要处理的层进行权重进行加入
total1 = 0
highest_thre1 = []
for m in model.modules():
    if isinstance(m, LRBranch):
        # conv_lr16x
        total1 += m.conv_lr16x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_lr16x.layers[1].bnorm.weight.data.abs())[-2].item())
    elif isinstance(m, HRBranch):
        # tohr_enc2x
        total1 += m.tohr_enc2x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.tohr_enc2x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_enc2x
        total1 += m.conv_enc2x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_enc2x.layers[1].bnorm.weight.data.abs())[-2].item())
        # tohr_enc4x
        total1+=m.tohr_enc4x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.tohr_enc4x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_enc4x
        total1+=m.conv_enc4x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_enc4x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_hr4x
        total1+=m.conv_hr4x[0].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr4x[0].layers[1].bnorm.weight.data.abs())[-2].item())
        total1+=m.conv_hr4x[1].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr4x[1].layers[1].bnorm.weight.data.abs())[-2].item())
        total1+=m.conv_hr4x[2].layers[1].bnorm.weight.data.shape[0]
        # print(sorted(m.conv_hr4x[2].layers[1].bnorm.weight.data.abs())[-2])
        highest_thre1.append(sorted(m.conv_hr4x[2].layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_hr2x
        total1+=m.conv_hr2x[0].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr2x[0].layers[1].bnorm.weight.data.abs())[-2].item())
        total1+=m.conv_hr2x[1].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr2x[1].layers[1].bnorm.weight.data.abs())[-2].item())
        total1+=m.conv_hr2x[2].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr2x[2].layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_hr
        total1+=m.conv_hr[0].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr[0].layers[1].bnorm.weight.data.abs())[-2].item())
    elif isinstance(m,FusionBranch):
        # conv_lr4x
        total1+=m.conv_lr4x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_lr4x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_f2x
        total1+=m.conv_f2x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_f2x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_f
        total1+=m.conv_f[0].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_f[0].layers[1].bnorm.weight.data.abs())[-2].item())

bn1 = torch.zeros(total1)
index1 = 0
for m in model.modules():
    if isinstance(m, LRBranch):
        # conv_lr16x
        size= m.conv_lr16x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)] = m.conv_lr16x.layers[1].bnorm.weight.data.abs().clone()
        index1+=size
    elif isinstance(m, HRBranch):
        # tohr_enc2x
        size= m.tohr_enc2x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)] = m.tohr_enc2x.layers[1].bnorm.weight.data.abs().clone()
        index1+=size
        # conv_enc2x
        size= m.conv_enc2x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)] = m.conv_enc2x.layers[1].bnorm.weight.data.abs().clone()
        index1+=size
        # tohr_enc4x
        size=m.tohr_enc4x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)] = m.tohr_enc4x.layers[1].bnorm.weight.data.abs().clone()
        index1+=size
        # conv_enc4x
        size=m.conv_enc4x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)] = m.conv_enc4x.layers[1].bnorm.weight.data.abs().clone()
        index1+=size
        # conv_hr4x
        size=m.conv_hr4x[0].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)] = m.conv_hr4x[0].layers[1].bnorm.weight.data.abs().clone()
        index1+=size
        
        size=m.conv_hr4x[1].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)] = m.conv_hr4x[1].layers[1].bnorm.weight.data.abs().clone()
        index1+=size

        size=m.conv_hr4x[2].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)]=m.conv_hr4x[2].layers[1].bnorm.weight.data.abs().clone()
        index1+=size

        # conv_hr2x
        size=m.conv_hr2x[0].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)]=m.conv_hr2x[0].layers[1].bnorm.weight.data.abs().clone()
        index1+=size

        size=m.conv_hr2x[1].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)]=m.conv_hr2x[1].layers[1].bnorm.weight.data.abs().clone()
        index1+=size

        size=m.conv_hr2x[2].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)]=m.conv_hr2x[2].layers[1].bnorm.weight.data.abs().clone()
        index1+=size
        # conv_hr
        size=m.conv_hr[0].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)]=m.conv_hr[0].layers[1].bnorm.weight.data.abs().clone()
        index1+=size
    elif isinstance(m,FusionBranch):
        # conv_lr4x
        size=m.conv_lr4x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)]=m.conv_lr4x.layers[1].bnorm.weight.data.abs().clone()
        index1+=size
        # conv_f2x
        size=m.conv_f2x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)]=m.conv_f2x.layers[1].bnorm.weight.data.abs().clone()
        index1+=size
        # conv_f
        size=m.conv_f[0].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1+size)]=m.conv_f[0].layers[1].bnorm.weight.data.abs().clone()
# modnet其余部分进行阈值选取
y1,i1 = torch.sort(bn1)
thre_index1 = int(total1*ratio_modnet_other)
thre1 = y1[thre_index1]
highest_thre1 = min(highest_thre1)
if thre1>=highest_thre1:
    thre1 = highest_thre1 - 1e-5
    print("The other of MODNet use highest_thre1")

# modnet其余部分进行mask确定
pruned1 = 0
cfg1 = []
cfg_mask1 = {}
idx = 0
for m in model.modules():
    if isinstance(m, LRBranch):
        # conv_lr16x
        weight_copy = m.conv_lr16x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_lr16x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_lr16x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_lr6x"] = mask.clone()
    elif isinstance(m, HRBranch):
        # tohr_enc2x
        weight_copy= m.tohr_enc2x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.tohr_enc2x.layers[1].bnorm.weight.data.mul_(mask)
        m.tohr_enc2x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["tohr_enc2x"] = mask.clone()
        # conv_enc2x
        weight_copy= m.conv_enc2x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_enc2x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_enc2x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_enc2x"] = mask.clone()
        # tohr_enc4x
        weight_copy=m.tohr_enc4x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.tohr_enc4x.layers[1].bnorm.weight.data.mul_(mask)
        m.tohr_enc4x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["tohr_enc4x"] = mask.clone()
        # conv_enc4x
        weight_copy=m.conv_enc4x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_enc4x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_enc4x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_enc4x"] = mask.clone()
        # conv_hr4x
        weight_copy=m.conv_hr4x[0].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr4x[0].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr4x[0].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr4x.0"] = mask.clone()
        
        weight_copy=m.conv_hr4x[1].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr4x[1].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr4x[1].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr4x.1"] = mask.clone()

        weight_copy=m.conv_hr4x[2].layers[1].bnorm.weight.data.clone()
        mask=weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1+ (mask.shape[0] - torch.sum(mask))
        m.conv_hr4x[2].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr4x[2].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr4x.2"] = mask.clone()

        # conv_hr2x
        weight_copy=m.conv_hr2x[0].layers[1].bnorm.weight.data.clone()
        mask=weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr2x[0].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr2x[0].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr2x.0"] = mask.clone()

        weight_copy=m.conv_hr2x[1].layers[1].bnorm.weight.data.clone()
        mask=weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr2x[1].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr2x[1].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr2x.1"] = mask.clone()

        weight_copy=m.conv_hr2x[2].layers[1].bnorm.weight.data.clone()
        mask=weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr2x[2].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr2x[2].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr2x.2"] = mask.clone()
        # conv_hr
        weight_copy=m.conv_hr[0].layers[1].bnorm.weight.data.clone()
        mask=weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr[0].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr[0].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr.0"] = mask.clone()
    elif isinstance(m,FusionBranch):
        # conv_lr4x
        weight_copy=m.conv_lr4x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_lr4x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_lr4x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_lr4x"] = mask.clone()
        # conv_f2x
        weight_copy=m.conv_f2x.layers[1].bnorm.weight.data.clone()
        mask=weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_f2x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_f2x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_f2x"] = mask.clone()
        # conv_f
        weight_copy=m.conv_f[0].layers[1].bnorm.weight.data.clone()
        mask=weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_f[0].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_f[0].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_f.0"] = mask.clone()
# print(cfg1)
# print(len(cfg))
# print(pruned1/total1)

# mobilenetv2部分进行剪枝
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
new_model = torch.nn.DataParallel(MODNet(backbone_pretrained=False,cfg=cfg,cfg1=cfg1))
new_model.cuda()
new_model_backbone = new_model.module.backbone.model
model_backbone = model.module.backbone.model
for [m0, m1] in zip(model_backbone.modules(), new_model_backbone.modules()):
    if isinstance(m0, nn.BatchNorm2d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[idx1.tolist()].clone()
        m1.bias.data = m0.bias.data[idx1.tolist()].clone()
        m1.running_mean = m0.running_mean[idx1.tolist()].clone()
        m1.running_var = m0.running_var[idx1.tolist()].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]
    elif isinstance(m0, nn.Conv2d):
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        # 下一层BN层对应的输入mask为start_mask
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        if m0.groups == 1:
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        else:
            w1 = m0.weight.data.clone()
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
# print(new_model.module.backbone.model.features[0][0].weight.data)

# modnet其余部分进行剪枝
new_model_lr_branch_se_block = new_model.module.lr_branch.se_block
model_lr_branch_se_block = model.module.lr_branch.se_block
for [m0, m1] in zip(model_lr_branch_se_block.modules(),new_model_lr_branch_se_block.modules()):
    if isinstance(m0, nn.Linear):
        w1 = m0.weight.data.clone()
        m1.weight.data = w1.clone()
### lr_branch
# conv_lr16x
    # conv2d
new_model_lr_branch_conv_lr16x = new_model.module.lr_branch.conv_lr16x
model_lr_branch_conv_lr16x = model.module.lr_branch.conv_lr16x
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr6x"].cpu().numpy())))
w1 = model_lr_branch_conv_lr16x.layers[0].weight.data.clone()
b1 = model_lr_branch_conv_lr16x.layers[0].bias.data.clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
b1 = b1[idx1.tolist()].clone()
new_model_lr_branch_conv_lr16x.layers[0].weight.data = w1.clone()
new_model_lr_branch_conv_lr16x.layers[0].bias.data = b1.clone()
    #bnorm 
new_model_lr_branch_conv_lr16x.layers[1].bnorm.weight.data = model_lr_branch_conv_lr16x.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_lr_branch_conv_lr16x.layers[1].bnorm.bias.data = model_lr_branch_conv_lr16x.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_lr_branch_conv_lr16x.layers[1].bnorm.running_mean = model_lr_branch_conv_lr16x.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_lr_branch_conv_lr16x.layers[1].bnorm.running_var = model_lr_branch_conv_lr16x.layers[1].bnorm.running_var[idx1.tolist()].clone()

# conv_lr8x
    # conv2d
new_model_lr_branch_conv_lr8x = new_model.module.lr_branch.conv_lr8x
model_lr_branch_conv_lr8x = model.module.lr_branch.conv_lr8x
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr6x"].cpu().numpy())))
w1 = model_lr_branch_conv_lr8x.layers[0].weight.data.clone()
w1 = w1[:,idx0.tolist(), :, :].clone()
new_model_lr_branch_conv_lr8x.layers[0].weight.data = w1.clone()
new_model_lr_branch_conv_lr8x.layers[0].bias.data = model_lr_branch_conv_lr8x.layers[0].bias.data.clone()

    # bnorm
new_model_lr_branch_conv_lr8x.layers[1].bnorm.weight.data = model_lr_branch_conv_lr8x.layers[1].bnorm.weight.data.clone()
new_model_lr_branch_conv_lr8x.layers[1].bnorm.bias.data = model_lr_branch_conv_lr8x.layers[1].bnorm.bias.data.clone()
new_model_lr_branch_conv_lr8x.layers[1].bnorm.running_mean = model_lr_branch_conv_lr8x.layers[1].bnorm.running_mean.clone()
new_model_lr_branch_conv_lr8x.layers[1].bnorm.running_var = model_lr_branch_conv_lr8x.layers[1].bnorm.running_var.clone()

# conv_lr
    # conv
new_model_lr_branch_conv_lr = new_model.module.lr_branch.conv_lr
model_lr_branch_conv_lr = model.module.lr_branch.conv_lr
new_model_lr_branch_conv_lr.layers[0].weight.data = model_lr_branch_conv_lr.layers[0].weight.data.clone()
new_model_lr_branch_conv_lr.layers[0].bias.data = model_lr_branch_conv_lr.layers[0].bias.data.clone()


### hr_branch
# tohr_enc2x
    # conv
new_model_hr_branch_tohr_enc2x = new_model.module.hr_branch.tohr_enc2x
model_hr_branch_tohr_enc2x = model.module.hr_branch.tohr_enc2x
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["tohr_enc2x"].cpu().numpy())))
w1 = model_hr_branch_tohr_enc2x.layers[0].weight.data.clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
b1 = model_hr_branch_tohr_enc2x.layers[0].bias.data.clone()
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc2x.layers[0].weight.data = w1.clone()
new_model_hr_branch_tohr_enc2x.layers[0].bias.data = b1.clone()

    # bnorm
new_model_hr_branch_tohr_enc2x.layers[1].bnorm.weight.data = model_hr_branch_tohr_enc2x.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc2x.layers[1].bnorm.bias.data = model_hr_branch_tohr_enc2x.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc2x.layers[1].bnorm.running_mean = model_hr_branch_tohr_enc2x.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc2x.layers[1].bnorm.running_var = model_hr_branch_tohr_enc2x.layers[1].bnorm.running_var[idx1.tolist()].clone()

    # conv_enc2x
new_model_hr_branch_conv_enc2x = new_model.module.hr_branch.conv_enc2x
model_hr_branch_conv_enc2x = model.module.hr_branch.conv_enc2x
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["tohr_enc2x"].cpu().numpy())))+3
img1 = np.squeeze(np.argwhere(np.ones(3)))
idx0 = np.hstack((img1,idx0))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_enc2x"].cpu().numpy())))
w1 = model_hr_branch_conv_enc2x.layers[0].weight.data.clone()
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_enc2x.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_enc2x.layers[0].bias.data.clone()
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_enc2x.layers[0].bias.data = b1.clone()
    # bnorm
new_model_hr_branch_conv_enc2x.layers[1].bnorm.weight.data = model_hr_branch_conv_enc2x.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_enc2x.layers[1].bnorm.bias.data = model_hr_branch_conv_enc2x.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_enc2x.layers[1].bnorm.running_mean = model_hr_branch_conv_enc2x.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_enc2x.layers[1].bnorm.runnign_var = model_hr_branch_conv_enc2x.layers[1].bnorm.running_var[idx1.tolist()].clone()

    # tohr_enc4x
new_model_hr_branch_tohr_enc4x = new_model.module.hr_branch.tohr_enc4x
model_hr_branch_tohr_enc4x = model.module.hr_branch.tohr_enc4x
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["tohr_enc4x"].cpu().numpy())))
w1 = model_hr_branch_tohr_enc4x.layers[0].weight.data.clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_tohr_enc4x.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_tohr_enc4x.layers[0].bias.data.clone()
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc4x.layers[0].bias.data = b1.clone()
    # bnorm
new_model_hr_branch_tohr_enc4x.layers[1].bnorm.weight.data = model_hr_branch_tohr_enc4x.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc4x.layers[1].bnorm.bias.data = model_hr_branch_tohr_enc4x.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc4x.layers[1].bnorm.running_mean = model_hr_branch_tohr_enc4x.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc4x.layers[1].bnorm.running_var = model_hr_branch_tohr_enc4x.layers[1].bnorm.running_var[idx1.tolist()].clone()

    # conv_enc4x
new_model_hr_branch_conv_enc4x = new_model.module.hr_branch.conv_enc4x
model_hr_branch_conv_enc4x = model.module.hr_branch.conv_enc4x
idx0 = np.squeeze(np.argwhere(np.hstack((np.asarray(cfg_mask1["conv_enc2x"].cpu().numpy()),np.asarray(cfg_mask1["tohr_enc4x"].cpu().numpy())))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_enc4x"].cpu().numpy())))
w1 = model_hr_branch_conv_enc4x.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_enc4x.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_enc4x.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_enc4x.layers[0].bias.data = b1.clone()
    # bnorm
new_model_hr_branch_conv_enc4x.layers[1].bnorm.weight.data = model_hr_branch_conv_enc4x.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_enc4x.layers[1].bnorm.bias.data = model_hr_branch_conv_enc4x.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_enc4x.layers[1].bnorm.running_mean = model_hr_branch_conv_enc4x.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_enc4x.layers[1].bnorm.running_var = model_hr_branch_conv_enc4x.layers[1].bnorm.running_var[idx1.tolist()].clone()

    # conv_hr4x
        # 0
new_model_hr_branch_conv_hr4x_0 = new_model.module.hr_branch.conv_hr4x[0]
model_hr_branch_conv_hr4x_0 = model.module.hr_branch.conv_hr4x[0]
idx0 = np.squeeze(np.argwhere(np.hstack((np.asarray(cfg_mask1["conv_enc4x"].cpu().numpy()),np.asarray(np.ones(35))))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.0"].cpu().numpy())))
w1 = model_hr_branch_conv_hr4x_0.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_hr4x_0.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr4x_0.layers[0].bias.data
b1 = b1[idx1.tolist()]
new_model_hr_branch_conv_hr4x_0.layers[0].bias.data = b1.clone()
    # bnorm
new_model_hr_branch_conv_hr4x_0.layers[1].bnorm.weight.data = model_hr_branch_conv_hr4x_0.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_0.layers[1].bnorm.bias.data = model_hr_branch_conv_hr4x_0.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_0.layers[1].bnorm.running_mean = model_hr_branch_conv_hr4x_0.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_0.layers[1].bnorm.running_var = model_hr_branch_conv_hr4x_0.layers[1].bnorm.running_var[idx1.tolist()].clone()

        # 1
new_model_hr_branch_conv_hr4x_1 = new_model.module.hr_branch.conv_hr4x[1]
model_hr_branch_conv_hr4x_1 = model.module.hr_branch.conv_hr4x[1]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.0"].cpu().numpy())))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.1"].cpu().numpy())))
w1 = model_hr_branch_conv_hr4x_1.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_hr4x_1.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr4x_1.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_1.layers[0].bias.data = b1.clone()
        # bnorm
new_model_hr_branch_conv_hr4x_1.layers[1].bnorm.weight.data = model_hr_branch_conv_hr4x_1.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_1.layers[1].bnorm.bias.data = model_hr_branch_conv_hr4x_1.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_1.layers[1].bnorm.running_mean = model_hr_branch_conv_hr4x_1.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_1.layers[1].bnorm.running_var = model_hr_branch_conv_hr4x_1.layers[1].bnorm.running_var[idx1.tolist()].clone()

        # 2
new_model_hr_branch_conv_hr4x_2 = new_model.module.hr_branch.conv_hr4x[2]
model_hr_branch_conv_hr4x_2 = model.module.hr_branch.conv_hr4x[2]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.1"].cpu().numpy())))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.2"].cpu().numpy())))
w1 = model_hr_branch_conv_hr4x_2.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_hr4x_2.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr4x_2.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_2.layers[0].bias.data = b1.clone()
        # bnorm
new_model_hr_branch_conv_hr4x_2.layers[1].bnorm.weight.data = model_hr_branch_conv_hr4x_2.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_2.layers[1].bnorm.bias.data = model_hr_branch_conv_hr4x_2.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_2.layers[1].bnorm.running_mean = model_hr_branch_conv_hr4x_2.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_2.layers[1].bnorm.running_var = model_hr_branch_conv_hr4x_2.layers[1].bnorm.running_var[idx1.tolist()].clone()

    # conv_hr2x
        # 0
new_model_hr_branch_conv_hr2x_0 = new_model.module.hr_branch.conv_hr2x[0]
model_hr_branch_conv_hr2x_0 = model.module.hr_branch.conv_hr2x[0]
idx0 = np.squeeze(np.argwhere(np.hstack((np.asarray(cfg_mask1["conv_hr4x.2"].cpu().numpy()),np.asarray(cfg_mask1["tohr_enc2x"].cpu().numpy())))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.0"].cpu().numpy())))
w1 = model_hr_branch_conv_hr2x_0.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_hr2x_0.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr2x_0.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_0.layers[0].bias.data = b1.clone()
        # bnorm
new_model_hr_branch_conv_hr2x_0.layers[1].bnorm.weight.data = model_hr_branch_conv_hr2x_0.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_0.layers[1].bnorm.bias.data = model_hr_branch_conv_hr2x_0.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_0.layers[1].bnorm.running_mean = model_hr_branch_conv_hr2x_0.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_0.layers[1].bnorm.running_var = model_hr_branch_conv_hr2x_0.layers[1].bnorm.running_var[idx1.tolist()].clone()

        # 1
new_model_hr_branch_conv_hr2x_1 = new_model.module.hr_branch.conv_hr2x[1]
model_hr_branch_conv_hr2x_1 = model.module.hr_branch.conv_hr2x[1]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.0"].cpu().numpy())))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.1"].cpu().numpy())))
w1 = model_hr_branch_conv_hr2x_1.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_hr2x_1.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr2x_1.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_1.layers[0].bias.data = b1.clone()
        # bnorm
new_model_hr_branch_conv_hr2x_1.layers[1].bnorm.weight.data = model_hr_branch_conv_hr2x_1.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_1.layers[1].bnorm.bias.data = model_hr_branch_conv_hr2x_1.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_1.layers[1].bnorm.running_mean = model_hr_branch_conv_hr2x_1.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_1.layers[1].bnorm.running_var = model_hr_branch_conv_hr2x_1.layers[1].bnorm.running_var[idx1.tolist()].clone()

        # 2
new_model_hr_branch_conv_hr2x_2 = new_model.module.hr_branch.conv_hr2x[2]
model_hr_branch_conv_hr2x_2 = model.module.hr_branch.conv_hr2x[2]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.1"].cpu().numpy())))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.2"].cpu().numpy())))
w1 = model_hr_branch_conv_hr2x_2.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_hr2x_2.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr2x_2.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_2.layers[0].bias.data = b1.clone()
        # bnorm
new_model_hr_branch_conv_hr2x_2.layers[1].bnorm.weight.data = model_hr_branch_conv_hr2x_2.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_2.layers[1].bnorm.bias.data = model_hr_branch_conv_hr2x_2.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_2.layers[1].bnorm.running_mean = model_hr_branch_conv_hr2x_2.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_2.layers[1].bnorm.running_var = model_hr_branch_conv_hr2x_2.layers[1].bnorm.running_var[idx1.tolist()].clone()

        # 3
new_model_hr_branch_conv_hr2x_3 = new_model.module.hr_branch.conv_hr2x[3]
model_hr_branch_conv_hr2x_3 = model.module.hr_branch.conv_hr2x[3]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.2"].cpu().numpy())))
w1 = model_hr_branch_conv_hr2x_3.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
new_model_hr_branch_conv_hr2x_3.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr2x_3.layers[0].bias.data
new_model_hr_branch_conv_hr2x_3.layers[0].bias.data = b1.clone()
        # bnorm
new_model_hr_branch_conv_hr2x_3.layers[1].bnorm.weight.data = model_hr_branch_conv_hr2x_3.layers[1].bnorm.weight.data.clone()
new_model_hr_branch_conv_hr2x_3.layers[1].bnorm.bias.data = model_hr_branch_conv_hr2x_3.layers[1].bnorm.bias.data.clone()
new_model_hr_branch_conv_hr2x_3.layers[1].bnorm.running_mean = model_hr_branch_conv_hr2x_3.layers[1].bnorm.running_mean.clone()
new_model_hr_branch_conv_hr2x_3.layers[1].bnorm.running_var = model_hr_branch_conv_hr2x_3.layers[1].bnorm.running_var.clone()
    
    # con_hr
        # 0
new_model_hr_branch_conv_hr_0 = new_model.module.hr_branch.conv_hr[0]
model_hr_branch_conv_hr_0 = model.module.hr_branch.conv_hr[0]
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr.0"].cpu().numpy())))
w1 = model_hr_branch_conv_hr_0.layers[0].weight.data
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_hr_branch_conv_hr_0.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr_0.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr_0.layers[0].bias.data = b1.clone()
        # bnorm
new_model_hr_branch_conv_hr_0.layers[1].bnorm.weight.data = model_hr_branch_conv_hr_0.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr_0.layers[1].bnorm.bias.data = model_hr_branch_conv_hr_0.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr_0.layers[1].bnorm.running_mean = model_hr_branch_conv_hr_0.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_hr_0.layers[1].bnorm.running_var = model_hr_branch_conv_hr_0.layers[1].bnorm.running_var[idx1.tolist()].clone()
        # 1
new_model_hr_branch_conv_hr_1 = new_model.module.hr_branch.conv_hr[1]
model_hr_branch_conv_hr_1 = model.module.hr_branch.conv_hr[1]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr.0"].cpu().numpy())))
w1 = model_hr_branch_conv_hr_1.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
new_model_hr_branch_conv_hr_1.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr_1.layers[0].bias.data
new_model_hr_branch_conv_hr_1.layers[0].bias.data = b1.clone()

# f_branch
    # conv_lr14x
new_model_f_branch_conv_lr4x = new_model.module.f_branch.conv_lr4x
model_f_branch_conv_lr4x = model.module.f_branch.conv_lr4x

idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr4x"].cpu().numpy())))
w1 = model_f_branch_conv_lr4x.layers[0].weight.data
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_f_branch_conv_lr4x.layers[0].weight.data = w1.clone()
b1 = model_f_branch_conv_lr4x.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_f_branch_conv_lr4x.layers[0].bias.data = b1.clone()
    # bnorm
new_model_f_branch_conv_lr4x.layers[1].bnorm.weight.data = model_f_branch_conv_lr4x.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_f_branch_conv_lr4x.layers[1].bnorm.bias.data = model_f_branch_conv_lr4x.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_f_branch_conv_lr4x.layers[1].bnorm.running_mean = model_f_branch_conv_lr4x.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_f_branch_conv_lr4x.layers[1].bnorm.running_var = model_f_branch_conv_lr4x.layers[1].bnorm.running_var[idx1.tolist()].clone()

    # conv_f2x
new_model_f_branch_conv_f2x = new_model.module.f_branch.conv_f2x
model_f_branch_conv_f2x = model.module.f_branch.conv_f2x
idx0 = np.squeeze(np.argwhere(np.hstack((np.asarray(cfg_mask1["conv_lr4x"].cpu().numpy()),np.asarray(np.ones(32))))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_f2x"].cpu().numpy())))
w1 = model_f_branch_conv_f2x.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_f_branch_conv_f2x.layers[0].weight.data = w1.clone()
b1 = model_f_branch_conv_f2x.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_f_branch_conv_f2x.layers[0].bias.data = b1.clone()
    # bnorm
new_model_f_branch_conv_f2x.layers[1].bnorm.weight.data = model_f_branch_conv_f2x.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_f_branch_conv_f2x.layers[1].bnorm.bias.data = model_f_branch_conv_f2x.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_f_branch_conv_f2x.layers[1].bnorm.running_mean = model_f_branch_conv_f2x.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_f_branch_conv_f2x.layers[1].bnorm.running_var = model_f_branch_conv_f2x.layers[1].bnorm.running_var[idx1.tolist()].clone()

    # conv_f
        # 0
new_model_f_branch_conv_f_0 = new_model.module.f_branch.conv_f[0]
model_f_branch_conv_f_0 = model.module.f_branch.conv_f[0]
idx0 = np.squeeze(np.argwhere(np.hstack((np.asarray(cfg_mask1["conv_f2x"].cpu().numpy()),np.asarray(np.ones(3))))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_f.0"].cpu().numpy())))
w1 = model_f_branch_conv_f_0.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
w1 = w1[idx1.tolist(),:,:,:].clone()
new_model_f_branch_conv_f_0.layers[0].weight.data = w1.clone()
b1 = model_f_branch_conv_f_0.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_f_branch_conv_f_0.layers[0].bias.data = b1.clone()
        # bnorm
new_model_f_branch_conv_f_0.layers[1].bnorm.weight.data = model_f_branch_conv_f_0.layers[1].bnorm.weight.data[idx1.tolist()].clone()
new_model_f_branch_conv_f_0.layers[1].bnorm.bias.data = model_f_branch_conv_f_0.layers[1].bnorm.bias.data[idx1.tolist()].clone()
new_model_f_branch_conv_f_0.layers[1].bnorm.running_mean = model_f_branch_conv_f_0.layers[1].bnorm.running_mean[idx1.tolist()].clone()
new_model_f_branch_conv_f_0.layers[1].bnorm.running_var = model_f_branch_conv_f_0.layers[1].bnorm.running_var[idx1.tolist()].clone()
        # 1 
new_model_f_branch_conv_f_1 = new_model.module.f_branch.conv_f[1]
model_f_branch_conv_f_1 = model.module.f_branch.conv_f[1]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_f.0"].cpu().numpy())))
w1 = model_f_branch_conv_f_1.layers[0].weight.data
w1 = w1[:,idx0.tolist(),:,:].clone()
new_model_f_branch_conv_f_1.layers[0].weight.data = w1.clone()
b1 = model_f_branch_conv_f_1.layers[0].bias.data
new_model_f_branch_conv_f_1.layers[0].bias.data = b1.clone()
torch.save(new_model.state_dict(),pruned_model_save)
# for k, v in new_model.state_dict().items():
#     print(k)
torch.save(new_model.module.backbone.model.state_dict(),pruned_model_save_v2)
# torch.save(new_model.)



print(cfg)
print(cfg1)
        



# # 将模型中所有的bn值放到一个数组中，计算绝对值，然后进行排序，通过设置的剪枝比例确定阈值
# total = 0
# for m in model.modules():
#     if isinstance(m, nn.BatchNorm2d):
#         total += m.weight.data.shape[0]
# bn = torch.zeros(total)
# index = 0
# for m in model.modules():
#     if isinstance(m, nn.BatchNorm2d):
#         size = m.weight.data.shape[0]
#         bn[index:(index+size)] = m.weight.data.abs().clone()
#         index += size
# y, i = torch.sort(bn)
# thre_index = int(total * 0.5)
# thre = y[thre_index]

# pruned = 0
# cfg = []
# cfg_mask = []
# for k, m in enumerate(model.modules()):
#     if isinstance(m, nn.BatchNorm2d):
#         weight_copy = m.weight.data.abs().clone()
#         # 根据获得的阈值确定mask，即0，1组成的数组
#         mask = weight_copy.gt(thre).float().cuda()
#         # 根据mask计算bn层的通道即输出通道，然后将mask相加，即计算1的个数，做差之后即为0的个数，就是要剪枝的个数 
#         pruned = pruned + mask.shape[0] - torch.sum(mask)
#         # 将bn层的权重和mask相乘，即mask为0的权重相乘后就是0
#         m.weight.data.mul_(mask)
#         m.bias.data.mul_(mask)
#         # cfg保存剪枝后的权重，bn层的权重
#         cfg.append(int(torch.sum(mask)))
#         # cfg_mask保存mask
#         cfg_mask.append(mask.clone())
#         print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
#             format(k, mask.shape[0], int(torch.sum(mask))))
# pruned_ratio = pruned/total
# print(pruned_ratio)