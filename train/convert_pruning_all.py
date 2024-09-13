import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.models.modnet import MODNet
import torch
import torch.nn as nn
import numpy as np
from src.models.backbones.mobilenetv2 import InvertedResidual, MobileNetV2
from src.models.modnet import LRBranch, HRBranch, FusionBranch
import matplotlib.pyplot as plt

cfg = None
cfg1 = None

# first pruning and the best accuracy of the model
ratio_mobilenetv2 = 0.5
ratio_modnet_other = 0.5
pruned_model_save = "./src/convert_new/first_pruning_27_0.0134_{}_{}.pth".format(ratio_mobilenetv2,ratio_modnet_other)
pruned_model_save_v2 = "./src/convert_new_v2/first_pruning_27_0.0134_v2_{}.pth".format(ratio_mobilenetv2)
# model_path = "./model_save/original_model_prue/new_trimap_0011_lr0.001.pth"
# model_path = "./model_save/original_model_prue(finetinue)1/new_trimap_0008_lr0.01.pth"
model_path = "./model_save/original_model_YOLO_3w_prue_MBV2/new_trimap_0027_lr0.0001.pth"
weight = torch.load(model_path)
model = torch.nn.DataParallel(MODNet(backbone_pretrained=False, cfg=cfg, cfg1=cfg1))
model.cuda()
model.load_state_dict(weight)


# ���mobilenetv2���ֵĴ�������Ҫ�ǶԲв�ģ�飬��Ϊmobilenetv2�Ǽ����˲в�ģ�飬���Զ�v2ÿ��ģ�����һ�㲻���д���?
# ---------------MobileNetV2��ģ�ͼ�֦----------------
# ��������bn����Ҫ��total�����Լ�mobilenetv2ģ������е�bn������ֵ
total = 0
highest_thre = []
for m in model.modules():
    if isinstance(m, MobileNetV2):
        total += m.features[0][1].weight.data.shape[0]
        highest_thre.append(m.features[0][1].weight.data.abs().max().item())
        for m1 in m.modules():
            if isinstance(m1, InvertedResidual):
                # ��Բв��һ��
                if len(m1.conv) == 5:
                    total += m1.conv[1].weight.data.shape[0]
                    highest_thre.append(m1.conv[1].weight.data.abs().max().item())
                    total += m1.conv[4].weight.data.shape[0]
                    highest_thre.append(m1.conv[4].weight.data.abs().max().item())
                if len(m1.conv) > 5:
                    total += m1.conv[1].weight.data.shape[0]
                    highest_thre.append(m1.conv[1].weight.data.abs().max().item())
                    total += m1.conv[4].weight.data.shape[0]
                    highest_thre.append(m1.conv[4].weight.data.abs().max().item())
                    total += m1.conv[7].weight.data.shape[0]
                    highest_thre.append(m1.conv[7].weight.data.abs().max().item())
        total += m.features[18][1].weight.data.shape[0]
        highest_thre.append(m.features[18][1].weight.data.abs().max().item())
bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, MobileNetV2):
        size = m.features[0][1].weight.data.shape[0]
        bn[index:(index + size)] = m.features[0][1].weight.data.abs().clone()
        index += size
        for m1 in m.modules():
            if isinstance(m1, InvertedResidual):
                for i in range(len(m1.conv)):
                    if isinstance(m1.conv[i], nn.BatchNorm2d):
                        size = m1.conv[i].weight.data.shape[0]
                        bn[index:(index + size)] = m1.conv[i].weight.data.abs().clone()
                        index += size

                # if len(m1.conv) > 5:
                #     for i in range(len(m1.conv) - 1):
                #         if isinstance(m1.conv[i], nn.BatchNorm2d):
                #             size = m1.conv[i].weight.data.shape[0]
                #             bn[index:(index + size)] = m1.conv[i].weight.data.abs().clone()
                #             index += size
# ����mobilenetv2����ѡȡ��ֵ
y, i = torch.sort(bn)
# plt.figure("v2")
# plt.plot(list(range(len(y))), y.detach().numpy())
thre_index = int(total * ratio_mobilenetv2)
thre = y[thre_index]
highest_thre = min(highest_thre)
if thre > highest_thre:
    thre = highest_thre - 1e-5
    print("The thre of mobilenetv2 use highest_thre")
# plt.hlines(thre, 0, len(y), color='g')

# mobilenetv2���ֽ���mask��ȷ��
pruned = 0
cfg = []
cfg_mask = []
idx = 0
can_cha = [3,5,6,8,9,10,12,13,15,16]
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
        count = 0
        can_cha_last = [weight_copy]
        for m1 in m.modules():
            if isinstance(m1, InvertedResidual):
                count+=1
                if len(m1.conv) > 5:
                    weight_copy = m1.conv[1].weight.data.clone()
                    mask = weight_copy.abs().gt(thre).float().cuda()
                    # pruned = pruned + mask.shape[0] - torch.sum(mask)
                    pruned = pruned + (mask.shape[0] - torch.sum(mask)) * 2
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
                    
                    if count in can_cha:
                        weight_copy = m1.conv[7].weight.data.clone()
                        mask = weight_copy.abs().gt(thre).float().cuda()
                        mask_x = cfg_mask[-3]
                        if int(torch.sum(mask)) > int(torch.sum(mask_x)):
                            # 尽可能的保留特征
                            # sum_da = int(torch.sum(mask))-int(torch.sum(mask_x))
                            # s0_index = np.where(mask_x.detach().cpu()==0)
                            # mask_x[s0_index[np.random.choice(len(s0_index[0]),sum_da,replace=False)]] = 1
                            # cfg_mask[-3] = mask_x
                            # cfg[-3] = int(torch.sum(mask_x))
                            # cfg.append(int(torch.sum(mask)))
                            # cfg_mask.append(mask.clone())
                            # # 2
                            # choose_first = int(torch.sum(mask))
                            # data_x, idx_x = torch.sort(can_cha_last[-1].abs())
                            # mask_x[idx_x[:-choose_first]] = 0
                            # cfg_mask[-3] = mask_x
                            # cfg[-3] = int(torch.sum(mask_x))
                            # cfg.append(int(torch.sum(mask)))
                            # cfg_mask.append(mask.clone())


                            # 尽可能少的保留特征层
                            # cfg.append(int(torch.sum(mask_x)))
                            # cfg_mask.append(mask_x.clone())
                            
                            choose_first = int(torch.sum(mask_x))
                            data_mask, idx_mask = torch.sort(weight_copy.abs())
                            mask[idx_mask[:-choose_first]] = 0
                            cfg.append(int(torch.sum(mask)))
                            cfg_mask.append(mask.clone())

                            
                        elif int(torch.sum(mask)) == int(torch.sum(mask_x)):
                            cfg.append(int(torch.sum(mask)))
                            cfg_mask.append(mask.clone())
                        else:
                            # sum_da = int(torch.sum(mask_x)) - int(torch.sum(mask))
                            # s0_index = np.where(mask.detach().cpu() == 0)[0]
                            # mask[s0_index[np.random.choice(len(s0_index), sum_da, replace=False)]] = 1
                            # cfg.append(int(torch.sum(mask)))
                            # cfg_mask.append(mask.clone())
                            # # 2
                            # choose_first = int(torch.sum(mask_x))
                            # data_mask, idx_mask = torch.sort(weight_copy.abs())
                            # mask_mask[idx_mask[:-choose_first]] = 0
                            # cfg.append(int(torch.sum(mask)))
                            # cfg_mask.append(mask.clone())

                            # 尽可能少的保留特征层
                            # cfg.append(int(torch.sum(mask)))
                            # cfg_mask.append(mask.clone())
                            choose_first = int(torch.sum(mask))
                            if count ==10:
                                data_x_x_x, idx_x_x_x = torch.sort(can_cha_last[-3].abs())
                                cfg_mask[-9][idx_x_x_x[:-choose_first]] = 0
                                cfg[-9] = int(torch.sum(cfg_mask[-9]))

                                data_x_x, idx_x_x = torch.sort(can_cha_last[-2].abs())
                                cfg_mask[-6][idx_x_x[:-choose_first]] = 0
                                cfg[-6] = int(torch.sum(cfg_mask[-6]))
                            if count in [6,9,12,16]:
                                data_x_x, idx_x_x = torch.sort(can_cha_last[-2].abs())
                                cfg_mask[-6][idx_x_x[:-choose_first]] = 0
                                cfg[-6] = int(torch.sum(cfg_mask[-6]))
                            data_x, idx_x = torch.sort(can_cha_last[-1].abs())
                            mask_x[idx_x[:-choose_first]] = 0
                            cfg_mask[-3] = mask_x
                            cfg[-3] = int(torch.sum(cfg_mask[-3]))
                            cfg.append(int(torch.sum(mask)))
                            cfg_mask.append(mask.clone())
                    else:
                        weight_copy = m1.conv[7].weight.data.clone()
                        mask = weight_copy.abs().gt(thre).float().cuda()
                        cfg.append(int(torch.sum(mask)))
                        cfg_mask.append(mask.clone())
                    # can_cha_last[-1] = m1.conv[7].weight.data.clone()
                    can_cha_last.append(m1.conv[7].weight.data.clone())
                    # cfg.append(m1.conv[7].weight.data.shape[0])
                    # cfg_mask.append(torch.ones(m1.conv[7].weight.data.shape[0]))

                    # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(idx, mask.shape[0], int(torch.sum(mask))))
                    idx += 1
                else:
                    idx += 1
                    # cfg.append(m1.conv[1].weight.data.shape[0])
                    # cfg_mask.append(torch.ones(m1.conv[1].weight.data.shape[0]))
                    weight_copy = m1.conv[4].weight.data.clone()
                    mask = weight_copy.abs().gt(thre).float().cuda()
                    m1.conv[4].weight.data.mul_(mask)
                    m1.conv[4].bias.data.mul_(mask)
                    cfg.append(int(torch.sum(mask)))
                    cfg_mask.append(mask.clone())
                    # cfg.append(m1.conv[4].weight.data.shape[0])
                    # cfg_mask.append(torch.ones(m1.conv[4].weight.data.shape[0]))
                    # cfg.append()
        weight_copy = m.features[18][1].weight.data.clone()
        mask = weight_copy.abs().gt(thre).float().cuda()
        m.features[18][1].weight.data.mul_(mask)
        m.features[18][1].bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        # cfg.append(m.features[18][1].weight.data.shape[0])
        # cfg_mask.append(torch.ones(m.features[18][1].weight.data.shape[0]))


# -------------MODNET���ಿ�ֵļ�֦---------------------
# modnet���ಿ����Ҫ�����Ĳ����Ȩ�ؽ��м���?
total1 = 0
highest_thre1 = []
for m in model.modules():
    if isinstance(m, LRBranch):
        # conv_lr16x
        total1 += m.conv_lr16x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_lr16x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_lr8x
        total1 += m.conv_lr8x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_lr8x.layers[1].bnorm.weight.data.abs())[-2].item())
    elif isinstance(m, HRBranch):
        # tohr_enc2x
        total1 += m.tohr_enc2x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.tohr_enc2x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_enc2x
        total1 += m.conv_enc2x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_enc2x.layers[1].bnorm.weight.data.abs())[-2].item())
        # tohr_enc4x
        total1 += m.tohr_enc4x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.tohr_enc4x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_enc4x
        total1 += m.conv_enc4x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_enc4x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_hr4x
        total1 += m.conv_hr4x[0].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr4x[0].layers[1].bnorm.weight.data.abs())[-2].item())
        total1 += m.conv_hr4x[1].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr4x[1].layers[1].bnorm.weight.data.abs())[-2].item())
        total1 += m.conv_hr4x[2].layers[1].bnorm.weight.data.shape[0]
        # print(sorted(m.conv_hr4x[2].layers[1].bnorm.weight.data.abs())[-2])
        highest_thre1.append(sorted(m.conv_hr4x[2].layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_hr2x
        total1 += m.conv_hr2x[0].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr2x[0].layers[1].bnorm.weight.data.abs())[-2].item())
        total1 += m.conv_hr2x[1].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr2x[1].layers[1].bnorm.weight.data.abs())[-2].item())
        total1 += m.conv_hr2x[2].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr2x[2].layers[1].bnorm.weight.data.abs())[-2].item())
        total1 += m.conv_hr2x[3].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr2x[3].layers[1].bnorm.weight.data.abs())[-2].item())

        # conv_hr
        total1 += m.conv_hr[0].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_hr[0].layers[1].bnorm.weight.data.abs())[-2].item())
    elif isinstance(m, FusionBranch):
        # conv_lr4x
        total1 += m.conv_lr4x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_lr4x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_f2x
        total1 += m.conv_f2x.layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_f2x.layers[1].bnorm.weight.data.abs())[-2].item())
        # conv_f
        total1 += m.conv_f[0].layers[1].bnorm.weight.data.shape[0]
        highest_thre1.append(sorted(m.conv_f[0].layers[1].bnorm.weight.data.abs())[-2].item())

bn1 = torch.zeros(total1)
index1 = 0
for m in model.modules():
    if isinstance(m, LRBranch):
        # conv_lr16x
        size = m.conv_lr16x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_lr16x.layers[1].bnorm.weight.data.abs().clone()
        index1 += size
        # # conv_lr8x
        size = m.conv_lr8x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_lr8x.layers[1].bnorm.weight.data.abs().clone()
        index1 += size
    elif isinstance(m, HRBranch):
        # tohr_enc2x
        size = m.tohr_enc2x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.tohr_enc2x.layers[1].bnorm.weight.data.abs().clone()
        index1 += size
        # conv_enc2x
        size = m.conv_enc2x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_enc2x.layers[1].bnorm.weight.data.abs().clone()
        index1 += size
        # tohr_enc4x
        size = m.tohr_enc4x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.tohr_enc4x.layers[1].bnorm.weight.data.abs().clone()
        index1 += size
        # conv_enc4x
        size = m.conv_enc4x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_enc4x.layers[1].bnorm.weight.data.abs().clone()
        index1 += size
        # conv_hr4x
        size = m.conv_hr4x[0].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_hr4x[0].layers[1].bnorm.weight.data.abs().clone()
        index1 += size

        size = m.conv_hr4x[1].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_hr4x[1].layers[1].bnorm.weight.data.abs().clone()
        index1 += size

        size = m.conv_hr4x[2].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_hr4x[2].layers[1].bnorm.weight.data.abs().clone()
        index1 += size

        # conv_hr2x
        size = m.conv_hr2x[0].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_hr2x[0].layers[1].bnorm.weight.data.abs().clone()
        index1 += size

        size = m.conv_hr2x[1].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_hr2x[1].layers[1].bnorm.weight.data.abs().clone()
        index1 += size

        size = m.conv_hr2x[2].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_hr2x[2].layers[1].bnorm.weight.data.abs().clone()
        index1 += size

        size = m.conv_hr2x[3].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_hr2x[3].layers[1].bnorm.weight.data.abs().clone()
        index1 += size

        # conv_hr
        size = m.conv_hr[0].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_hr[0].layers[1].bnorm.weight.data.abs().clone()
        index1 += size
    elif isinstance(m, FusionBranch):
        # conv_lr4x
        size = m.conv_lr4x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_lr4x.layers[1].bnorm.weight.data.abs().clone()
        index1 += size
        # conv_f2x
        size = m.conv_f2x.layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_f2x.layers[1].bnorm.weight.data.abs().clone()
        index1 += size
        # conv_f
        size = m.conv_f[0].layers[1].bnorm.weight.data.shape[0]
        bn1[index1:(index1 + size)] = m.conv_f[0].layers[1].bnorm.weight.data.abs().clone()
# modnet���ಿ�ֽ�����ֵѡȡ
y1, i1 = torch.sort(bn1)
# plt.figure("other")
# plt.plot(list(range(len(y1))), y1.detach().numpy())

thre_index1 = int(total1 * ratio_modnet_other)
thre1 = y1[thre_index1]
highest_thre1 = min(highest_thre1)
if thre1 >= highest_thre1:
    thre1 = highest_thre1 - 1e-5
    print("The other of MODNet use highest_thre1")
# plt.hlines(thre1, 0, len(y1), color='g')
# plt.show()
# modnet���ಿ�ֽ���maskȷ��
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

        # # conv_lr8x
        weight_copy = m.conv_lr8x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        # pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_lr8x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_lr8x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_lr8x"] = mask.clone()
    elif isinstance(m, HRBranch):
        # tohr_enc2x
        weight_copy = m.tohr_enc2x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        # pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.tohr_enc2x.layers[1].bnorm.weight.data.mul_(mask)
        m.tohr_enc2x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["tohr_enc2x"] = mask.clone()
        # conv_enc2x
        weight_copy = m.conv_enc2x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        # pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_enc2x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_enc2x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_enc2x"] = mask.clone()
        # tohr_enc4x
        weight_copy = m.tohr_enc4x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.tohr_enc4x.layers[1].bnorm.weight.data.mul_(mask)
        m.tohr_enc4x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["tohr_enc4x"] = mask.clone()
        # conv_enc4x
        weight_copy = m.conv_enc4x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_enc4x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_enc4x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_enc4x"] = mask.clone()
        # conv_hr4x
        weight_copy = m.conv_hr4x[0].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr4x[0].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr4x[0].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr4x.0"] = mask.clone()

        weight_copy = m.conv_hr4x[1].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr4x[1].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr4x[1].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr4x.1"] = mask.clone()

        weight_copy = m.conv_hr4x[2].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr4x[2].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr4x[2].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr4x.2"] = mask.clone()

        # conv_hr2x
        weight_copy = m.conv_hr2x[0].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr2x[0].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr2x[0].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr2x.0"] = mask.clone()

        weight_copy = m.conv_hr2x[1].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr2x[1].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr2x[1].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr2x.1"] = mask.clone()

        weight_copy = m.conv_hr2x[2].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr2x[2].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr2x[2].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr2x.2"] = mask.clone()

        weight_copy = m.conv_hr2x[3].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr2x[3].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr2x[3].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr2x.3"] = mask.clone()

        # conv_hr
        weight_copy = m.conv_hr[0].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_hr[0].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_hr[0].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_hr.0"] = mask.clone()
    elif isinstance(m, FusionBranch):
        # conv_lr4x
        weight_copy = m.conv_lr4x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_lr4x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_lr4x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_lr4x"] = mask.clone()
        # conv_f2x
        weight_copy = m.conv_f2x.layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_f2x.layers[1].bnorm.weight.data.mul_(mask)
        m.conv_f2x.layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_f2x"] = mask.clone()
        # conv_f
        weight_copy = m.conv_f[0].layers[1].bnorm.weight.data.clone()
        mask = weight_copy.abs().gt(thre1).float().cuda()
        pruned1 = pruned1 + (mask.shape[0] - torch.sum(mask))
        m.conv_f[0].layers[1].bnorm.weight.data.mul_(mask)
        m.conv_f[0].layers[1].bnorm.bias.data.mul_(mask)
        cfg1.append(int(torch.sum(mask)))
        cfg_mask1["conv_f.0"] = mask.clone()
# print(cfg1)
# print(len(cfg))
# print(pruned1/total1)

# mobilenetv2���ֽ��м�֦
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
new_model = torch.nn.DataParallel(MODNet(backbone_pretrained=False, cfg=cfg, cfg1=cfg1))
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
        # ��һ��BN���Ӧ������maskΪstart_mask
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        if idx1.size == 1:
            idx1 = np.resize(idx1, (1,))
        # ���������м�֦
        if m0.groups == 1:
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
        else:
            w1 = m0.weight.data.clone()
        # ���������м�֦
        w1 = w1[idx1.tolist(), :, :, :].clone()
        m1.weight.data = w1.clone()
# print(new_model.module.backbone.model.features[0][0].weight.data)

# modnet���ಿ�ֽ��м�֦
new_model_lr_branch_se_block = new_model.module.lr_branch.se_block
model_lr_branch_se_block = model.module.lr_branch.se_block
m_lin = 0
for [m0, m1] in zip(model_lr_branch_se_block.modules(), new_model_lr_branch_se_block.modules()):
    if isinstance(m0, nn.Linear):
        m_lin +=1
        idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
        if idx0.size == 1:
            idx0 = np.resize(idx0, (1,))
        # ��֦
        if m_lin == 1:
            m1.weight.data = m0.weight.data[:, idx0].clone()
        else:
            m1.weight.data = m0.weight.data[idx0,:].clone()
        # w1 = m0.weight.data.clone()
        # m1.weight.data = w1.clone()
### lr_branch
# conv_lr16x
# conv2d
new_model_lr_branch_conv_lr16x = new_model.module.lr_branch.conv_lr16x
model_lr_branch_conv_lr16x = model.module.lr_branch.conv_lr16x
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr6x"].cpu().numpy())))
w1 = model_lr_branch_conv_lr16x.layers[0].weight.data.clone()
b1 = model_lr_branch_conv_lr16x.layers[0].bias.data.clone()
w1 = w1[:, idx0, :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
b1 = b1[idx1.tolist()].clone()
new_model_lr_branch_conv_lr16x.layers[0].weight.data = w1.clone()
new_model_lr_branch_conv_lr16x.layers[0].bias.data = b1.clone()
# bnorm
new_model_lr_branch_conv_lr16x.layers[1].bnorm.weight.data = model_lr_branch_conv_lr16x.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_lr_branch_conv_lr16x.layers[1].bnorm.bias.data = model_lr_branch_conv_lr16x.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_lr_branch_conv_lr16x.layers[1].bnorm.running_mean = model_lr_branch_conv_lr16x.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_lr_branch_conv_lr16x.layers[1].bnorm.running_var = model_lr_branch_conv_lr16x.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# conv_lr8x
# conv2d
new_model_lr_branch_conv_lr8x = new_model.module.lr_branch.conv_lr8x
model_lr_branch_conv_lr8x = model.module.lr_branch.conv_lr8x
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr6x"].cpu().numpy())))
w1 = model_lr_branch_conv_lr8x.layers[0].weight.data.clone()
b1 = model_lr_branch_conv_lr8x.layers[0].bias.data.clone()
w1 = w1[:, idx0.tolist(), :, :].clone()
idx2 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr8x"].cpu().numpy())))
w1 = w1[idx2.tolist(), :,: , :].clone()
b1 = b1[idx2.tolist()].clone()
new_model_lr_branch_conv_lr8x.layers[0].weight.data = w1.clone()
new_model_lr_branch_conv_lr8x.layers[0].bias.data = b1.clone()

# bnorm
new_model_lr_branch_conv_lr8x.layers[1].bnorm.weight.data = model_lr_branch_conv_lr8x.layers[
    1].bnorm.weight.data[idx2.tolist()].clone()
new_model_lr_branch_conv_lr8x.layers[1].bnorm.bias.data = model_lr_branch_conv_lr8x.layers[1].bnorm.bias.data[idx2.tolist()].clone()
new_model_lr_branch_conv_lr8x.layers[1].bnorm.running_mean = model_lr_branch_conv_lr8x.layers[
    1].bnorm.running_mean[idx2.tolist()].clone()
new_model_lr_branch_conv_lr8x.layers[1].bnorm.running_var = model_lr_branch_conv_lr8x.layers[
    1].bnorm.running_var[idx2.tolist()].clone()

# conv_lr
# conv
new_model_lr_branch_conv_lr = new_model.module.lr_branch.conv_lr
model_lr_branch_conv_lr = model.module.lr_branch.conv_lr
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr8x"].cpu().numpy())))
w1 = model_lr_branch_conv_lr.layers[0].weight.data.clone()
w1 = w1[:, idx0.tolist(), :, :].clone()
b1 = model_lr_branch_conv_lr.layers[0].bias.data.clone()
new_model_lr_branch_conv_lr.layers[0].weight.data = w1
new_model_lr_branch_conv_lr.layers[0].bias.data = b1

### hr_branch
# tohr_enc2x
# conv
new_model_hr_branch_tohr_enc2x = new_model.module.hr_branch.tohr_enc2x
model_hr_branch_tohr_enc2x = model.module.hr_branch.tohr_enc2x
idx_enc2x = np.squeeze(np.argwhere(np.asarray(cfg_mask[2].cpu().numpy())))
w1 = model_hr_branch_tohr_enc2x.layers[0].weight.data.clone()
w1 = w1[:,idx_enc2x.tolist(),:,:].clone()
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["tohr_enc2x"].cpu().numpy())))
w1 = w1[idx1.tolist(), :, :, :].clone()
b1 = model_hr_branch_tohr_enc2x.layers[0].bias.data.clone()
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc2x.layers[0].weight.data = w1.clone()
new_model_hr_branch_tohr_enc2x.layers[0].bias.data = b1.clone()

# bnorm
new_model_hr_branch_tohr_enc2x.layers[1].bnorm.weight.data = model_hr_branch_tohr_enc2x.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_tohr_enc2x.layers[1].bnorm.bias.data = model_hr_branch_tohr_enc2x.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_tohr_enc2x.layers[1].bnorm.running_mean = model_hr_branch_tohr_enc2x.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_tohr_enc2x.layers[1].bnorm.running_var = model_hr_branch_tohr_enc2x.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# conv_enc2x
new_model_hr_branch_conv_enc2x = new_model.module.hr_branch.conv_enc2x
model_hr_branch_conv_enc2x = model.module.hr_branch.conv_enc2x
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["tohr_enc2x"].cpu().numpy()))) + 3
img1 = np.squeeze(np.argwhere(np.ones(3)))
idx0 = np.hstack((img1, idx0))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_enc2x"].cpu().numpy())))
w1 = model_hr_branch_conv_enc2x.layers[0].weight.data.clone()
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_enc2x.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_enc2x.layers[0].bias.data.clone()
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_enc2x.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_enc2x.layers[1].bnorm.weight.data = model_hr_branch_conv_enc2x.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_enc2x.layers[1].bnorm.bias.data = model_hr_branch_conv_enc2x.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_enc2x.layers[1].bnorm.running_mean = model_hr_branch_conv_enc2x.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_enc2x.layers[1].bnorm.runnign_var = model_hr_branch_conv_enc2x.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# tohr_enc4x
new_model_hr_branch_tohr_enc4x = new_model.module.hr_branch.tohr_enc4x
model_hr_branch_tohr_enc4x = model.module.hr_branch.tohr_enc4x
idx_enc4x = np.squeeze(np.argwhere(np.asarray(cfg_mask[8].cpu().numpy())))
w1 = model_hr_branch_tohr_enc4x.layers[0].weight.data.clone()
w1 = w1[:,idx_enc4x.tolist(),:,:].clone()
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["tohr_enc4x"].cpu().numpy())))
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_tohr_enc4x.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_tohr_enc4x.layers[0].bias.data.clone()
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_tohr_enc4x.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_tohr_enc4x.layers[1].bnorm.weight.data = model_hr_branch_tohr_enc4x.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_tohr_enc4x.layers[1].bnorm.bias.data = model_hr_branch_tohr_enc4x.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_tohr_enc4x.layers[1].bnorm.running_mean = model_hr_branch_tohr_enc4x.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_tohr_enc4x.layers[1].bnorm.running_var = model_hr_branch_tohr_enc4x.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# conv_enc4x
new_model_hr_branch_conv_enc4x = new_model.module.hr_branch.conv_enc4x
model_hr_branch_conv_enc4x = model.module.hr_branch.conv_enc4x
idx0 = np.squeeze(np.argwhere(
    np.hstack((np.asarray(cfg_mask1["conv_enc2x"].cpu().numpy()), np.asarray(cfg_mask1["tohr_enc4x"].cpu().numpy())))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_enc4x"].cpu().numpy())))
w1 = model_hr_branch_conv_enc4x.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_enc4x.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_enc4x.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_enc4x.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_enc4x.layers[1].bnorm.weight.data = model_hr_branch_conv_enc4x.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_enc4x.layers[1].bnorm.bias.data = model_hr_branch_conv_enc4x.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_enc4x.layers[1].bnorm.running_mean = model_hr_branch_conv_enc4x.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_enc4x.layers[1].bnorm.running_var = model_hr_branch_conv_enc4x.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# conv_hr4x
# 0
new_model_hr_branch_conv_hr4x_0 = new_model.module.hr_branch.conv_hr4x[0]
model_hr_branch_conv_hr4x_0 = model.module.hr_branch.conv_hr4x[0]
hstack1 = np.hstack((np.asarray(cfg_mask1["conv_enc4x"].cpu().numpy()), np.asarray(cfg_mask1["conv_lr8x"].cpu().numpy())))
idx0 = np.squeeze(np.argwhere(np.hstack((hstack1, np.asarray(np.ones(3))))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.0"].cpu().numpy())))

w1 = model_hr_branch_conv_hr4x_0.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_hr4x_0.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr4x_0.layers[0].bias.data
b1 = b1[idx1.tolist()]
new_model_hr_branch_conv_hr4x_0.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_hr4x_0.layers[1].bnorm.weight.data = model_hr_branch_conv_hr4x_0.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_0.layers[1].bnorm.bias.data = model_hr_branch_conv_hr4x_0.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_0.layers[1].bnorm.running_mean = model_hr_branch_conv_hr4x_0.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_0.layers[1].bnorm.running_var = model_hr_branch_conv_hr4x_0.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# 1
new_model_hr_branch_conv_hr4x_1 = new_model.module.hr_branch.conv_hr4x[1]
model_hr_branch_conv_hr4x_1 = model.module.hr_branch.conv_hr4x[1]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.0"].cpu().numpy())))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.1"].cpu().numpy())))
w1 = model_hr_branch_conv_hr4x_1.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_hr4x_1.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr4x_1.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_1.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_hr4x_1.layers[1].bnorm.weight.data = model_hr_branch_conv_hr4x_1.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_1.layers[1].bnorm.bias.data = model_hr_branch_conv_hr4x_1.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_1.layers[1].bnorm.running_mean = model_hr_branch_conv_hr4x_1.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_1.layers[1].bnorm.running_var = model_hr_branch_conv_hr4x_1.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# 2
new_model_hr_branch_conv_hr4x_2 = new_model.module.hr_branch.conv_hr4x[2]
model_hr_branch_conv_hr4x_2 = model.module.hr_branch.conv_hr4x[2]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.1"].cpu().numpy())))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr4x.2"].cpu().numpy())))
w1 = model_hr_branch_conv_hr4x_2.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_hr4x_2.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr4x_2.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_2.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_hr4x_2.layers[1].bnorm.weight.data = model_hr_branch_conv_hr4x_2.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_2.layers[1].bnorm.bias.data = model_hr_branch_conv_hr4x_2.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_2.layers[1].bnorm.running_mean = model_hr_branch_conv_hr4x_2.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr4x_2.layers[1].bnorm.running_var = model_hr_branch_conv_hr4x_2.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# conv_hr2x
# 0
new_model_hr_branch_conv_hr2x_0 = new_model.module.hr_branch.conv_hr2x[0]
model_hr_branch_conv_hr2x_0 = model.module.hr_branch.conv_hr2x[0]
idx0 = np.squeeze(np.argwhere(
    np.hstack((np.asarray(cfg_mask1["conv_hr4x.2"].cpu().numpy()), np.asarray(cfg_mask1["tohr_enc2x"].cpu().numpy())))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.0"].cpu().numpy())))
w1 = model_hr_branch_conv_hr2x_0.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_hr2x_0.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr2x_0.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_0.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_hr2x_0.layers[1].bnorm.weight.data = model_hr_branch_conv_hr2x_0.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_0.layers[1].bnorm.bias.data = model_hr_branch_conv_hr2x_0.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_0.layers[1].bnorm.running_mean = model_hr_branch_conv_hr2x_0.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_0.layers[1].bnorm.running_var = model_hr_branch_conv_hr2x_0.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# 1
new_model_hr_branch_conv_hr2x_1 = new_model.module.hr_branch.conv_hr2x[1]
model_hr_branch_conv_hr2x_1 = model.module.hr_branch.conv_hr2x[1]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.0"].cpu().numpy())))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.1"].cpu().numpy())))
w1 = model_hr_branch_conv_hr2x_1.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_hr2x_1.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr2x_1.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_1.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_hr2x_1.layers[1].bnorm.weight.data = model_hr_branch_conv_hr2x_1.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_1.layers[1].bnorm.bias.data = model_hr_branch_conv_hr2x_1.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_1.layers[1].bnorm.running_mean = model_hr_branch_conv_hr2x_1.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_1.layers[1].bnorm.running_var = model_hr_branch_conv_hr2x_1.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# 2
new_model_hr_branch_conv_hr2x_2 = new_model.module.hr_branch.conv_hr2x[2]
model_hr_branch_conv_hr2x_2 = model.module.hr_branch.conv_hr2x[2]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.1"].cpu().numpy())))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.2"].cpu().numpy())))
w1 = model_hr_branch_conv_hr2x_2.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_hr2x_2.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr2x_2.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_2.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_hr2x_2.layers[1].bnorm.weight.data = model_hr_branch_conv_hr2x_2.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_2.layers[1].bnorm.bias.data = model_hr_branch_conv_hr2x_2.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_2.layers[1].bnorm.running_mean = model_hr_branch_conv_hr2x_2.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_2.layers[1].bnorm.running_var = model_hr_branch_conv_hr2x_2.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# 3
new_model_hr_branch_conv_hr2x_3 = new_model.module.hr_branch.conv_hr2x[3]
model_hr_branch_conv_hr2x_3 = model.module.hr_branch.conv_hr2x[3]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.2"].cpu().numpy())))
w1 = model_hr_branch_conv_hr2x_3.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.3"].cpu().numpy())))
w1 = w1[idx1.tolist(),:,:,:].clone()
b1 = model_hr_branch_conv_hr2x_3.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_3.layers[0].weight.data = w1.clone()
new_model_hr_branch_conv_hr2x_3.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_hr2x_3.layers[1].bnorm.weight.data = model_hr_branch_conv_hr2x_3.layers[
    1].bnorm.weight.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_3.layers[1].bnorm.bias.data = model_hr_branch_conv_hr2x_3.layers[
    1].bnorm.bias.data[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_3.layers[1].bnorm.running_mean = model_hr_branch_conv_hr2x_3.layers[
    1].bnorm.running_mean[idx1.tolist()].clone()
new_model_hr_branch_conv_hr2x_3.layers[1].bnorm.running_var = model_hr_branch_conv_hr2x_3.layers[
    1].bnorm.running_var[idx1.tolist()].clone()

# con_hr
# 0
new_model_hr_branch_conv_hr_0 = new_model.module.hr_branch.conv_hr[0]
model_hr_branch_conv_hr_0 = model.module.hr_branch.conv_hr[0]
idx_0 = np.squeeze(np.argwhere(np.hstack((cfg_mask1["conv_hr2x.3"].cpu().numpy(), np.asarray(np.ones(3))))))
# idx_0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr2x.3"].cpu().numpy())))
w1 = model_hr_branch_conv_hr_0.layers[0].weight.data
w1 = w1[:,idx_0.tolist(),:,:].clone()
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr.0"].cpu().numpy())))
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_hr_branch_conv_hr_0.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr_0.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_hr_branch_conv_hr_0.layers[0].bias.data = b1.clone()
# bnorm
new_model_hr_branch_conv_hr_0.layers[1].bnorm.weight.data = model_hr_branch_conv_hr_0.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr_0.layers[1].bnorm.bias.data = model_hr_branch_conv_hr_0.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr_0.layers[1].bnorm.running_mean = model_hr_branch_conv_hr_0.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_hr_branch_conv_hr_0.layers[1].bnorm.running_var = model_hr_branch_conv_hr_0.layers[1].bnorm.running_var[
    idx1.tolist()].clone()
# 1
new_model_hr_branch_conv_hr_1 = new_model.module.hr_branch.conv_hr[1]
model_hr_branch_conv_hr_1 = model.module.hr_branch.conv_hr[1]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_hr.0"].cpu().numpy())))
w1 = model_hr_branch_conv_hr_1.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
new_model_hr_branch_conv_hr_1.layers[0].weight.data = w1.clone()
b1 = model_hr_branch_conv_hr_1.layers[0].bias.data
new_model_hr_branch_conv_hr_1.layers[0].bias.data = b1.clone()

# f_branch
# conv_lr14x
new_model_f_branch_conv_lr4x = new_model.module.f_branch.conv_lr4x
model_f_branch_conv_lr4x = model.module.f_branch.conv_lr4x
idx_lr8x = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr8x"].cpu().numpy())))
w1 = model_f_branch_conv_lr4x.layers[0].weight.data
w1 = w1[:,idx_lr8x.tolist(),:,:]
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_lr4x"].cpu().numpy())))
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_f_branch_conv_lr4x.layers[0].weight.data = w1.clone()
b1 = model_f_branch_conv_lr4x.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_f_branch_conv_lr4x.layers[0].bias.data = b1.clone()
# bnorm
new_model_f_branch_conv_lr4x.layers[1].bnorm.weight.data = model_f_branch_conv_lr4x.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_f_branch_conv_lr4x.layers[1].bnorm.bias.data = model_f_branch_conv_lr4x.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_f_branch_conv_lr4x.layers[1].bnorm.running_mean = model_f_branch_conv_lr4x.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_f_branch_conv_lr4x.layers[1].bnorm.running_var = model_f_branch_conv_lr4x.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# conv_f2x
new_model_f_branch_conv_f2x = new_model.module.f_branch.conv_f2x
model_f_branch_conv_f2x = model.module.f_branch.conv_f2x
idx0 = np.squeeze(np.argwhere(np.hstack((np.asarray(cfg_mask1["conv_lr4x"].cpu().numpy()), np.asarray(cfg_mask1["conv_hr2x.3"].cpu().numpy())))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_f2x"].cpu().numpy())))
w1 = model_f_branch_conv_f2x.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_f_branch_conv_f2x.layers[0].weight.data = w1.clone()
b1 = model_f_branch_conv_f2x.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_f_branch_conv_f2x.layers[0].bias.data = b1.clone()
# bnorm
new_model_f_branch_conv_f2x.layers[1].bnorm.weight.data = model_f_branch_conv_f2x.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_f_branch_conv_f2x.layers[1].bnorm.bias.data = model_f_branch_conv_f2x.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_f_branch_conv_f2x.layers[1].bnorm.running_mean = model_f_branch_conv_f2x.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_f_branch_conv_f2x.layers[1].bnorm.running_var = model_f_branch_conv_f2x.layers[1].bnorm.running_var[
    idx1.tolist()].clone()

# conv_f
# 0
new_model_f_branch_conv_f_0 = new_model.module.f_branch.conv_f[0]
model_f_branch_conv_f_0 = model.module.f_branch.conv_f[0]
idx0 = np.squeeze(np.argwhere(np.hstack((np.asarray(cfg_mask1["conv_f2x"].cpu().numpy()), np.asarray(np.ones(3))))))
idx1 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_f.0"].cpu().numpy())))
w1 = model_f_branch_conv_f_0.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
w1 = w1[idx1.tolist(), :, :, :].clone()
new_model_f_branch_conv_f_0.layers[0].weight.data = w1.clone()
b1 = model_f_branch_conv_f_0.layers[0].bias.data
b1 = b1[idx1.tolist()].clone()
new_model_f_branch_conv_f_0.layers[0].bias.data = b1.clone()
# bnorm
new_model_f_branch_conv_f_0.layers[1].bnorm.weight.data = model_f_branch_conv_f_0.layers[1].bnorm.weight.data[
    idx1.tolist()].clone()
new_model_f_branch_conv_f_0.layers[1].bnorm.bias.data = model_f_branch_conv_f_0.layers[1].bnorm.bias.data[
    idx1.tolist()].clone()
new_model_f_branch_conv_f_0.layers[1].bnorm.running_mean = model_f_branch_conv_f_0.layers[1].bnorm.running_mean[
    idx1.tolist()].clone()
new_model_f_branch_conv_f_0.layers[1].bnorm.running_var = model_f_branch_conv_f_0.layers[1].bnorm.running_var[
    idx1.tolist()].clone()
# 1
new_model_f_branch_conv_f_1 = new_model.module.f_branch.conv_f[1]
model_f_branch_conv_f_1 = model.module.f_branch.conv_f[1]
idx0 = np.squeeze(np.argwhere(np.asarray(cfg_mask1["conv_f.0"].cpu().numpy())))
w1 = model_f_branch_conv_f_1.layers[0].weight.data
w1 = w1[:, idx0.tolist(), :, :].clone()
new_model_f_branch_conv_f_1.layers[0].weight.data = w1.clone()
b1 = model_f_branch_conv_f_1.layers[0].bias.data
new_model_f_branch_conv_f_1.layers[0].bias.data = b1.clone()
torch.save(new_model.state_dict(), pruned_model_save)
# for k, v in new_model.state_dict().items():
#     print(k)
torch.save(new_model.module.backbone.model.state_dict(), pruned_model_save_v2)
# torch.save(new_model.)


print(cfg)
print(cfg1)

# # ��ģ�������е�bnֵ�ŵ�һ�������У��������ֵ��Ȼ���������ͨ�����õļ�֦����ȷ����ֵ
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
#         # ���ݻ�õ���ֵȷ��mask����0��1��ɵ�����?
#         mask = weight_copy.gt(thre).float().cuda()
#         # ����mask����bn���ͨ�������ͨ����Ȼ��mask��ӣ�������?1�ĸ���������֮��Ϊ0�ĸ���������Ҫ��֦�ĸ���
#         pruned = pruned + mask.shape[0] - torch.sum(mask)
#         # ��bn���Ȩ�غ�mask��ˣ���maskΪ0��Ȩ����˺����0
#         m.weight.data.mul_(mask)
#         m.bias.data.mul_(mask)
#         # cfg�����֦���Ȩ�أ�bn���Ȩ��?
#         cfg.append(int(torch.sum(mask)))
#         # cfg_mask����mask
#         cfg_mask.append(mask.clone())
#         print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
#             format(k, mask.shape[0], int(torch.sum(mask))))
# pruned_ratio = pruned/total
# print(pruned_ratio)