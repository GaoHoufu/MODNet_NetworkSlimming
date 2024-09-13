import numpy as np
import sys
sys.path.append("/home/gaojun/data1/study/my_experiment/MODNet-master")
from glob import glob
from src.models.modnet import MODNet
# from models.modnet_auto import MODNet_auto
from PIL import Image
from infer import predit_matte
import torch.nn as nn
import torch
import os

def cal_mad(pred, gt):
    diff = pred - gt
    diff = np.abs(diff)
    mad = np.mean(diff)
    return mad
    

def cal_mse(pred, gt):
    diff = pred - gt
    diff = diff ** 2
    mse = np.mean(diff)
    return mse


def load_eval_dataset(dataset_root_dir="./train/PPM-100"):
    image_path = dataset_root_dir + '/train/fg/*'
    matte_path = dataset_root_dir + '/train/alpha/*'
    image_file_name_list = glob(image_path)
    image_file_name_list = sorted(image_file_name_list)
    matte_file_name_list = glob(matte_path)
    matte_file_name_list = sorted(matte_file_name_list)

    return image_file_name_list, matte_file_name_list


def eval(modnet: MODNet, dataset):
    mse = total_mse = 0.0
    mad = total_mad = 0.0
    cnt = 0 
    for im_pth, mt_pth in zip(dataset[0], dataset[1]):
        im = Image.open(im_pth)
        pd_matte = predit_matte(modnet, im)

        gt_matte = Image.open(mt_pth).convert('L')
        gt_matte = np.asarray(gt_matte) / 255

        total_mse += cal_mse(pd_matte, gt_matte)
        total_mad += cal_mad(pd_matte, gt_matte)

        cnt += 1
    if cnt > 0:
        mse = total_mse / cnt
        mad = total_mad / cnt

    return mse, mad


if __name__ == '__main__':
    # create?MODNet?and?loa d?the?pre-trained?ckpt
    # cfg = None
    # cfg1 =None
    
    # first prue
    cfg = [32, 32, 16, 93, 93, 24, 131, 131, 24, 128, 128, 32, 138, 138, 32, 134, 134, 32, 138, 138, 64, 243, 243, 64, 172, 172, 64, 117, 117, 64, 214, 214, 96, 181, 181, 96, 167, 167, 96, 237, 237, 160, 308, 308, 160, 284, 284, 160, 556, 556, 320, 1280]
    cfg1 = [30, 15, 5, 2, 15, 13, 31, 32, 16, 19, 11, 10, 11, 16, 15, 11, 8]
    
    
    # first pruning
    #second prue
    # cfg = [16, 16, 16, 78, 78, 24, 98, 98, 24, 103, 103, 32, 78, 78, 32, 57, 57, 32, 95, 95, 64, 111, 111, 64, 39, 39, 64, 35, 35, 64, 87, 87, 96, 52, 52, 96, 51, 51, 96, 82, 82, 160, 32, 32, 160, 54, 54, 160, 97, 97, 320, 1280]
    # cfg1 = [30, 15, 4, 2, 14, 13, 31, 32, 16, 18, 11, 9, 10, 16, 15, 11, 8]
    # torch.set_float32_matmul_precision('high')
    # torch.backends.cuda.is_built()
    modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False,cfg=cfg,cfg1=cfg1))
    
    # modnet = MODNet(backbone_pretrained=False,cfg=cfg,cfg1=cfg1)
    # modnet = torch.compile(modnet)
    # modnet = torch.nn.DataParallel(modnet)
    
    # ckp_list1 = "./model_save/model_original(25M_no)"
    # ckp_list1 = "./model_save/model_original(25M_no_bn_in_all)"
    # ckp_list1 = "./model_save/model_original(25M_on_bn_in_all)"
    # ckp_list1 = "./model_save/model_first_after_pruning(14M_no_bn_in_all_with)"

    # ckp_list1 = "./model_save/original_model"
    # ckp_list1 = "./model_save/original_model_prue"
    # ckp_list1 = "./model_save/original_model_prue(finetinue)"
    # ckp_list1 = "./model_save/original_model_YOLO_3w_prue_MBV2"
    # ckp_list1 = "./model_save/original_model_YOLO_4w_prue_MBV2_finetinue"
    ckp_list1 = "./model_save_soc/model_11M_0.0135"
    ckp_pth1 = sorted([os.path.join(ckp_list1,i) for i in os.listdir(ckp_list1)],reverse=True)
    k = len(ckp_pth1)
    min_mse = 1
    min_pth = 0
    for ckp_pth in ckp_pth1:
        if torch.cuda.is_available():
            # modnet = modnet.cuda().half()
            modnet = modnet.cuda()
            weights = torch.load(ckp_pth)
        else:
            weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
        modnet.load_state_dict(weights)
        # torch.save(modnet.module.backbone.model.state_dict(), "./src/convert_new_v2_gai/")

        dataset = load_eval_dataset()
        mse, mad = eval(modnet, dataset)
        if mse <= min_mse:
            min_mse = mse
            min_pth = ckp_pth
        print(k," ",f'mse: {mse:6f}, mad: {mad:6f}')
        k-=1
    print("min_mse:",min_mse,"min_pth:",min_pth)