import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.models.modnet import MODNet
from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2 as cv
import os
def predit_matte(modnet: MODNet, im: Image):
    # define image to tensor transform
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # define hyper-parameters
    ref_size = 512

    modnet.eval()

    # unify image channels to 3
    im = np.asarray(im)
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    im = Image.fromarray(im)
    # convert image to PyTorch tensor
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    # _, _, matte = modnet(im.cuda().half() if torch.cuda.is_available() else im, True)
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)


    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte

def infer_one_image(model, img_path):
    img = Image.open(img_path)

    matte = predit_matte(model, img)
    h, w = matte.shape[:2]
    res = cv.resize(matte, (int(w / 5), int(h / 5)))

    prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')

    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    prd_img = cv.cvtColor(np.asarray(prd_img), cv.COLOR_RGB2BGR)
    res = cv.add(img, 255 - prd_img)
    h, w = img.shape[:2]
    res = cv.resize(res, (int(w / 4), int(h / 4)))
    # path = "./the_infer_img_with_prue/" + img_path.split('/')[-1]
    path1 = "./the_infer_img_with_original1/"
    if not os.path.exists(path1):
        os.makedirs(path1)
    path = path1 + img_path.split('/')[-1]

    cv.imwrite(path,res)
    # cv.imshow('result', res)
    # cv.waitKey(0)
    # cv.destroyAllWindows()


if __name__ == '__main__':
    # create?MODNet?and?load?the?pre-trained?ckpt
    cfg = None
    cfg1 = None
    # cfg = [15, 15, 16, 84, 84, 24, 106, 106, 24, 105, 105, 32, 102, 102, 32, 81, 81, 32, 101, 101, 64, 157, 157, 64, 73, 73, 64, 45, 45, 64, 126, 126, 96, 81, 81, 96, 63, 63, 96, 109, 109, 160, 117, 117, 160, 111, 111, 160, 211, 211, 320, 1280]
    # cfg1 = [43, 6, 2, 11, 12, 18, 16, 8, 18, 6, 7, 10, 8, 6, 8]
    
    # cfg = [32, 32, 16, 93, 93, 24, 131, 131, 24, 128, 128, 32, 138, 138, 32, 134, 134, 32, 138, 138, 64, 243, 243, 64, 172, 172, 64, 117, 117, 64, 214, 214, 96, 181, 181, 96, 167, 167, 96, 237, 237, 160, 308, 308, 160, 284, 284, 160, 556, 556, 320, 1280]
    # cfg1 = [30, 15, 5, 2, 15, 13, 31, 32, 16, 19, 11, 10, 11, 16, 15, 11, 8]
    modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False,cfg=cfg,cfg1=cfg1))

    # modnet = MODNet(backbone_pretrained=False)

    # ckp_pth = 'pretrained/modnet_photographic_portrait_matting.ckpt'
    # ckp_pth = "/home/data1/study/video/gai_model/new_trimap_0021_lr0.0001.pth"
    # ckp_pth = "/home/data1/study/video/new1_hdf5/finish1_converted.pth"
    # ckp_pth = "./the_new_convert_model4_continue_trained_finetune_with_pth_v2_0.4_other_0.45_without_acc/new_trimap_0019_lr0.001.pth"
    ckp_pth = "./model_save/original_model/new_trimap_0020_lr0.0001.pth"
    
    # ckp_pth = "./model_save/original_model_YOLO_4w_prue_MBV2_finetinue/new_trimap_0018_lr0.001.pth"
    # ckp_pth = "./model_save_soc/model_11M_0.0135/soc_0005.pth"
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckp_pth)
    else:
        weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)

    pth = './train/PPM-100/val/fg/'
    path_list = [os.path.join(pth,i) for i in os.listdir(pth)]
    # pth = './PPM-100/train/fg/49740440761_f9ffe43f60_o.jpg'
    for i in path_list:
        infer_one_image(modnet, i)
    # img = Image.open(pth)

    # matte = predit_matte(modnet, img)
    # prd_img = Image.fromarray(((matte * 255).astype('uint8')), mode='L')
    # prd_img.save('test_predic.jpg')
    


