import sys,os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
from src.models.modnet import MODNet
import os
import torch
from torch.utils import data
from PIL import Image
import cv2 as cv
import torchvision
from visdom import Visdom
viz = Visdom()
import time
import copy
import numpy as np
import logging
import random
from src.trainer import soc_adaptation_iter
logging.basicConfig(filename='soc_train.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('------------------------------------')

device = torch.device('cuda')

transforms_imag=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
])
class MyDataset(data.Dataset):#继承了Dataset子类
    def __init__(self,input_root,transform=None,w = 512,h = 512):
        #分别读取输入/标签图片的路径信息
        self.input_root=input_root
        self.input_files=os.listdir(input_root)#列出指定路径下的所有文件
        self.input_img_path = sorted([os.path.join(self.input_root,img) for img in self.input_files])
        self.transforms=transform
        self.w = w
        self.h = h
    def __len__(self):
        #获取数据集大小
        return len(self.input_files)
    def __getitem__(self, index):
        #根据索引(id)读取对应的图片
        # input_img_path=os.path.join(self.input_root,self.input_files[index])
        # input_img=Image.open(self.input_img_path[index])
        img = cv.imread(self.input_img_path[index])
        img = cv.resize(img,(self.w,self.h))
        #视频教程使用skimage来读取的图片，但我在之后使用transforms处理图片时会报错
        #所以在此我修改为使用PIL形式读取的图片
        gai = random.random()
        if gai >0.5:
            img = cv.flip(img,1)

        if self.transforms:
            #transforms方法如果有就先处理，然后再返回最后结果
            input_img=self.transforms(img)
        # print(label_img)
        # input_img=input_img.to(device)
        return input_img
cfg = [32, 32, 16, 93, 93, 24, 131, 131, 24, 128, 128, 32, 138, 138, 32, 134, 134, 32, 138, 138, 64, 243, 243,
        64, 172, 172, 64, 117, 117, 64, 214, 214, 96, 181, 181, 96, 167, 167, 96, 237, 237, 160, 308, 308, 160,
        284, 284, 160, 556, 556, 320, 1280]
cfg1 = [30, 15, 5, 2, 15, 13, 31, 32, 16, 19, 11, 10, 11, 16, 15, 11, 8]
def soc_train(input_root,std=0):
    # the_first_model = './model_save/model_10M_1pruning_finetune/new_trimap_0038_lr1e-05.pth'
    model_save = "./model_save_soc/model_11M_0.0135_new/"
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    pretrained_pth = "./model_save/original_model_YOLO_4w_prue_MBV2_finetinue/new_trimap_0018_lr0.001.pth"
    # pretrained_model_name = sorted(os.listdir(the_first_model))[-1]
    # pretrained_pth = os.path.join(the_first_model,pretrained_model_name)
    print(pretrained_pth)
    logging.info(f"load model {pretrained_pth}")
    # modnet = torch.nn.DataParallel(MODNet())
    modnet = torch.nn.DataParallel(MODNet(cfg=cfg,cfg1=cfg1))
    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print('Use GPU...')
        modnet = modnet.cuda()
        modnet.load_state_dict(torch.load(pretrained_pth))
    else:
        print('Use CPU...')
        modnet.load_state_dict(torch.load(pretrained_pth, map_location=torch.device('cpu')))
    bs = 16
    lr = 0.00001
    epochs = 10
    num_work = 2
    optimizer = torch.optim.Adam(modnet.parameters(), lr=lr, betas=(0.9, 0.99))
    dataset = MyDataset(input_root,transforms_imag)
    dataloader = data.DataLoader(dataset,batch_size = bs,num_workers=num_work, pin_memory=True)
    viz.line([0.], [0], win='soc_semantic_loss', opts=dict(title='soc_semantic_loss',legend=['soc_semantic_loss']))
    viz.line([0.], [0], win='soc_matte_loss', opts=dict(title='soc_matte_loss',legend=['soc_matte_loss']))
    for epoch in range(std, epochs):
        semantic_loss=[]
        detail_loss=[]
        backup_modnet = copy.deepcopy(modnet)
        for idx, image in enumerate(dataloader):
            image = image.cuda()
            soc_semantic_loss, soc_detail_loss = soc_adaptation_iter(modnet, backup_modnet, optimizer, image)
            info = f"epoch: {epoch+1}/{epochs} soc_semantic_loss: {soc_semantic_loss}, soc_detail_loss: {soc_detail_loss}"
            # print(idx, info)
            semantic_loss.append(float(soc_semantic_loss))
            detail_loss.append(float(soc_detail_loss))
        avg_semantic_loss = float(np.mean(semantic_loss))
        avg_detail_loss = float(np.mean(detail_loss))
        viz.line([avg_semantic_loss], [epoch], win='soc_semantic_loss', update='append')
        viz.line([avg_detail_loss], [epoch], win='soc_matte_loss', update='append')
        logging.info(f"epoch: {epoch+1}/{epochs}, avg_semantic_loss: {avg_semantic_loss}, avg_detail_loss: {avg_detail_loss}")
        torch.save(modnet.state_dict(), os.path.join(model_save, 'soc_{:0>4d}.pth'.format(epoch+1)))
        print(f'------save soc model------{epoch+1}  {epoch+1}.pth')
        logging.info(f'------save model------{epoch+1}  {epoch+1}.pth')
# img_root = '/home/gaojun/data1/study/video/mynewDataset/my_new_dataset/train/'
# img_root = '/home/gaojun/data1/study/mydataset/fa2/image/'
# img_root = "/home/gaojun/data1/study/video/new1_again_PPM/image/"
img_root = "/home/gaojun/data1/study/video/new1_again_PPM_soc_3k/image/"
soc_train(input_root=img_root)
