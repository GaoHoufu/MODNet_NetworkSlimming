import sys,os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.append(BASE_DIR)


import torch
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import transforms
import cv2 as cv
from visdom import Visdom
from scipy.ndimage import morphology
import numpy as np
import logging
from src.models.modnet import MODNet
from  src.trainer1 import supervised_training_iter
import random
from PIL import Image
viz = Visdom()
logging.basicConfig(filename='./my_log/my_train.log', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logging.info('------------------------------------')
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    
])

label_transform = transforms.Compose([
    transforms.ToTensor()
])

class my_dataset(Dataset):
    def __init__(self,input_path,label_path,img_transform=None,label_transform=None,w=512,h=512):
        self.input_path = input_path
        self.label_path = label_path
        self.img_path = sorted([os.path.join(self.input_path,img) for img in os.listdir(self.input_path)])
        self.alpha_path = sorted([os.path.join(self.label_path,label) for label in os.listdir(self.label_path)])
        self.w = w
        self.h = h
        self.img_transform = img_transform
        self.label_transform = label_transform
        print(len(self.img_path))
        assert len(self.img_path)==len(self.alpha_path) ,'the length is different'
    def __len__(self):
        return len(self.img_path)
    def __getitem__(self, index):
        img = cv.imread(self.img_path[index])
        label = cv.imread(self.alpha_path[index])
        img = cv.resize(img,(self.w,self.h))
        label = cv.resize(label,(self.w,self.h))


        gai = random.random()
        if gai >0.5:
            img = cv.flip(img,1)
            label = cv.flip(label,1)

        gai1 = random.random()
        if gai1 > 0.5:
            img = cv.GaussianBlur(img, (5, 5), 1.5)

        trimap = self.getTrimap(label)

        # img = Image.fromarray(cv.cvtColor(img,cv.COLOR_BGR2RGB)).convert('RGB')  

        if self.img_transform:
            img = self.img_transform(img)
        if self.label_transform:
            label = self.label_transform(label[:,:,0])
        return img,trimap,label
    def getTrimap(self,alpha):
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32))  # unknown = alpha > 0
        unknown = unknown - fg
        unknown = morphology.distance_transform_edt(unknown == 0) <= np.random.randint(10, 20)
        trimap = fg
        trimap[unknown] = 0.5
        return trimap[:, :, :1]
# cfg = None
# cfg1 = None
# 14.26M
cfg = [30, 30, 16, 94, 94, 24, 128, 128, 24, 131, 131, 32, 159, 159, 32, 149, 149, 32, 142, 142, 64, 312, 312, 64, 265, 265, 64, 258, 258, 64, 299, 299, 96, 392, 392, 96, 394, 394, 96, 397, 397, 160, 668, 668, 160, 642, 642, 160, 683, 683, 320, 1280]
cfg1 = [48, 16, 16, 16, 16, 32, 32, 32, 16, 32, 16, 16, 16, 16, 15, 16, 8]
# cfg = [32, 32, 16, 94, 94, 24, 133, 133, 24, 132, 132, 32, 158, 158, 32, 148, 148, 32, 160, 160, 64, 289, 289, 64, 270, 270, 64, 235, 235, 64, 290, 290, 96, 375, 375, 96, 365, 365, 96, 391, 391, 160, 647, 647, 160, 615, 615, 160, 660, 660, 320, 1280]
# cfg1 = [45, 16, 16, 16, 32, 32, 32, 16, 32, 16, 16, 16, 16, 13, 8]

# cfg = [32, 32, 16, 82, 82, 24, 113, 113, 24, 109, 109, 32, 102, 102, 32, 74, 74, 32, 108, 108, 64, 165, 165, 64, 94, 94, 64, 71, 71, 64, 145, 145, 96, 108, 108, 96, 104, 104, 96, 135, 135, 160, 195, 195, 160, 158, 158, 160, 208, 208, 320, 1280]
# cfg1 = [22, 5, 3, 8, 15, 16, 16, 8, 15, 6, 7, 8, 8, 6, 4]

# cfg = [32, 32, 16, 88, 88, 24, 121, 121, 24, 118, 118, 32, 126, 126, 32, 99, 99, 32, 127, 127, 64, 205, 205, 64, 146, 146, 64, 130, 130, 64, 190, 190, 96, 196, 196, 96, 202, 202, 96, 216, 216, 160, 350, 350, 160, 316, 316, 160, 357, 357, 320, 1280]
# cfg1 = [22, 5, 3, 8, 15, 16, 16, 8, 15, 6, 7, 8, 8, 6, 4]
def train(img_path,label_path,resume = False,std=0):
    # model_save = "./model_save/original_model/" # the pth of original model
    # model_save = "./model_save/model_ibnorm_bn_max/"
    # model_save = "./model_save/model_25M_1pruning_train/" #the pth of the first trained model to prue
    # model_save = "./model_save/model_15M_1pruning_finetune/"# the pth of the first pruning model with finetune with pretrained
    # model_save = "./model_save/model_15M_2pruning_train/"
    # model_save = "./model_save/model_10M_2_after_soc_pruning_finetune/"
    model_save = "./model_save2/model_new_8M"
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    if resume:
        model_name = sorted(os.listdir(model_save))[-1]
        pretrained_model = os.path.join(model_save,model_name)
    else:
        # pretrained_model = '/home/gaojun/data1/study/bisai/MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
        # pretrained_model = "./src/convert_new/1pruning_19_0.01172_0.3_0.5.pth"
        # pretrained_model = "./model_save/model_ibnorm_bn_max/new_trimap_0015_lr0.001.pth"
        # pretrained_model = "./model_save/model_15M_1pruning_finetune/new_trimap_0019_lr0.001.pth"
        # pretrained_model = "./model_save/soc_10M_train/soc_0003.pth"
        pretrained_model = "./src/convert_new/2pruning_22_0.0123_0.2_0.5.pth"
    logging.info(f'model load {pretrained_model}')

    # modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=True))
    modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=True,cfg=cfg,cfg1=cfg1))
    # modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False,cfg=cfg,cfg1=cfg1))

    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print("Use GPU...")
        # modnet.cuda().half()
        modnet.cuda()
        # modnet.load_state_dict(torch.load(pretrained_model))
    else:
        print('Use CPU...')
        # modnet.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu'))) 

    # print(modnet.state_dict())
    bs = 16
    lr = 0.01
    epochs = 40
    scaler = torch.cuda.amp.GradScaler()
    logging.info(f'batch_size: {bs},lr :{lr}, epochs: {epochs}')
    optimizer = torch.optim.SGD(modnet.parameters(), lr=lr, momentum=0.9)
    # optimizer=torch.optim.Adam(modnet.parameters(),lr=lr,betas=(0.9,0.99))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * epochs), gamma=0.1)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    dataset_train=my_dataset(img_path,label_path,img_transform,label_transform)
    trainloader=DataLoader(dataset_train,batch_size=bs,num_workers=16,pin_memory=True,shuffle=True)
    viz.line([0.], [0], win='semantic_loss', opts=dict(title='semantic_loss',legend=['semantic_loss']))
    viz.line([0.], [0], win='detail_loss', opts=dict(title='detail_loss',legend=['detail_loss']))
    viz.line([0.], [0], win='matte_loss', opts=dict(title='matte_loss',legend=['matte_loss']))
    for epoch in range(std, epochs):
        semantic_loss1 = []
        detail_loss1=[]
        matte_loss1 = []
        for idx, (image, trimap, gt_matte) in enumerate(trainloader):
            trimap = np.transpose(trimap, (0, 3, 1, 2)).float().cuda()
            semantic_loss, detail_loss, matte_loss = \
                            supervised_training_iter(modnet, optimizer, image.cuda(), trimap, gt_matte.cuda().half(),idx,scaler = scaler)  
                # supervised_training_iter(modnet, optimizer, image.cuda().half(), trimap.half(), gt_matte.cuda().half(),idx)
                # supervised_training_iter(modnet, optimizer, image.cuda(), trimap, gt_matte.cuda(),idx)                
                # supervised_training_iter(modnet, optimizer, image.half(), trimap.half(), gt_matte.half())
            
            semantic_loss1.append(float(semantic_loss))
            detail_loss1.append(float(detail_loss))
            matte_loss1.append(float(matte_loss))
        # viz.line([semantic_loss.cpu().detach().numpy()], [epoch], win='semantic_loss', update='append')
        # viz.line([detail_loss.cpu().detach().numpy()], [epoch], win='detail_loss', update='append')
        # viz.line([matte_loss.cpu().detach().numpy()], [epoch], win='matte_loss', update='append')
        avg_semantic=float(np.mean(semantic_loss1))
        avg_detail = float(np.mean(detail_loss1))
        avg_matte = float(np.mean(matte_loss1))
        info = f"epoch: {epoch+1}/{epochs} semantic_loss: {avg_semantic}, detail_loss: {avg_detail}, matte_loss: {avg_matte}"
        print(idx,info,optimizer.param_groups[0]['lr'])
        logging.info(f"epoch: {epoch+1}/{epochs}, matte_loss: {avg_semantic}")
        logging.info(f"epoch: {epoch+1}/{epochs}, matte_loss: {avg_detail}")
        logging.info(f"epoch: {epoch+1}/{epochs}, matte_loss: {avg_matte}")
        viz.line([avg_semantic], [epoch], win='semantic_loss', update='append')
        viz.line([avg_detail], [epoch], win='detail_loss', update='append')
        viz.line([avg_matte], [epoch], win='matte_loss', update='append')
        lr_scheduler.step()
        torch.save(modnet.state_dict(), os.path.join(model_save, 'new_trimap_{:0>4d}_lr{}.pth'.format(epoch+1,optimizer.param_groups[0]['lr'])))
        print(f'----------{epoch+1}--------------save model over-----------------------------------')
        logging.info(f'------save model------{epoch+1}  {epoch+1}.pth')

# img_root = '/home/gaojun/data1/study/video/mynewDataset/my_new_dataset/train/'
# label_root = '/home/gaojun/data1/study/video/mynewDataset/my_new_dataset/label/'
# img_root = '/home/gaojun/data1/study/mydataset/fa2/image/'
# label_root = '/home/gaojun/data1/study/mydataset/fa2/alpha/'

img_root = "/home/gaojun/data1/study/video/new1_again_PPM/image/"
label_root = "/home/gaojun/data1/study/video/new1_again_PPM/alpha/"
train(img_path=img_root,label_path=label_root)