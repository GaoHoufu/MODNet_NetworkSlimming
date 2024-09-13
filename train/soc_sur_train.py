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
from  src.trainer import supervised_training_iter
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
cfg = [32, 32, 16, 93, 93, 24, 131, 131, 24, 128, 128, 32, 138, 138, 32, 134, 134, 32, 138, 138, 64, 243, 243,
        64, 172, 172, 64, 117, 117, 64, 214, 214, 96, 181, 181, 96, 167, 167, 96, 237, 237, 160, 308, 308, 160,
        284, 284, 160, 556, 556, 320, 1280]
cfg1 = [30, 15, 5, 2, 15, 13, 31, 32, 16, 19, 11, 10, 11, 16, 15, 11, 8]
def train(img_path,label_path,resume = False,std=0):
    # model_save = "./model_save/original_model/" # the pth of original model
    # model_save = "./model_save/model_ibnorm_bn_max/"
    # model_save = "./model_save/model_25M_1pruning_train/" #the pth of the first trained model to prue
    # model_save = "./model_save/model_15M_1pruning_finetune/"# the pth of the first pruning model with finetune with pretrained
    # model_save = "./model_save/model_15M_2pruning_train/"
    model_save = "./model_save_soc/model_11M_0.0135/"
    if not os.path.exists(model_save):
        os.makedirs(model_save)
    if resume:
        model_name = sorted(os.listdir(model_save))[-1]
        pretrained_model = os.path.join(model_save,model_name)
    else:
        # pretrained_model = '/home/gaojun/data1/study/bisai/MODNet/pretrained/modnet_webcam_portrait_matting.ckpt'
        # pretrained_model = "./src/convert_new/1pruning_19_0.01172_0.3_0.5.pth"
        # pretrained_model = "./model_save/model_ibnorm_bn_max/new_trimap_0015_lr0.001.pth"
        pretrained_model = "./model_save/original_model_YOLO_4w_prue_MBV2_finetinue/new_trimap_0018_lr0.001.pth"
    logging.info(f'model load {pretrained_model}')

    # modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=True))
    # modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=True,cfg=cfg,cfg1=cfg1))
    modnet = torch.nn.DataParallel(MODNet(backbone_pretrained=False,cfg=cfg,cfg1=cfg1))

    GPU = True if torch.cuda.device_count() > 0 else False
    if GPU:
        print("Use GPU...")
        # modnet.cuda().half()
        modnet.cuda()
        modnet.load_state_dict(torch.load(pretrained_model))
    else:
        print('Use CPU...')
        # modnet.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu'))) 

    # print(modnet.state_dict())
    bs = 16
    lr = 0.01
    epochs = 40
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
                            supervised_training_iter(modnet, optimizer, image.cuda(), trimap, gt_matte.cuda(),idx)  
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