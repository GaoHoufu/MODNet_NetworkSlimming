import cv2
import numpy as np
import random
import os
import torch
from torchvision import transforms
from PIL import Image
def clamp(pv):                                 # 保证RGB三色数值不超过255
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv


def gaussian_noise(image):                      # 高斯噪声
    h, w, c = image.shape                       # 获取图像高度、宽度、通道
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)      # 获取随机数，3个数的数组
            b = image[row, col, 0]      # 蓝
            g = image[row, col, 1]      # 绿
            r = image[row, col, 2]      # 红

            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    return image
def add_noise_Guass(image, mean=0, var=0.01):  # 添加高斯噪声
    #设置高斯分布的均值和方差
    mean = 0
    #设置高斯分布的标准差
    sigma = 25
    #根据均值和标准差生成符合高斯分布的噪声
    gauss = np.random.normal(mean,sigma,(image.shape[0],image.shape[1],image.shape[2]))
    #给图片添加高斯噪声
    noisy_img = image + gauss
    #设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
    return noisy_img
def crop1(alpha):
    # alpha = cv2.imread(after_path,0)
    # alpha=cv2.resize(alpha,(768,768))
    alpha1=np.array(alpha)
    # print(np.argwhere(alpha1==1))
    h1_min = min(np.argwhere(alpha1==1)[:,0])
    h1_max = max(np.argwhere(alpha1==1)[:,0])
    w1_min = min(np.argwhere(alpha1==1)[:,1])
    w1_max = max(np.argwhere(alpha1==1)[:,1])
    # print(h1_min)
    alpha1_h = alpha1.shape[0]
    alpha1_w = alpha1.shape[1]
    if h1_min<=256:
        h1_start = random.randint(int(0.9*h1_min),h1_min)
        h1_end = h1_start+512
    else:
        h1_start = random.randint(240,256)
        h1_end = h1_start+512
    if w1_min<=256:
        if w1_max<512:
            w1_start = random.randint(0,w1_min)
        else:
            if w1_max-w1_min<=512:
                w1_start = random.randint(512+w1_min-w1_max,w1_min)
            else:
                w1_start = int(w1_min+(w1_max-w1_min)/2 - 256)
        w1_end = w1_start + 512

    else:
        if w1_max<512:
            w1_start = random.randint(0,256)
        else:
            w1_start = random.randint(w1_max-512,256)
        w1_end = w1_start+512
    # k = alpha[h1_start:h1_end,w1_start:w1_end]
    return h1_start,h1_end,w1_start,w1_end


def dataset_make(img_root, mask_root, background_root,j = 0):

    img = cv2.imread(img_root)
    # img = cv2.resize(img,())
    mask = cv2.imread(mask_root, 0)  # 闂傚倷娴囧畷鍨叏閺夋嚚娲煛閸滀焦鏅悷婊勫灴婵＄敻骞囬弶璺ㄥ€炲銈嗗笂缁€渚€鍩€椤掆偓閻栧ジ寮婚埄鍐ㄧ窞濠电姴瀚搹搴☆渻閵堝棙鐓ラ柨鏇ㄤ邯瀵鈽夐姀鐘栥劑鏌ㄩ弴妤€浜鹃梺鍛婃煟閸婃繈寮诲☉娆愬劅妞ゆ牗绋戦锟�?

    height, width, channel = img.shape

    # b, g, r = cv2.split(img)

    # -----------------1.闂傚倸鍊风粈渚€宕ョ€ｎ喖纾块柟鎯版鎼村﹪鏌ら懝鎵牚濞存粌缍婇弻娑㈠Ψ椤旂厧顫梺缁樺笒椤兘鎮￠锕€鐐婇柕濠忚吂閹风懓顪冮妶搴′簼闁绘濞€瀵鏁愭径濞⑩晠鏌曟径鍫濆姶濞寸姴鍚嬬换婵嬪煕閳ь剟宕ㄩ娑欑€伴柣搴㈩問閸犳鎮￠敓鐘偓浣糕槈濮楀棙鍍靛銈嗘尵婵厼煤椤栫偞鈷掑ù锝呮嚈閸︻厾鐭欓柛鏇ㄥ灠缁狀垶鏌ㄩ悤鍌涘�?-----------
    # dstt = np.zeros((4, height, width), dtype=img.dtype)
    #
    # dstt[0][0:height, 0:width] = b
    # dstt[1][0:height, 0:width] = g
    # dstt[2][0:height, 0:width] = r
    # dstt[3][0:height, 0:width] = mask
    # cv2.imshow("fore.png", cv2.merge(dstt))

    # -----------------2.濠电姷鏁搁崑鐐哄垂閸洖绠伴柛婵勫劗閸嬫挸鈽夐幒鎾寸彋閻庢鍠氱划顖滄崲濠靛鐐婇柕濞垮劜閻ｉ亶姊虹拠鑼闁稿鍠栧畷鎴﹀箻缂佹ê浠у銈嗙墱閸嬬偤鎮￠弴銏＄厪濠㈣埖绋撻悾閬嶆煕閺傝鈧繈寮诲☉娆愬劅闁靛繒濮撮幗鐢告倵鐟欏嫭纾搁柛搴ㄦ涧閻ｉ攱绺介崨濠備簻婵＄偛顑呯花鐓幟归敓锟�?-----------
    # bg = np.zeros((3, height, width), dtype=img.dtype)  # 闂傚倸鍊烽悞锕傛儑瑜版帒鍨傚┑鐘宠壘缁愭鏌熼悧鍫熺凡闁搞劌鍊归幈銊ノ熼崹顔惧帿闂佹悶鍊曠€氫即骞冨鈧幃娆撴濞戞顥氶梻浣虹帛缁诲秹宕归崼鏇炶摕婵炴垯鍨瑰敮闂佹寧娲嶉崑鎾绘煕閺傝鈧繈寮诲☉娆愬劅妞ゆ牗绋戦锟�
    # bg[2][0:height, 0:width] = 255  # 闂傚倸鍊烽懗鍫曞储瑜斿畷顖炲锤濡も偓鐎氬銇勯幒鍡椾壕闂佸疇顕х粔鎾煝鎼淬劌绠荤€规洖娲ら埀顒傚仱濮婅櫣鎷犻垾宕団偓濠氭煕閹扳晛濡兼い銉︾洴濮婂宕掑▎鎴濆閻熸粍婢橀崯顐ゅ弲闂侀潧艌閺呮盯寮告笟鈧弻鏇㈠醇濠靛洤绐涘┑鈽嗗亽閸ㄥ爼寮婚弴锛勭杸闁靛／鍐ㄧ厒婵＄偑鍊戦崕濠氬箯閿燂拷
    bg = cv2.imread(background_root)
    bg = cv2.resize(bg,(768,768))
    bg = np.array(bg).transpose(2,0,1)
    # bg = cv2.resize(bg,(width,height))
    # dstt = np.zeros((3, height, width), dtype=img.dtype)
    c1 ,h1, w1 = bg.shape
    # rate = float(height/width)
    # if height>=h1:
    #     height = random.randint(int(0.95*h1),int(h1))
    #     width = int(height/rate)
    # if width>w1:
    #     width = random.randint(int(0.95*w1),int(w1))
    #     height = int(width*rate)
    height = 768
    width = 768
    img = cv2.resize(img,(width,height))
    mask = cv2.resize(mask,(width,height))
    dstt = bg
    dstt_mask = np.zeros((h1,w1))
    # print(height)
    # print(width)
    local = h1-height
    local1 = random.randint(0,w1-width)
    for i in range(3):
        dstt[i][local:height+local, local1:local1+width] = bg[i][local:local+height, local1:local1+width] * (255.0 - mask) / 255
        dstt[i][local:local+height, local1:local1+width] += np.array(img[:, :, i] * (mask / 255), dtype=np.uint8)
    # path_img = "/home/gaojun/data1/study/mydataset/fa2/image/"+str(j)+img_root.split('/')[-1]
    # path_alpha = "/home/gaojun/data1/study/mydataset/fa2/alpha/"+str(j)+mask_root.split('/')[-1]

    path_img = "/home/gaojun/data1/study/dataset_save/PPM_YOLO_MODNet_9w_clear/image/"+str(j)+img_root.split('/')[-1]
    path_alpha = "/home/gaojun/data1/study/dataset_save/PPM_YOLO_MODNet_9w_clear/alpha/"+str(j)+mask_root.split('/')[-1]
    path_img1 = "/home/gaojun/data1/study/dataset_save/PPM_YOLO_MODNet_9w_clear/image/"
    path_alpha1 = "/home/gaojun/data1/study/dataset_save/PPM_YOLO_MODNet_9w_clear/alpha/"
    if not os.path.exists(path_img1):
        os.makedirs(path_img1)
        os.makedirs(path_alpha1)
    # cv2.imwrite(path_img, cv2.merge(dstt))
    # cv2.imshow('t',cv2.resize(cv2.merge(dstt),(512,512)))
    dstt_mask[local:height+local,local1:local1+width] = mask
    h1_min, h1_max, w1_min, w1_max = crop1(dstt_mask)
    dstt = cv2.merge(dstt)[h1_min:h1_max,w1_min:w1_max]

    dstt = Image.fromarray(cv2.cvtColor(dstt,cv2.COLOR_BGR2RGB)).convert('RGB')  
    transformk = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    dstt = transformk(dstt)
    dstt = cv2.cvtColor(np.asarray(dstt),cv2.COLOR_RGB2BGR)
    # dstt = gaussian_noise(dstt)
    
    # gai = random.random()
    # if gai >0.5:
    #     dstt = add_noise_Guass(dstt)
    #     dstt = cv2.GaussianBlur(dstt, (3, 3), 0)
    # cv2.imwrite('./1.jpg', dstt)
    # print("ok")
    dstt_mask=dstt_mask[h1_min:h1_max,w1_min:w1_max]
    # cv2.imwrite("./1.jpg", dstt)
    cv2.imwrite(path_img, dstt)
    # cv2.imwrite("./2.jpg", dstt_mask)
    cv2.imwrite(path_alpha, dstt_mask)
    # dstt_mask=cv2.resize(dstt_mask,(512,512))
    # cv2.imshow('k',dstt_mask)
    # cv2.waitKey()
# img_root = "/home/gaojun/data1/study/mydataset/dataset/matting"
img_root = "/home/gaojun/data1/study/mydataset/new_data/image"
img_file = os.listdir(img_root)
img_path = sorted([os.path.join(img_root,img) for img in img_file])
label_root = "/home/gaojun/data1/study/mydataset/new_data/alpha"
label_file = os.listdir(label_root)
label_path = sorted([os.path.join(label_root,label) for label in label_file])
bg_root = "/home/gaojun/data1/study/mydataset/new_data/background"
bg_file = os.listdir(bg_root)
bg_path = sorted([os.path.join(bg_root,bg) for bg in bg_file])
# print(bg_path[1761])
# print(bg_path)
for i in range(len(img_path)):
    print(i)
    m = np.random.choice(len(bg_path),30,replace=False)
    m1 = np.random.choice(len(bg_path),30,replace=False)
    for j in range(len(m)):
        dataset_make(img_path[i],label_path[i],bg_path[m[j]],m1[j])