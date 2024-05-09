# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:07:41 2022

@author: l1415
"""
import torchvision
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import cv2
import random
import torch

# 从大图片中随机选择小图片
def random_cut_image(image, img_size, num_cut=20):      #宽为img_size的正方形图片
    width, height = image.size
    idx_width = np.random.randint(0, width-img_size, num_cut)
    idx_height = np.random.randint(0, height-img_size, num_cut)
    box_list = []
    # (left, upper, right, lower)
    for i in range(num_cut):
        box = (idx_width[i], idx_height[i], idx_width[i]+img_size, idx_height[i]+img_size)
        box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list

#图片保存
def save_images(image_list,dir_name,file_name):
    index = 1
    for image in image_list:
        image.save(dir_name + '/' + file_name+'_' + str(index) + '.jpg')
        index += 1
#有标签数据
class Mydata (Dataset):
    def __init__(self, root_dir, label):
        self.root_dir = root_dir
        #self.label = label
        self.img_path = sorted(os.listdir(self.root_dir))
        self.label = label

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        #img = img.resize((178, 218))
        #tensor_trans = torchvision.transforms.ToTensor()
        tensor_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
            #torchvision.transforms.Resize(128),
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = tensor_trans(img)
        i = idx//20
        label = self.label[i]
        return img, label
    
    def __len__(self):
        return len(self.img_path)
#无标签数据
class image_data(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #self.label = label
        self.img_path = sorted(os.listdir(self.root_dir))
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_item_path)
        #img = img.resize((178, 218))
        #tensor_trans = torchvision.transforms.ToTensor()
        tensor_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize(128),
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = tensor_trans(img)
        return img
    def __len__(self):
        return len(self.img_path)
#性能较低的那些图像
class image_data_min(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #self.label = label
        self.img_path = sorted(os.listdir(self.root_dir))
        self.img_path_min = []
        for i in range(len(self.img_path)):
            img_name = self.img_path[i]
            if img_name[:2] in ["05", "06", "07", "08", "09", "10", "19", "29"]:
                self.img_path_min.append(img_name)
    def __getitem__(self, idx):
        img_idx = self.img_path_min[idx]
        img_item_path = os.path.join(self.root_dir, img_idx)
        img = Image.open(img_item_path)
        tensor_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize(128),
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = tensor_trans(img)
        return img
    def __len__(self):
        return len(self.img_path_min)
#性能较高的那些图像
class image_data_max(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        #self.label = label
        self.img_path = sorted(os.listdir(self.root_dir))
        self.img_path_max = []
        for i in range(len(self.img_path)):
            img_name = self.img_path[i]
            if img_name[:2] in ["15", "24", "25", "26"]:
                self.img_path_max.append(img_name)
    def __getitem__(self, idx):
        img_idx = self.img_path_max[idx]
        img_item_path = os.path.join(self.root_dir, img_idx)
        img = Image.open(img_item_path)
        tensor_trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
            torchvision.transforms.Resize(128),
            #torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        img = tensor_trans(img)
        return img
    def __len__(self):
        return len(self.img_path_max)
#图像增强，还需增加其它方法
def imgextend(img):
    ansimg = torch.empty(img.shape)
    for i in range(img.shape[0]):
        imgi = img[i]
        flag = random.random() #随机翻转
        imgi = imgi.permute((1, 2, 0)).numpy()# transpose只能交换两个维度
        width, height, _ = imgi.shape
        if 0.5 <= flag < 0.6:
            imgi = cv2.flip(imgi, 1) #水平翻转函数
        elif 0.6 <= flag < 0.7:#二维旋转
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), np.random.randint(-20,20), 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        elif 0.8 <= flag < 0.9:#仿射变换
            pts1 = np.float32([[50,50],[200,50],[50,200]])
            pts2 = np.float32([[55,55],[190,50],[70,220]])
            M = cv2.getAffineTransform(pts1,pts2)
            imgi = cv2.warpAffine(imgi,M,(width,height))
        elif 0.9 <= flag < 1:#平移
            M = np.float32([[1,0,np.random.randint(0,10)],[0,1,np.random.randint(0,10)]])
            imgi = cv2.warpAffine(imgi,M,(width,height))
        elif 0.4 <= flag < 0.5:#高斯模糊
            imgi = cv2.GaussianBlur(imgi, (5, 5), 5)
        elif 0.3 <= flag < 0.4:#膨胀
            kenel = np.ones((2, 2), np.uint8)
            imgi = cv2.erode(imgi, kenel)
        elif 0.2 <= flag < 0.3:#腐蚀
            kenel = np.ones((2, 2), np.uint8)
            imgi = cv2.dilate(imgi, kenel)
        elif 0.1 <= flag < 0.2:#随机裁剪
            idx_width = np.random.randint(0, width-100)
            idx_height = np.random.randint(0, height-100)
            imgi = imgi[idx_width:idx_width+100, idx_height:idx_height+100]
        elif 0.7 <= flag < 0.8:#随机添加三块遮挡
            idx_width = np.random.randint(0, width-20, 3)
            idx_height = np.random.randint(0, height-20, 3)
            imgi_1 = imgi.copy()
            imgi_1[idx_width[0]:idx_width[0]+20, idx_height[0]:idx_height[0]+20] = 0
            imgi_1[idx_width[1]:idx_width[1]+20, idx_height[1]:idx_height[1]+20] = 0
            imgi_1[idx_width[2]:idx_width[2]+20, idx_height[2]:idx_height[2]+20] = 0
            imgi = imgi_1
        elif 0 <= flag < 0.1:#上下翻转
            imgi = cv2.flip(imgi, 0)
        else:
            imgi = imgi
        toten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
        imgi = toten(imgi)
        ansimg[i]= imgi
    return ansimg

#反转和旋转变化
def imgextend_1(img):
    ansimg = torch.empty(img.shape)
    for i in range(img.shape[0]):
        imgi = img[i]
        flag = random.random()*1.2 #随机
        imgi = imgi.permute((1, 2, 0)).numpy()# transpose只能交换两个维度
        width, height, _ = imgi.shape
        if 0 <= flag < 0.2:
            imgi = cv2.flip(imgi, 1) #水平翻转函数
        elif 0.2 <= flag < 0.4:#上下翻转
            imgi = cv2.flip(imgi, 0)
        elif 0.4 <= flag < 0.6:#二维旋转
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        elif 0.6 <= flag < 0.8:#二维旋转
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 180, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        elif 0.8 <= flag < 1.0:#二维旋转
            rotationMatrix = cv2.getRotationMatrix2D((width/2, height/2), 270, 1)
            imgi = cv2.warpAffine(imgi, rotationMatrix, (width, height))
        else:
            imgi = imgi
        toten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
        imgi = toten(imgi)
        ansimg[i]= imgi
    return ansimg

    #其它变化
def imgextend_2(img):
    ansimg = torch.empty(img.shape)
    for i in range(img.shape[0]):
        imgi = img[i]
        flag = random.random() #随机翻转
        imgi = imgi.permute((1, 2, 0)).numpy()# transpose只能交换两个维度
        width, height, _ = imgi.shape
        if 0 <= flag < 0.2:#随机添加三块遮挡
            idx_width = np.random.randint(0, width-20, 3)
            idx_height = np.random.randint(0, height-20, 3)
            imgi_1 = imgi.copy()
            imgi_1[idx_width[0]:idx_width[0]+20, idx_height[0]:idx_height[0]+20] = 0
            imgi_1[idx_width[1]:idx_width[1]+20, idx_height[1]:idx_height[1]+20] = 0
            imgi_1[idx_width[2]:idx_width[2]+20, idx_height[2]:idx_height[2]+20] = 0
            imgi = imgi_1
        elif 0.2 <= flag < 0.4:#高斯模糊
            imgi = cv2.GaussianBlur(imgi, (5, 5), 5)
        elif 0.4 <= flag < 0.6:#仿射变换
            pts1 = np.float32([[50,50],[200,50],[50,200]])
            pts2 = np.float32([[55,55],[190,50],[70,220]])
            M = cv2.getAffineTransform(pts1,pts2)
            imgi = cv2.warpAffine(imgi,M,(width,height))
        elif 0.6 <= flag < 0.8:#腐蚀
            kenel = np.ones((2, 2), np.uint8)
            imgi = cv2.dilate(imgi, kenel)
        else:#膨胀
            kenel = np.ones((2, 2), np.uint8)
            imgi = cv2.erode(imgi, kenel)
        toten = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
        imgi = toten(imgi)
        ansimg[i]= imgi
    return ansimg
    
