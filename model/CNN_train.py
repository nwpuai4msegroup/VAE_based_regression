from PIL import Image
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from CNN_model import *
from torchvision import transforms
import pandas as pd
import cv2
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score as r2
import matplotlib.image as mpimg
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



# 数据加载
df_label = pd.read_csv('./label.csv')["YS_RT"]
root_dir = '../data/euler_200_cut'
dataset = Mydata(root_dir, df_label)
print(dataset)
#train_dataset, test_dataset = torch.utils.data.random_split(dataset, [400, 100])
train_dataset, test_dataset = torch.utils.data.Subset(dataset, range(0, 400)), torch.utils.data.Subset(dataset, range(400, 500))
train_loader = DataLoader(train_dataset, batch_size = 50, drop_last=True)#, shuffle=True
test_loader = DataLoader(test_dataset, batch_size = 50, drop_last=True)#, shuffle=True

#model = ResNet18()
model = CNN_model()
if torch.cuda.is_available():
    model = model.cuda()
print(model)

# 损失函数和优化器
loss_func = nn.MSELoss()
if torch.cuda.is_available():
    loss_func = loss_func.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.005)#, weight_decay=0.01
#optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200, verbose = True)

# 训练过程
epoch = 200
out = []
out_train = []
out_test = []
train_loss_list = []
test_loss_list = []
#total_step = 0
for i in range(epoch):
    #训练步骤
    model.train()
    print("-----------第{}轮-----------".format(i+1))
    for data in train_loader:
        imgs, labels = data
        imgs = Variable(imgs).type(torch.FloatTensor)#/255
        labels = Variable(labels).type(torch.FloatTensor)

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            labels = labels.cuda()
        outputs, _ = model(imgs) #, _
        outputs = outputs.squeeze(-1)
        if i == epoch-1:
            out_1 = outputs.data.cpu().numpy()
            out.append(out_1)
        loss = loss_func(outputs, labels)
        R2 = sum((outputs-labels.mean())**2)/sum((labels-labels.mean())**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    #测试步骤
    model.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, labels = data
            imgs = Variable(imgs).type(torch.FloatTensor)#/255
            labels = Variable(labels).type(torch.FloatTensor)

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            test_outputs, _ = model(imgs)#, _
            test_outputs = test_outputs.squeeze(-1)
            if i == epoch-1:
                out_2 = test_outputs.data.cpu().numpy()
                out_test.append(out_2)
            test_loss = loss_func(test_outputs, labels)
            #R2_test = sum((test_outputs-labels.mean())**2)/sum((labels-labels.mean())**2)
            R2_test = r2(test_outputs.cpu(), labels.cpu())

        #因为存在dropout层所以有这一步
        for data in train_loader:
            imgs, labels = data
            imgs = Variable(imgs).type(torch.FloatTensor)#/255
            labels = Variable(labels).type(torch.FloatTensor)

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                labels = labels.cuda()
            train_outputs, _ = model(imgs)#, _
            train_outputs = train_outputs.squeeze(-1)
            if i == epoch-1:
                out_3 = train_outputs.data.cpu().numpy()
                out_train.append(out_3)
            train_loss = loss_func(train_outputs, labels)
            #R2_train = sum((train_outputs-labels.mean())**2)/sum((labels-labels.mean())**2)
            R2_train = r2(train_outputs.cpu(), labels.cpu())
    scheduler.step()
    print("epoch: {}, Loss: {}, R_2: {}".format(i+1, loss.item(), R2.item()))
    print("epoch: {}, Train_Loss: {}, R_2: {}".format(i+1, train_loss.item(), R2_train.item()))
    print("epoch: {}, Test_Loss: {}, R_2: {}".format(i+1, test_loss.item(), R2_test.item()))
    train_loss_list.append(train_loss.item())
    test_loss_list.append(test_loss.item())


plt.figure()
plt.plot(range(200), train_loss_list)
plt.plot(range(200), test_loss_list)