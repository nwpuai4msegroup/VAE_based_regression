from Function import *
from torch.utils.data import DataLoader

import torch
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from torch import optim

import torchvision
from torchvision import datasets, transforms
from torchvision.utils import save_image

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from VAE_model import *

#dataset
root_dir = "../data/euler_all_cut/"
dataset = image_data(root_dir)
#dataset = torch.load("/home/liaoweijie/Inconel_625/EBSD_images/euler_all_cut/")#all_ipf_cut.pt
print(len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [4000, 1000])
train_loader = DataLoader(train_dataset, batch_size = 50, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size = 50, shuffle=True, drop_last=True)

class Args:
  pass

# model
args = Args()
args.batch_size = 50
args.epochs = 200
args.cuda = True
args.log_interval = 10
args.model = 'optimal_sigma_vae'  # Which model to use:  mse_vae,  gaussian_vae, or sigma_vae or optimal_sigma_vae

if not torch.cuda.is_available():
  args.cuda = False
device = torch.device("cuda" if args.cuda else "cpu")


## Training Model
model = ConvVAE(device, 3, args).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 200, verbose = True)


def train(epoch):
    model.train()
    train_loss = 0
    l1_loss_train = []
    l1_loss_test = []
    #for batch_idx, data in enumerate(train_loader):
    for batch_idx, data in enumerate(train_loader):
        if epoch < 101:
            data = imgextend_1(data)
            #data = imgextend_erode(data)
            data = data.to(device)
            optimizer.zero_grad()

            # Run VAE
            recon_batch, mu, logvar = model(data)
            # Compute loss
        else:
            data = imgextend_1(data)
            #data = imgextend_erode(data)
            data_ext = imgextend_2(data)
            data = data.to(device)
            data_ext = data_ext.to(device)
            optimizer.zero_grad()
            # Run VAE
            recon_batch, mu, logvar = model(data_ext)
        # Compute loss
        rec, kl = model.loss_function(recon_batch, data, mu, logvar)

        total_loss = rec + kl
        total_loss.backward()
        train_loss += total_loss.item()
        optimizer.step()
        
        l1_loss = F.l1_loss(data, recon_batch)
        #l1_loss_train += l1_loss.item()
        
    train_loss /=  len(train_loader.dataset)
    l1_loss_train.append(l1_loss.item())
    
    print('====> Epoch: {} Average train loss: {:.4f}'.format(epoch, train_loss))
    print('====> Epoch: {} Average train L1loss: {:.4f}'.format(epoch, l1_loss))
    model.eval()
    with torch.no_grad():
        test_loss = 0
        for batch_idx, data in enumerate(test_loader):
            if epoch < 101:
                data = imgextend_1(data)
                #data = imgextend_erode(data)
                data = data.to(device)
                optimizer.zero_grad()

                # Run VAE
                recon_batch, mu, logvar = model(data)
                # Compute loss
            else:
                data = imgextend_1(data)
                #data = imgextend_erode(data)
                data_ext = imgextend_2(data)
                data = data.to(device)
                data_ext = data_ext.to(device)
                optimizer.zero_grad()
                # Run VAE
                recon_batch, mu, logvar = model(data_ext)

            rec, kl = model.loss_function(recon_batch, data, mu, logvar)
            total_loss = rec + kl
            test_loss += total_loss.item()
            
            l1_loss = F.l1_loss(data, recon_batch)
            #l1_loss_test += l1_loss.item()
        test_loss /=  len(test_loader.dataset)
        l1_loss_test.append(l1_loss.item())
        
        print('====> Epoch: {} Average test loss: {:.4f}'.format(epoch, test_loss))
        print('====> Epoch: {} Average test L1loss: {:.4f}'.format(epoch, l1_loss))
    return train_loss, test_loss, np.mean(l1_loss_train), np.mean(l1_loss_test)

train_loss_all = []
test_loss_all = []

train_l1loss_all = []
test_l1loss_all = []

## Training
for epoch in range(1, args.epochs + 1):
    train_loss, test_loss, l1_loss_train, l1_loss_test = train(epoch)
    train_loss_all.append(train_loss)
    test_loss_all.append(test_loss)
    
    train_l1loss_all.append(l1_loss_train)
    test_l1loss_all.append(l1_loss_test)
    
    scheduler.step()

plt.figure()
plt.plot(range(200), train_l1loss_all)
plt.plot(range(200), test_l1loss_all)