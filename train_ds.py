"""

Training script

"""

import os
from time import time

import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from visdom import Visdom

from dataset.dataset import Dataset

from loss.Dice import DiceLoss
from loss.ELDice import ELDiceLoss
from loss.WBCE import WCELoss
from loss.Jaccard import JaccardLoss
from loss.SS import SSLoss
from loss.Tversky import TverskyLoss
from loss.Hybrid import HybridLoss
from loss.BCE import BCELoss

from net.ResUNet import net

import parameter as para

# Set up visdom
viz = Visdom(port=666)
step_list = [0]
win = viz.line(X=np.array([0]), Y=np.array([1.0]), opts=dict(title='loss'))

# Set graphics card related
os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu
cudnn.benchmark = para.cudnn_benchmark

# Define the network
net = torch.nn.DataParallel(net).cuda()
net.train()

# Define Dataset
train_ds = Dataset(os.path.join(para.training_set_path, 'ct'), os.path.join(para.training_set_path, 'seg'))

# Define data loading
train_dl = DataLoader(train_ds, para.batch_size, True, num_workers=para.num_workers, pin_memory=para.pin_memory)

# Pick loss function
loss_func_list = [DiceLoss(), ELDiceLoss(), WCELoss(), JaccardLoss(), SSLoss(), TverskyLoss(), HybridLoss(), BCELoss()]
loss_func = loss_func_list[5]

# Define optimizer
opt = torch.optim.Adam(net.parameters(), lr=para.learning_rate)

# Learning rate decay
lr_decay = torch.optim.lr_scheduler.MultiStepLR(opt, para.learning_rate_decay)

# Depth supervision attenuation coefficient
alpha = para.alpha

# Training network
start = time()
for epoch in range(para.Epoch):

    lr_decay.step()

    mean_loss = []

    for step, (ct, seg) in enumerate(train_dl):

        ct = ct.cuda()
        seg = seg.cuda()

        outputs = net(ct)

        loss1 = loss_func(outputs[0], seg)
        loss2 = loss_func(outputs[1], seg)
        loss3 = loss_func(outputs[2], seg)
        loss4 = loss_func(outputs[3], seg)

        loss = (loss1 + loss2 + loss3) * alpha + loss4

        mean_loss.append(loss4.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 5 is 0:
            
            step_list.append(step_list[-1] + 1)
            viz.line(X=np.array([step_list[-1]]), Y=np.array([loss4.item()]), win=win, update='append')
            
            print('epoch:{}, step:{}, loss1:{:.3f}, loss2:{:.3f}, loss3:{:.3f}, loss4:{:.3f}, time:{:.3f} min'
                  .format(epoch, step, loss1.item(), loss2.item(), loss3.item(), loss4.item(), (time() - start) / 60))

    mean_loss = sum(mean_loss) / len(mean_loss)

    # Save the model
    if epoch % 50 is 0 and epoch is not 0:

        # The naming method of the network model is: the number of epoch rounds + the loss of the current minibatch + the average loss of the current epoch
        torch.save(net.state_dict(), './module/net{}-{:.3f}-{:.3f}.pth'.format(epoch, loss, mean_loss))

    # Attenuate the depth supervision coefficient
    if epoch % 40 is 0 and epoch is not 0:
        alpha *= 0.8

# In-depth supervision of coefficient changes
# 1.000
# 0.800
# 0.640
# 0.512
# 0.410
# 0.328
# 0.262
# 0.210
# 0.168
# 0.134
# 0.107
# 0.086
# 0.069
# 0.055
# 0.044
# 0.035
# 0.028
# 0.023
# 0.018
# 0.014
# 0.012
# 0.009
# 0.007
# 0.006
# 0.005
# 0.004
# 0.003
# 0.002
# 0.002
# 0.002
# 0.001
# 0.001
# 0.001
# 0.001
# 0.001
# 0.000
# 0.000
