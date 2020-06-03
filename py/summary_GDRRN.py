import os
import argparse
import scipy.io as sio
import numpy as np
import time
import random
import torch
import glob
#from scipy.misc import imresize
import h5py
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import torch.nn as nn
from torch.autograd import Variable
from models_GDRRN import DRNN
from torchsummary import summary
from torch.utils.data import DataLoader
import torch.utils.data as data


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sam_lamd = 0.1
mse_lamd = 1
group = 2
if_control_blc = False

sigma = 25


def main():
    print("===> Building model")
    model = DRNN(input_chns=103, chns=128)
    print(model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    summary(model, (103, 144, 144))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)
if __name__ == "__main__":
    main()
    exit(0)
