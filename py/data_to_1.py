import os
import argparse
import scipy.io as sio
import numpy as np
import time
import random
import torch

path = 'D:\data\PaviaU'
input = sio.loadmat(path)
mat = input['paviaU']
print('mat {}'.format(mat.shape))
print(mat)
print(mat.max())
hr = np.array(mat,dtype=np.float32)
hr = hr/mat.max()
print(hr)
print('hr {}'.format(hr.shape))
sio.savemat('paviaU_to_1.mat', {'paviaU':hr})