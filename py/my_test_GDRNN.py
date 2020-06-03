import os.path as osp
import glob
import cv2
import time
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data as data
from scipy.ndimage import gaussian_filter


def test(test_name):
    model_path = 'model_HSI_SR_DRNN2_totalloss_SAVE_s2_200/model_epoch_200.pth'  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
    filename_gt = 'D:/data/test/hr' + test_name + '.mat'
    filename_lr = 'D:/data/test/lr' + test_name + '.mat'
    device = torch.device('cuda')  # if you want to run on CPU, change 'cuda' -> cpu
    # device = torch.device('cpu')

    t1 = time.time()
    # model = arch.RRDBNet(31, 31, 64, 7, gc=32)
    model = torch.load(model_path)["model"]
    model = model.to(device)
    # model = model.cuda()



    img_gt = sio.loadmat(file_name=filename_gt)
    gt = img_gt['hr']
    gt = np.expand_dims(gt, axis=0)
    hr_t = np.transpose(gt, (0, 3, 2, 1))
    # print("hr shape {}".format(hr_t.shape))
    img_lr = sio.loadmat(file_name=filename_lr)
    lr = img_lr['lr']
    lr = np.expand_dims(lr, axis=0)
    lr_t = torch.from_numpy(np.transpose(lr, (0, 3, 2, 1)))
    lr_t = lr_t.float()
    # print("lr shape {}".format(lr_t.shape))

    lr_t = lr_t.cuda()
    output = model(lr_t)
    output = output.cpu().detach().numpy()
    recon = np.transpose(output, (0, 3, 2, 1))
    lr_t = lr_t.cpu().detach().numpy()
    res = recon - gt
    # print("res shape {}".format(res.shape))

    print(test_name)

    test_diff_sqr = abs(res)
    test_diff_sqr_avg = np.average(test_diff_sqr)
    print('l1: %.6f' % (test_diff_sqr_avg))

    test_diff_sqr = np.square(res)
    test_diff_sqr_avg = np.average(test_diff_sqr)
    print('l2: %.6f' % (test_diff_sqr_avg))

    test_psnr = -10.0 * np.log10(test_diff_sqr_avg)
    print('test_psnr: %.6f' % (test_psnr))

    cnt = time.time() - t1
    print('time cost {}'.format(cnt))
    print('')


if __name__ == "__main__":
    test_list = ['1', '2']
    for test_name in test_list:
        test(test_name)

