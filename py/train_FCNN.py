import os
import argparse
import scipy.io as sio
import numpy as np
import time
import random
import torch
import glob
import h5py
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as func
import torch.nn as nn
import FCNN
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as data


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sam_lamd = 0.1
mse_lamd = 1
group = 2
if_control_blc = False
sigma = 25
scale = 2
method_name = 'PaviaU_FCNN_batch4_SAVE_s3_200'
sz = 72
rsz = 144
nspec = 103
batchSize = 4
def main():
    # Training settings
    nEpochs = 200
    lr = 0.001
    step = 80
    cuda = True
    resume = ""
    start_epoch = 1
    clip = 0.005
    threads = 1
    momentum = 0.9
    weight_decay = 1e-4

    # model_path = 'model_HSI_SR_DUALNET2_d2_totalloss_SAVE_s2_250/model_epoch_150.pth'

    path_hr = 'D:\data\hr.mat'
    path_lr = 'D:\data\lr_3.mat'

    global model

    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = True

    # print("===> Loading datasets")
    # train_set = DatasetFromMat(opt.dataset, sigma)
    # train_set = DatasetFromMat7_3(opt.dataset)

    print("===> Loading datasets ")
    train_data_lr, train_data_hr = DataFromMat(path_lr, path_hr)
    _, _, S, H, W = train_data_lr.shape
    torch_dataset = data.TensorDataset(train_data_lr, train_data_hr)
    training_data_loader = data.DataLoader(
        dataset=torch_dataset,
        num_workers=threads,
        batch_size=batchSize,
        shuffle=False
    )

    print("===> Building model")
    #  model = torch.load(model_path)["model"]
    model = FCNN.FCNN(1, 1)
    # criterion = nn.MSELoss()
    # criterion = nn.MSELoss(size_average=False)

    print("===> Setting GPU")
    if cuda:
        # model = to2rch.nn.DataParallel(model).cuda()
        model = dataparallel(model, 1)
        # set the number of parallel GPUs
        # criterion = criterion.cuda()
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(resume))

    print("===> Setting Optimizer")
    # optimizer = optim.SGD([
    #     {'params': model.parameters()}
    # ], lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    optimizer = optim.Adam([
        {'params': model.parameters()}
    ], lr=lr, weight_decay=weight_decay)

    print("===> Training")
    lossAarry = np.zeros(nEpochs)
    losspath = 'losses/'
    if not os.path.exists(losspath):
        os.makedirs(losspath)

    for epoch in range(start_epoch, nEpochs + 1):
        start_time = time.time()
        lossAarry[epoch - 1] = lossAarry[epoch - 1] + train(training_data_loader, optimizer, model, epoch, step, lr, cuda)
        print("===> Epoch[{}]: Loss={:.7f}, time = {:.4f}".format(epoch, lossAarry[epoch - 1], time.time() - start_time))
        save_checkpoint(model, epoch)

    sio.savemat(losspath + method_name + '_lossArray.mat', {'lossArray': lossAarry})


def train(training_data_loader, optimizer, model, epoch, step, lr, cuda):
    lr = adjust_learning_rate(lr, epoch - 1, step)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch={}, low_lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    start_time = time.time()

    model.train()
    lossValue = 0

    for iteration, (hsi, label) in enumerate(training_data_loader):
        if cuda:
            hsi = hsi.cuda()
            label = label.cuda()
        output = model(hsi)
        # loss = criterion(res, label)
        lossfunc = myloss_spe(hsi.data.shape[0], lamd=sam_lamd, mse_lamd=mse_lamd)

        loss = lossfunc.forward(output, label)

        # loss = criterion(res, label)/(input.data.shape[0]*2)

        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm(model.parameters(), opt.clip)
        optimizer.step()

        lossValue = lossValue + loss.data.item()
        if (iteration + 1) % 1 == 0:
            elapsed_time = time.time() - start_time
            # save_checkpoint(model, iteration)
            print("===> Epoch[{}]: iteration[{}]: Loss={:f}, time = {:f}".format(epoch, iteration + 1,
                                                                                     # criterion(lres + hres, target).data[0], loss_low.data[0], 0, elapsed_time))
                                                                                     loss.data.item(), elapsed_time))

    elapsed_time = time.time() - start_time
    lossValue = lossValue / (iteration + 1)
    # print("===> Epoch[{}]: Loss={:.5f}, time = {:.4f}".format(epoch, lossValue, elapsed_time))
    return lossValue


class myloss_spe(nn.Module):
    def __init__(self, N, lamd=1e-2, mse_lamd=1, epoch=None):
        super(myloss_spe, self).__init__()
        self.N = N
        self.lamd = lamd
        self.mse_lamd = mse_lamd
        self.epoch = epoch
        return

    def forward(self,res, label):
        loss = func.mse_loss(res, label, size_average=True)
        return loss


def adjust_learning_rate(lr, epoch, step):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    # if epoch < step:
    #     lr = opt.lr #* (0.1 ** (epoch // opt.step))#0.2
    # elif epoch < 3 * step:
    #     lr = opt.lr * 0.1 #* (0.1 ** (epoch // opt.step))#0.2
    # elif epoch < 5 * step:
    #     lr = opt.lr * 0.01  # * (0.1 ** (epoch // opt.step))#0.2
    # else:
    #     lr = opt.lr * 0.001
    lr = lr * (0.1 ** (epoch // step))  # 0.2
    return lr


class DatasetFromHDF5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHDF5, self).__init__()
        self.file_path = file_path
        data = h5py.File(os.path.join(self.file_path, 'gt.h5'), 'r')
        self.keys = list(data.keys())
        random.shuffle(self.keys)
        data.close()

    def __getitem__(self, index):
        hdf_gt = h5py.File(os.path.join(self.file_path, 'gt.h5'), 'r')
        key = str(self.keys[index])
        hdf_c = h5py.File(os.path.join(self.file_path, 'c.h5'), 'r')
        hdf_hsi = h5py.File(os.path.join(self.file_path, 'hsi_t.h5'), 'r')
        # test patch pair
        # hsi_ = np.array(hdf_hsi[key])
        # c_ = np.array(hdf_c[key])
        # gt_ = np.array(hdf_gt[key])
        # sio.savemat('tmp.mat', {'hsi': hsi_, 'c': c_, 'gt': gt_})
        gt = torch.from_numpy(np.array(hdf_gt[key], dtype=np.float32))
        c = torch.from_numpy(np.array(hdf_c[key], dtype=np.float32))
        hsi = torch.from_numpy(np.array(hdf_hsi[key], dtype=np.float32))

        hdf_gt.close()
        hdf_c.close()
        hdf_hsi.close()
        return hsi, c, gt

    def __len__(self):
        return len(self.keys)


def DataFromMat(path_lr, path_hr):
    mat_lr = sio.loadmat(path_lr)
    mat_hr = sio.loadmat(path_hr)
    lr_t = np.transpose(mat_lr['lr'], (0, 3, 2, 1))
    hr_t = np.transpose(mat_hr['lr'], (0, 3, 2, 1))
    lr = torch.from_numpy(np.expand_dims(np.array(lr_t, dtype=np.float32), axis=1))
    hr = torch.from_numpy(np.expand_dims(np.array(hr_t, dtype=np.float32), axis=1))
    print('lr {}'.format(lr.shape))
    print('hr {}'.format(hr.shape))
    """
    lr_t = np.transpose(lr,(3,0,2,1))
    hr_t = np.transpose(hr,(3,0,2,1))
    print('lr_t {}'.format(lr_t.shape))
    print('hr_t {}'.format(hr_t.shape))
    return lr_t, hr_t
    """
    return lr, hr

def save_checkpoint(model, epoch):
    fold = "model_" + method_name + "/"
    model_out_path = fold + "model_epoch_{}.pth".format(epoch)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(fold):
        os.makedirs(fold)

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))


def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0 + ngpus))
    assert torch.cuda.device_count() >= gpu0 + ngpus
    if ngpus > 1:
        if not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    elif ngpus == 1:
        model = model.cuda()
    return model


if __name__ == "__main__":
    main()

