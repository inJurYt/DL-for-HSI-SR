import torch
import numpy as np
import torch.nn as nn

class SubNetwork(nn.Module):
    def __init__(self, input_chns, output_chns, chns):
        super(SubNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_chns, out_channels=96, kernel_size=9, stride=1, padding=4, bias=True)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=output_chns, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out+x

class DRNN(nn.Module):
    def __init__(self, input_chns, chns):
        super(DRNN, self).__init__()
        self.conv1 = SubNetwork(input_chns, input_chns, chns)
        self.conv2 = SubNetwork(input_chns, input_chns, chns)
        self.conv3 = SubNetwork(input_chns, input_chns, chns)
        self.conv4 = SubNetwork(input_chns, input_chns, chns)
        self.conv5 = SubNetwork(input_chns, input_chns, chns)
        self.conv6 = SubNetwork(input_chns, input_chns, chns)


    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.conv2(out1)
        out1 = self.conv3(out1)
        out2 = self.conv4(out1)
        out2 = self.conv5(out2)
        out2 = self.conv6(out2)
        return out2