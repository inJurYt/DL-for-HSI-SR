import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary

class SubNetwork(nn.Module):
    def __init__(self, input_chns, output_chns, chns):
        super(SubNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_chns, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=output_chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, pre_feature):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        feature = self.relu(self.conv4(out3)) + pre_feature
        out = self.relu(self.conv5(feature)) + out3
        out = self.relu(self.conv6(out)) + out2
        out = self.relu(self.conv7(out)) + out1
        out = self.conv8(out) + x
        return out, feature

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
        out1, feature = self.conv1(x, 0)
        out1, feature = self.conv2(out1, feature)
        out1, feature = self.conv3(out1, feature)
        out2, feature = self.conv4(out1, feature)
        out2, feature = self.conv5(out2, feature)
        out2, feature = self.conv6(out2, feature)
        return out2
