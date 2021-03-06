import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary

class SubNetwork(nn.Module):
    def __init__(self, input_chns, output_chns, chns):
        super(SubNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_chns, out_channels=chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=chns, out_channels=chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=chns, out_channels=chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=chns, out_channels=chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=chns, out_channels=chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv6 = nn.Conv2d(in_channels=chns, out_channels=chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv7 = nn.Conv2d(in_channels=chns, out_channels=output_chns, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out1 = self.relu(self.conv1(x))
        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out = self.relu(self.conv4(out3)) + out3
        out = self.relu(self.conv5(out)) + out2
        out = self.relu(self.conv6(out)) + out1
        out = self.conv7(out)
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
