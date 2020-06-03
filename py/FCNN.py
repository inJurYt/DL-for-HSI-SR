import functools
import torch
import torch.nn as nn
from torchsummary import summary


class FCNN(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(FCNN, self).__init__()

        self.conv1 = nn.Conv3d(in_nc, 64, (7, 9, 9), stride=(1, 1, 1), padding=(3, 4, 4))
        self.conv2 = nn.Conv3d(64, 32, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv3 = nn.Conv3d(32, 9, (1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
        self.conv4 = nn.Conv3d(9, out_nc, (3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        # conv
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.conv4(out)
        return out

def main():
    print("===> Building model")

    model = FCNN(1, 1)
    print(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for name,parameters in model.named_parameters():
        print(name,':',parameters.size())

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)

    summary(model, (1, 103, 144, 144))


if __name__ == "__main__":
    main()
    exit(0)