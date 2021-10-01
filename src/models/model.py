import torchvision
import torch.nn as nn

"""
Class function used to replace some layer[s] of a network with identity
"""
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

    
"""
Returns a resnet18 network with some of the latter layers cut off and a variable number of channels in the input

args:

device --> Device to place the network on
n_cut --> Number of layers to remove from the network (should be in range 0 <= n_cut <= 3)
conv_dim --> Number of channels in the input images/spectrograms.

"""
def resnet18(device, n_cut=0, conv_dim=3):
    net = torchvision.models.resnet18(pretrained=False)
    net.conv1 = nn.Conv2d(conv_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    for i in range(n_cut):
        idx = 4 - i
        layer_string = 'layer' + str(idx)
        net._modules[layer_string] = Identity()
    net.fc = nn.Linear(int(512 / (2** n_cut)), 2, bias=True)
    net.to(device)
    return net
