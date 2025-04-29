import torch
import os

from models.arch.DexNet import DExNet, DExNet_expand


def DExNet(in_channels=1475, out_channels=3, **kwargs):
    return DExNet_expand(in_channels, out_channels, **kwargs)



if __name__ == '__main__':
    x = torch.ones(1, 1475, 256, 256)
    net = DExNet(1475, 3)
    print(net)
    url = "./tmp.pth"
    torch.save(net.state_dict(), url)
    print('\n', os.path.getsize(url) / (1024 * 1024), 'MB')
    l, r, rr = net(x)