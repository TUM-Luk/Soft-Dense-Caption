import torch

file = '/home/luk/Downloads/softgroup_scannet_spconv2.pth'
file2 = '/home/luk/Downloads/hais_ckpt_spconv2.pth'
net = torch.load(file)
net2 = torch.load(file2)


for k in net.keys():
    print(k)

for key,value in net['net'].items():
    print(key, value.size(), sep='\t')
