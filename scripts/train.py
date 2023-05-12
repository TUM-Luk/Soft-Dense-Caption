import torch
import pytorch_lightning as pl

from datamodule import ScanReferDataModule
from capnet import CapNet


# prepare dataset and dataloader
data = ScanReferDataModule()
data.prepare_data()
data.setup(stage='fit')

# create model
model = CapNet()

# load pretrained model
file = '/home/luk/Downloads/epoch_10.pth'
net = torch.load(file)
model.softgroup_module.load_state_dict(net['net'], strict=True)

# start training
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=-1)
trainer.fit(model, data)
