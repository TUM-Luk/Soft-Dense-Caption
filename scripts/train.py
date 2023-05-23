import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import ScanReferDataModule
from capnet import CapNet

# prepare dataset and dataloader
data = ScanReferDataModule()

# create model
model = CapNet(val_tf_on=False)

# model = model.load_from_checkpoint(
#     checkpoint_path='/home/luk/DenseCap/scripts/model0523_pretrian_decay_avgobject_epoch1.ckpt')

file = '/home/luk/Downloads/epoch_26.pth'
checkpoint = torch.load(file)
model.softgroup_module.load_state_dict(checkpoint['net'])

# # freeze sofrgroup
# for param in model.softgroup_module.parameters():
#     param.requires_grad = False

# start training
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=-1,
                     logger=pl.loggers.TensorBoardLogger('logs/'), log_every_n_steps=10,
                     num_sanity_val_steps=10)

trainer.fit(model, data)

# # 测试no_tf下的语句生成
# data.prepare_data()
# data.setup(stage='')
# trainer.test(model, data.val_dataloader())
