import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import ScanReferDataModule
from capnet import CapNet

# prepare dataset and dataloader
data = ScanReferDataModule()

# create model
model = CapNet(val_tf_on=False)

# model = model.load_from_checkpoint(
#     checkpoint_path='/home/luk/DenseCap/scripts/ppt_model0527_pretrain_gru_decay1.ckpt')

file = '/home/luk/Downloads/epoch_1.pth'
checkpoint = torch.load(file)
model.softgroup_module.load_state_dict(checkpoint['net'])

# 创建 ModelCheckpoint 回调
checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints',
    filename='test_checkpoint_{epoch}_{step}',
    save_last=True,  # 保存最后一个检查点
    every_n_train_steps=2000  # 每隔 2000 个迭代保存一次
)

# start training
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=-1,
                     logger=pl.loggers.TensorBoardLogger('logs/'), log_every_n_steps=10,
                     num_sanity_val_steps=1, limit_test_batches=200, limit_val_batches=1,
                     callbacks=[checkpoint_callback])

# trainer.fit(model, data)

# 测试验证集上的表现
data.prepare_data()
data.setup(stage='')
trainer.test(model, data.test_dataloader())
