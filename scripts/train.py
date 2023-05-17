import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from datamodule import ScanReferDataModule
from capnet import CapNet

# prepare dataset and dataloader
data = ScanReferDataModule()

# create model
model = CapNet()
file = '/home/luk/DenseCap/scripts/model_checkpoint_epoch1.ckpt'
checkpoint = torch.load(file)
model.load_state_dict(checkpoint['state_dict'])

# start training
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=2, limit_train_batches=500,
                     logger=pl.loggers.TensorBoardLogger('logs/'), log_every_n_steps=10)
trainer.fit(model, data)
