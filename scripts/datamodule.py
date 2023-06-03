import os
import sys

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from lib.dataset import ScannetReferenceDataset, ScannetReferenceTestDataset
from lib.dataset import get_scanrefer

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder


class ScanReferDataModule(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.Scanrefer_eval_val_scene = None
        self.dataset_val = None
        self.dataset_test = None
        self.dataset_train = None
        self.all_scene_list = None
        self.test_scene_list = None
        self.Scanrefer_eval_val = None
        self.Scanrefer_eval_train = None
        self.Scanrefer_train = None

    def prepare_data(self):
        self.Scanrefer_train, self.Scanrefer_eval_train, self.Scanrefer_eval_val, self.all_scene_list, self.Scanrefer_eval_val_scene = get_scanrefer(
            model='')

    def setup(self, stage: str):
        self.dataset_train = ScannetReferenceDataset(
            scanrefer=self.Scanrefer_train,
            scanrefer_all_scene=self.all_scene_list,
            split='train',
            num_points=50000,
            augment=True,
        )
        self.dataset_val = ScannetReferenceDataset(
            scanrefer=self.Scanrefer_eval_val,
            scanrefer_all_scene=self.all_scene_list,
            split='val',
            num_points=40000,
            augment=False,
        )

        # test要改的
        self.dataset_test = ScannetReferenceDataset(
            scanrefer=self.Scanrefer_eval_val_scene,
            scanrefer_all_scene=self.all_scene_list,
            split='val',
            num_points=40000,
            augment=False,
        )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=4, shuffle=True, num_workers=4,
                          collate_fn=self.dataset_train.collate_fn, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=4, shuffle=False, num_workers=4,
                          collate_fn=self.dataset_val.collate_fn, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=1, shuffle=False, num_workers=4,
                          collate_fn=self.dataset_test.collate_fn)


#
#
# test = ScanReferDataModule()
# test.prepare_data()
# test.setup(stage='fit')
#
# print(test.dataset_test[0])
# print(len(test.dataset_test))

# print(test.dataset_train.scene_data['scene0394_00']['instance_labels'])
# print(test.dataset_train.scene_data['scene0394_00']['instance_labels'].max())
# print(test.dataset_train.scene_data['scene0394_00']['instance_labels'].size)
# print(test.dataset_train.scene_data['scene0497_00']['instance_labels'].size)
# for i in test.train_dataloader():
#     print(i['scan_ids'])
#     print(i['instance_id'])
#     break


# for i in test.train_dataloader():
#     print(i['batch_idxs'])
#     print(i['batch_idxs'].shape)
#     break
#
#
#
#
# # print(test.dataset_train[0]['coord_float'])
# # print(test.dataset_train[0]['coord_float'].shape)
# #
#
# for i in test.train_dataloader():
#     print(i['instance_id'])
#     print(i['inst_nums'])
#     break

# for i in test.train_dataloader():
#     print(i['object_id_labels'])
#     print(i['object_id_labels'].max())
#     print(i['instance_labels'])
#     print(i['instance_labels'].max())
#     a = i['object_id_labels'] == 19
#     b = i['instance_labels'] == 19
#     print(False in (a==b))
#     break
