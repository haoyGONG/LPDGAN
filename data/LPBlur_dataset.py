import os
from glob import glob
from glog import logger
from torch.utils.data import Dataset
from data import aug
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd


class LPBlurDataset(Dataset):
    def __init__(self, opt):
        super(LPBlurDataset, self).__init__()
        self.opt = opt

        self.files_a = os.path.join(opt.dataroot, opt.mode, 'blur')
        self.files_b = os.path.join(opt.dataroot, opt.mode, 'sharp')

        self.blur = glob(os.path.join(self.files_a, '*.jpg'))
        self.sharp = glob(os.path.join(self.files_b, '*.jpg'))
        assert len(self.blur) == len(self.sharp)

        if self.opt.mode == 'train':
            df = pd.read_csv(os.path.join(opt.dataroot, 'plate_info.txt'), header=None,
                             names=['ImageName', 'PlateInfo'])
            self.txt = df.set_index('ImageName')['PlateInfo'].to_dict()
            self.transform_fn = aug.get_transforms(size=(112, 224))
            self.transform_fn1 = aug.get_transforms(size=(56, 112))
            self.transform_fn2 = aug.get_transforms(size=(28, 56))
            self.transform_fn3 = aug.get_transforms(size=(14, 28))

        else:
            self.transform_fn = aug.get_transforms_fortest(size=(112, 224))
            self.transform_fn1 = aug.get_transforms_fortest(size=(56, 112))
            self.transform_fn2 = aug.get_transforms_fortest(size=(28, 56))
            self.transform_fn3 = aug.get_transforms_fortest(size=(14, 28))

        self.normalize_fn = aug.get_normalize()
        logger.info(f'Dataset has been created with {len(self.blur)} samples')

    def __len__(self):
        return len(self.blur)

    def __getitem__(self, idx):
        blur_image = Image.open(self.blur[idx])
        sharp_image = Image.open(self.sharp[idx])
        blur_image = np.array(blur_image)
        sharp_image = np.array(sharp_image)

        blur_image, sharp_image = self.transform_fn(blur_image, sharp_image)
        blur_image1, sharp_image1 = self.transform_fn1(blur_image, sharp_image)
        blur_image2, sharp_image2 = self.transform_fn2(blur_image, sharp_image)
        blur_image3, sharp_image3 = self.transform_fn3(blur_image, sharp_image)

        blur_image, sharp_image = self.normalize_fn(blur_image, sharp_image)
        blur_image1, sharp_image1 = self.normalize_fn(blur_image1, sharp_image1)
        blur_image2, sharp_image2 = self.normalize_fn(blur_image2, sharp_image2)
        blur_image3, sharp_image3 = self.normalize_fn(blur_image3, sharp_image3)

        blur_image = transforms.ToTensor()(blur_image)
        sharp_image = transforms.ToTensor()(sharp_image)
        blur_image1 = transforms.ToTensor()(blur_image1)
        sharp_image1 = transforms.ToTensor()(sharp_image1)
        blur_image2 = transforms.ToTensor()(blur_image2)
        sharp_image2 = transforms.ToTensor()(sharp_image2)
        blur_image3 = transforms.ToTensor()(blur_image3)
        sharp_image3 = transforms.ToTensor()(sharp_image3)

        if self.opt.mode == 'train':
            plate_info = self.txt[os.path.basename(self.sharp[idx])]
            try:
                plate_info = np.fromstring(plate_info, sep=' ')

            except (SyntaxError, ValueError) as e:
                print(f"Error restoring array: {e}")

            plate_info = torch.from_numpy(plate_info)

            return {'A': blur_image, 'B': sharp_image, 'A_paths': self.blur[idx], 'B_paths': self.sharp[idx],
                    'A1': blur_image1, 'B1': sharp_image1, 'A2': blur_image2, 'B2': sharp_image2, 'A3': blur_image3,
                    'B3': sharp_image3, 'plate_info': plate_info}  # A: blur B: sharp

        else:
            return {'A': blur_image, 'B': sharp_image, 'A_paths': self.blur[idx], 'B_paths': self.sharp[idx],
                    'A1': blur_image1, 'B1': sharp_image1, 'A2': blur_image2, 'B2': sharp_image2, 'A3': blur_image3,
                    'B3': sharp_image3}

    def load_data(self):
        dataloader = torch.utils.data.DataLoader(
            self,
            batch_size=self.opt.batch_size,
            shuffle=True,
            num_workers=int(self.opt.num_threads))
        return dataloader


def create_dataset(opt):
    return LPBlurDataset(opt).load_data()






