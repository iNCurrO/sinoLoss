import os
import numpy as np
import zipfile
import PIL.Image
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import pyspng


class sinogramDataset(Dataset):
    def __init__(self,
                 path: str,
                 ):
        self._path = path
        self._all_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self._path)
            for root, _dirs, files in os.walk(self._path)
            for fname in files
        }

        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) == '.npy')
        if len(self._image_fnames) == 0:
            raise IOError(f'No image files found in the specified path : {self._path}')

        self._name = os.path.splitext(os.path.basename(self._path))[0]
        self._img_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

    def num_channels(self):
        return self._img_shape[1]

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self._image_fnames)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        image = np.load(os.path.join(self._path, fname))
        image = torchvision.transforms.functional.to_tensor(
            image,
        )
        return image

    def __getitem__(self, item):
        image = self._load_raw_image(item)
        assert list(image.shape) == self._img_shape[1:4], print(image.shape, self._img_shape[1:4])
        return image


class singleDataset(Dataset):
    def __init__(self,
                 path: str,
                 ):
        self._path = path
        self._all_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self._path)
            for root, _dirs, files in os.walk(self._path)
            for fname in files
        }

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self._name = os.path.splitext(os.path.basename(self._path))[0]
        self._img_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)

    def num_channels(self):
        return self._img_shape[1]

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self._image_fnames)

    def _open_file(self, fname):
        return open(os.path.join(self._path, fname), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = torchvision.transforms.functional.to_tensor(
            image,
        )
        return image

    def __getitem__(self, item):
        image = self._load_raw_image(item)
        assert list(image.shape) == self._img_shape[1:4], print(image.shape, self._img_shape[1:4])
        return image


class TotalDataset(Dataset):
    def __init__(self, inputdataset, sinodataset, targetdataset):
        self._inputdataset = inputdataset
        self._sinodataset = sinodataset
        self._targetdataset = targetdataset
        if self._sinodataset is not None:
            self._datasets = [self._inputdataset, self._sinodataset, self._targetdataset]
            assert len(self._inputdataset) == len(self._sinodataset),\
                f"{len(self._inputdataset)},{len(self._sinodataset)}"
        else:
            self._datasets = [self._inputdataset, self._targetdataset]
        assert len(self._inputdataset) == len(self._targetdataset)
        assert self._inputdataset.num_channels() == self._targetdataset.num_channels(), \
            print(f"input data {self._inputdataset.num_channels()}: target data{self._targetdataset.num_channels()}")
        self._datalen = len(self._targetdataset)

    def num_channels(self):
        return self._inputdataset.num_channels()

    def __getitem__(self, idx):
        return tuple([dataset[idx] for dataset in self._datasets])

    def __len__(self):
        return self._datalen


def set_dataset(config):
    basedir = os.path.join(config.datadir, config.dataname)
    __batchsize__ = config.batchsize

    # Dataset for training
    __inputdir__ = os.path.join(basedir, "view"+str(config.view)+"_recon")
    __sinodir__ = os.path.join(basedir, "view"+str(config.view)+"_sino")
    __targetdir__ = os.path.join(basedir, "Fullview_recon")
    ds = TotalDataset(
            inputdataset=sinogramDataset(path=__inputdir__),
            sinodataset=sinogramDataset(path=__sinodir__),
            targetdataset=sinogramDataset(path=__targetdir__)
        )

    # Dataset for validation
    __inputdir__ = os.path.join(basedir, "view"+str(config.view)+"_recon_val")
    __sinodir__ = os.path.join(basedir, "view"+str(config.view)+"_sino_val")
    __targetdir__ = os.path.join(basedir, "Fullview_recon_val")
    ds_v = TotalDataset(
            inputdataset=sinogramDataset(path=__inputdir__),
            sinodataset=sinogramDataset(path=__sinodir__),
            targetdataset=sinogramDataset(path=__targetdir__)
        )

    return DataLoader(
        dataset=ds,
        batch_size=__batchsize__,
        shuffle=True,
        num_workers=config.numworkers,
        pin_memory=True
    ), DataLoader(
        dataset=ds_v,
        batch_size=config.valbatchsize,
        shuffle=False,
        num_workers=config.numworkers,
        pin_memory=True
    ), ds.num_channels()
