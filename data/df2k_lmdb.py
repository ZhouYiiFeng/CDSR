import os
from data import multiscalesrdata
import pickle
import lmdb
import numpy as np
from data import common
import torch
import cv2

class DF2K_lmdb(multiscalesrdata.SRData):
    def __init__(self, args, name='DF2K', train=True, benchmark=False):
        super(DF2K_lmdb, self).__init__(args, name=name, train=train, benchmark=benchmark)
        if name is 'DF2K':
            self.env = lmdb.open(os.path.join(self.args.dir_data, 'df2k_imgs_train'), readonly=True, lock=False, readahead=False,
                     meminit=False)

    def _scan(self):
        # names_hr = super(DF2K, self)._scan()
        # names_hr = names_hr[self.begin - 1:self.end]
        meta_info = pickle.load(open(os.path.join(self.args.dir_data, 'meta_info.pkl'), 'rb'))
        paths = meta_info['keys']
        self.ress = meta_info['res']

        return paths

    def _set_filesystem(self, dir_data):
        super(DF2K_lmdb, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')

    def __getitem__(self, idx):
        idx_or = idx
        idx = self._get_index(idx)
        key = self.images_hr[idx]
        size_ = self.ress[idx]
        with self.env.begin(write=False) as txn:
            buf = txn.get(key.encode('ascii'))
        img_flat = np.frombuffer(buf, dtype=np.uint8)
        H, W, C = size_
        hr = img_flat.reshape(H, W, C)
        # cv2.imwrite("1.png", hr)
        # hr, filename = self._load_file(idx)
        hr = self.get_patch(hr)
        hr = [common.set_channel(img, n_channels=self.args.n_colors) for img in hr]
        hr_tensor = [common.np2Tensor(img, rgb_range=self.args.rgb_range)
                     for img in hr]

        return torch.stack(hr_tensor, 0), idx_or



    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)