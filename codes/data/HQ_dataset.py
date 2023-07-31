import numpy as np
import torch
import torch.utils.data as data
import data.util as util
import cv2
import random

class HQDataset(data.Dataset):
    '''Read HQ images only.'''

    def __init__(self, opt):
        super(HQDataset, self).__init__()
        self.opt = opt
        self.paths_HQ = None
        self.HQ_env = None  # environment for lmdb

        # read image list from lmdb or image files
        self.paths_HQ, self.HQ_env  = util.get_image_paths(opt['data_type'], opt['dataroot_HQ'])
        assert self.paths_HQ, 'Error: HQ paths are empty.'

    def __getitem__(self, index):
        HQ_path = None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size'] if self.opt['GT_size'] is not None else self.opt['HQ_size']

        # get HQ image
        HQ_path = self.paths_HQ[index]
        img_HQ = util.read_img(self.HQ_env, HQ_path)
        H, W, C = img_HQ.shape

        if self.opt['phase'] == 'train':
            # randomly crop
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_HQ = img_HQ[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_HQ, _ = util.augment([img_HQ, img_HQ], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_HQ = util.channel_convert(C, self.opt['color'], [img_HQ])[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HQ.shape[2] == 3:
            img_HQ = img_HQ[:, :, [2, 1, 0]]
        img_HQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ, (2, 0, 1)))).float()

        #print(img_HQ.shape)
 
        return {'HQ': img_HQ, 'HQ_path': HQ_path, 'LQ': img_HQ, 'LQ_path': HQ_path}

    def __len__(self):
        return len(self.paths_HQ)
