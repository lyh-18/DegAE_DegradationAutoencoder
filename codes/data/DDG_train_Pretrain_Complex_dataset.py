import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
import data.bsrgan_degradation_transfer as bsrgan_degradation


class DDGTrainDataset(data.Dataset):
    def __init__(self, opt):
        super(DDGTrainDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_HQ = None
        self.sizes_HQ = None
        self.HQ_env = None  # environment for lmdb

        self.paths_HQ, self.sizes_HQ = util.get_image_paths(self.data_type, opt['dataroot_HQ'])
        assert self.paths_HQ, 'Error: HQ path is empty.'
        
        self.random_scale_list = [1]

        

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.HQ_env = lmdb.open(self.opt['dataroot_HQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if self.HQ_env is None:
                self._init_lmdb()
        HQ_path = None
        scale = self.opt['scale']
        HQ_size = self.opt['HQ_size']

        # get HQ image
        HQ1_path = self.paths_HQ[index]
        random_index = random.randint(0, len(self.paths_HQ)-1)
        HQ2_path = self.paths_HQ[random_index]
        
        #print(HQ1_path)
        #print(HQ2_path)
        
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_HQ[index].split('_')]
        else:
            resolution = None
        img_HQ1 = util.read_img(self.HQ_env, HQ1_path, resolution)
        img_HQ2 = util.read_img(self.HQ_env, HQ2_path, resolution)
        
        if self.opt['phase'] == 'train':
            # if the image size is too small
            #print(HQ_size)
            H, W, _ = img_HQ1.shape
            #print(H, W)
            if H < HQ_size or W < HQ_size:
                img_HQ1 = cv2.resize(np.copy(img_HQ1), (HQ_size, HQ_size),
                                    interpolation=cv2.INTER_LINEAR)
                
            H, W, _ = img_HQ2.shape
            if H < HQ_size or W < HQ_size:
                img_HQ2 = cv2.resize(np.copy(img_HQ2), (HQ_size, HQ_size),
                                    interpolation=cv2.INTER_LINEAR)
                
            #('111: ', img_HQ2.shape)
        
        
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_HQ1 = util.modcrop(img_HQ1, scale)
            img_HQ2 = util.modcrop(img_HQ2, scale)
        
        # change gray to 3 channel
        if len(img_HQ1.shape) != 3: 
            img_HQ1 = np.expand_dims(img_HQ1, axis=2)
            img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=-1)
        else:
            if img_HQ1.shape[2] != 3:
                img_HQ1 = np.concatenate((img_HQ1, img_HQ1, img_HQ1), axis=-1)
                
        if len(img_HQ2.shape) != 3: 
            img_HQ2 = np.expand_dims(img_HQ2, axis=2)
            img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=-1)
        else:
            if img_HQ2.shape[2] != 3:
                img_HQ2 = np.concatenate((img_HQ2, img_HQ2, img_HQ2), axis=-1)
        
        # change color space if necessary
        if self.opt['color']:
            img_HQ1 = util.channel_convert(img_HQ1.shape[2], self.opt['color'], [img_HQ1])[0]
            img_HQ2 = util.channel_convert(img_HQ2.shape[2], self.opt['color'], [img_HQ2])[0]

        # get synthetic LQ image
        # randomly generate LQ images during training
        
        img_LQ1 = img_HQ1.copy()
        img_LQ2 = img_HQ2.copy()



        #######################################################
        # Add degradation to get target LQ1 and reference LQ2 #
        #######################################################
     
        # add complex degradations
        clean_prob, blur_prob, gaussian_prob, poisson_prob, speckle_prob, jpeg_prob = 0.3, 0.7, 0.5, 0.2, 0.2, 0.5
        img_LQ_list, _, degradation_record = bsrgan_degradation.degradation_transfer_lyh([img_HQ1.copy(), img_HQ2.copy()], 
                                            clean_prob, blur_prob, gaussian_prob, poisson_prob, speckle_prob, jpeg_prob, sf=1, shuffle_prob=0.5, use_sharp=False)
        img_LQ1 = img_LQ_list[0]
        img_LQ2 = img_LQ_list[1]
            

 
        
        if img_LQ1.ndim == 2:
            img_LQ1 = np.expand_dims(img_LQ1, axis=2)
            img_LQ2 = np.expand_dims(img_LQ2, axis=2)
        
        target_deg_type = '_'.join(degradation_record)

        ####################################
        # Add degradation to get input HQ  #
        ####################################
        input_noise_flag = False
        input_blur_flag = False
        # randomly apply Gaussian blur
        clean_prob, blur_prob, gaussian_prob, poisson_prob, speckle_prob, jpeg_prob = 0.3, 0.7, 0.5, 0.2, 0.2, 0.5
        img_HQ_list, _, degradation_record = bsrgan_degradation.degradation_transfer_lyh([img_HQ1.copy(), img_HQ2.copy()], 
                                        clean_prob, blur_prob, gaussian_prob, poisson_prob, speckle_prob, jpeg_prob, sf=1, shuffle_prob=0.5, use_sharp=False)
                                        
                                        
        img_HQ1 = img_HQ_list[0]
        img_HQ2 = img_HQ_list[1]

        input_deg_type = '_'.join(degradation_record)




        if self.opt['phase'] == 'train':
            

            # randomly crop
            H1, W1, C = img_LQ1.shape
            LQ_size = HQ_size // scale
            rnd_h = random.randint(0, max(0, H1 - LQ_size))
            rnd_w = random.randint(0, max(0, W1 - LQ_size))
            img_LQ1 = img_LQ1[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_HQ, rnd_w_HQ = int(rnd_h * scale), int(rnd_w * scale)
            img_HQ1 = img_HQ1[rnd_h_HQ:rnd_h_HQ + HQ_size, rnd_w_HQ:rnd_w_HQ + HQ_size, :]
            
            H2, W2, C = img_LQ2.shape
            LQ_size = HQ_size // scale
            rnd_h = random.randint(0, max(0, H2 - LQ_size))
            rnd_w = random.randint(0, max(0, W2 - LQ_size))
            img_LQ2 = img_LQ2[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_HQ, rnd_w_HQ = int(rnd_h * scale), int(rnd_w * scale)
            img_HQ2 = img_HQ2[rnd_h_HQ:rnd_h_HQ + HQ_size, rnd_w_HQ:rnd_w_HQ + HQ_size, :]

            # augmentation - flip, rotate
            img_LQ1, img_LQ2, img_HQ1, img_HQ2 = util.augment([img_LQ1, img_LQ2, img_HQ1, img_HQ2], self.opt['use_flip'],
                                          self.opt['use_rot'])
            
            
            #print(img_LQ1.shape, img_LQ2.shape, img_HQ1.shape, img_HQ2.shape)

        # change color space if necessary
        if self.opt['color']:
            img_LQ1 = util.channel_convert(C, self.opt['color'], [img_LQ1])[0]  # TODO during val no definition
            img_LQ2 = util.channel_convert(C, self.opt['color'], [img_LQ2])[0]  # TODO during val no definition
            
                 

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_HQ1.shape[2] == 3:
            img_HQ1 = img_HQ1[:, :, [2, 1, 0]]
            img_HQ2 = img_HQ2[:, :, [2, 1, 0]]
            img_LQ1 = img_LQ1[:, :, [2, 1, 0]]
            img_LQ2 = img_LQ2[:, :, [2, 1, 0]]
            
        #print(input_deg_type, target_deg_type)
         
        #import matplotlib.pyplot as plt
        #plt.subplot(2,2,1)
        #plt.imshow(img_HQ1)
        #plt.subplot(2,2,2)
        #plt.imshow(img_LQ1)
        #plt.subplot(2,2,3)
        #plt.imshow(img_HQ2)
        #plt.subplot(2,2,4)
        #plt.imshow(img_LQ2)
        #plt.show()
        
        
        
        img_HQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ1, (2, 0, 1)))).float()
        img_HQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_HQ2, (2, 0, 1)))).float()
        img_LQ1 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ1, (2, 0, 1)))).float()
        img_LQ2 = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ2, (2, 0, 1)))).float()
        
        #print('xxxxxxxxxxx')
        #import pdb 
        #pdb.set_trace()


        return {'LQ1': img_LQ1, 'LQ2': img_LQ2, 'HQ1': img_HQ1, 'HQ2': img_HQ2, 
                'HQ1_path': HQ1_path, 'HQ2_path': HQ2_path,
                'input_deg_type': input_deg_type, 'target_deg_type': target_deg_type}

    def __len__(self):
        return len(self.paths_HQ)
