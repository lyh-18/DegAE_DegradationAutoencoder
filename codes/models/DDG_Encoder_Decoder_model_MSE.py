import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks_DDG as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss, PatchGANLoss, MSE_blur_loss
from models.degradations_pytorch import *

from models.modules.loss import GaussianSmoothing_withPad
import torch.optim as optim

import random

logger = logging.getLogger('base')

def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)
    
    return img, mask


class DDGModel(BaseModel):
    def __init__(self, opt):
        super(DDGModel, self).__init__(opt)
        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netEncoder = networks.define_DDE(opt).to(self.device)
        self.netDecoder1 = networks.define_DDG(opt).to(self.device)

        
        if opt['dist']:
            self.netEncoder = DistributedDataParallel(self.netEncoder, device_ids=[torch.cuda.current_device()])
            self.netDecoder1 = DistributedDataParallel(self.netDecoder1, device_ids=[torch.cuda.current_device()])

        else:
            self.netEncoder = DataParallel(self.netEncoder)
            self.netDecoder1 = DataParallel(self.netDecoder1)

        if self.is_train:
            self.netEncoder.train()
            self.netDecoder1.train()

            # loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss().to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            fix_encoder = train_opt['fix_encoder']
            # Encoder
            for k, v in self.netEncoder.named_parameters():  # can optimize for a part of the model
                if fix_encoder == True:
                    v.requires_grad = False
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            # Decoder1
            for k, v in self.netDecoder1.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))

            if train_opt['optimizer'] == 'Adam':
                print('Optimizer: Adam')
                logger.info('Optimizer: Adam')
                self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            elif train_opt['optimizer'] == 'AdamW':
                print('Optimizer: AdamW')
                logger.info('Optimizer: AdamW')
                self.optimizer_G = torch.optim.AdamW(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            elif train_opt['lr_scheme'] == 'Warmup_cosine':
                print('LR schedule: warmup cosine!')
                for optimizer in self.optimizers:
                    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, train_opt['niter']-train_opt['warmup_iter'], eta_min=1e-6)
                    scheduler = lr_scheduler.GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=train_opt['warmup_iter'], 
                                                                        after_scheduler=scheduler_cosine)
                    self.schedulers.append(scheduler)
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

        self.Gaussian_blur = GaussianSmoothing_withPad(channels=3, kernel_size=21, sigma=2, dim=2, device=self.device)
        self.global_residual = opt['network_DDE']['GR']
        print('[global_residual: ]', self.global_residual)

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQ'].to(self.device)  # LQ
        if need_GT:
            self.var_H = data['GT'].to(self.device)  # GT
            input_ref = data['ref'] if 'ref' in data else data['GT']
            self.var_ref = input_ref.to(self.device)
            


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        
        # Pretrain Encoder
        if self.global_residual:
            self.encoded_feature, _ = self.netEncoder(self.var_L)
            self.fake_H = self.netDecoder1(self.encoded_feature)
        else:
            self.encoded_feature = self.netEncoder(self.var_L)
            self.fake_H = self.netDecoder1(self.encoded_feature)


        # test
        # print('###### DDR Encoder ######')
        # for k, v in self.netEncoder.named_parameters():
        #     print(k, v.requires_grad)

        # print('###### DDR Generator ######')
        # for k, v in self.netGenerator.named_parameters():
        #     print(k, v.requires_grad)
        # exit()

        l_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
        
        l_pix.backward()
        self.optimizer_G.step()

        # set log
        self.log_dict['l_pix'] = l_pix.item()


    def test(self):
        self.netEncoder.eval()
        self.netDecoder1.eval()

        
        #print(self.var_input.shape)
        with torch.no_grad():
            if self.global_residual:
                self.encoded_feature, self.encoded_feature_res  = self.netEncoder(self.var_L)
                self.fake_D1 = self.netDecoder1(self.encoded_feature)
            else:
                self.encoded_feature = self.netEncoder(self.var_L)
                self.fake_D1 = self.netDecoder1(self.encoded_feature)
        
        self.netEncoder.train()
        self.netDecoder1.train()

    def test_Uformer(self):
        self.netEncoder.eval()
        self.netDecoder1.eval()

        
        #print(self.var_input.shape)
        B, C, H, W = self.var_L.shape
        self.var_L, mask = expand2square(self.var_L, factor=128)
        with torch.no_grad():
            if self.global_residual:
                self.encoded_feature, self.encoded_feature_res  = self.netEncoder(self.var_L)
                self.fake_D1 = self.netDecoder1(self.encoded_feature)
            else:
                self.encoded_feature = self.netEncoder(self.var_L)
                self.fake_D1 = self.netDecoder1(self.encoded_feature)
                
            self.fake_D1 = torch.masked_select(self.fake_D1, mask.bool()).reshape(1,3,H,W)
        
        self.netEncoder.train()
        self.netDecoder1.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        
        if self.is_train:
            out_dict['input'] = self.var_L.detach()[0].float().cpu()
            out_dict['SR'] = self.fake_D1.detach()[0].float().cpu()
            
        else:
            out_dict['input'] = self.var_L.detach()[0].float().cpu()
            out_dict['SR'] = self.fake_D1.detach()[0].float().cpu()
            if self.global_residual:
                out_dict['FEA_res_final'] = self.encoded_feature.detach()[0].float().cpu()
                out_dict['FEA_res'] = self.encoded_feature_res.detach()[0].float().cpu()
            else:
                out_dict['FEA'] = self.encoded_feature.detach()[0].float().cpu()

        if need_GT:
            out_dict['GT'] = self.var_H.detach()[0].float().cpu()
            #out_dict['GT_target'] = self.var_target.detach()[0].float().cpu()
        
            
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netDecoder1)
        if isinstance(self.netDecoder1, nn.DataParallel) or isinstance(self.netDecoder1, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netDecoder1.__class__.__name__,
                                             self.netDecoder1.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netDecoder1.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network netGenerator structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_Encoder']
        if load_path_G is not None:
            logger.info('Loading model for Encoder [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netEncoder, self.opt['path']['strict_load'])
        
        load_path_G = self.opt['path']['pretrain_model_Decoder1']
        if load_path_G is not None:
            logger.info('Loading model for Decoder1 [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netDecoder1, self.opt['path']['strict_load'])
        

    def save(self, iter_step):
        self.save_network(self.netEncoder, 'E', iter_step)
        self.save_network(self.netDecoder1, 'De1', iter_step)
