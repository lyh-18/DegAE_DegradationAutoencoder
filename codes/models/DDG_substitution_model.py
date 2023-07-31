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

import random

logger = logging.getLogger('base')


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
        self.netDecoder2 = networks.define_DDG(opt).to(self.device)
        self.netDecoder3 = networks.define_DDG(opt).to(self.device)
        self.netDecoder4 = networks.define_DDG(opt).to(self.device)
        
        if opt['dist']:
            self.netEncoder = DistributedDataParallel(self.netEncoder, device_ids=[torch.cuda.current_device()])
            self.netDecoder1 = DistributedDataParallel(self.netDecoder1, device_ids=[torch.cuda.current_device()])
            self.netDecoder2 = DistributedDataParallel(self.netDecoder2, device_ids=[torch.cuda.current_device()])
            self.netDecoder3 = DistributedDataParallel(self.netDecoder3, device_ids=[torch.cuda.current_device()])
            self.netDecoder4 = DistributedDataParallel(self.netDecoder4, device_ids=[torch.cuda.current_device()])
        else:
            self.netEncoder = DataParallel(self.netEncoder)
            self.netDecoder1 = DataParallel(self.netDecoder1)
            self.netDecoder2 = DataParallel(self.netDecoder2)
            self.netDecoder3 = DataParallel(self.netDecoder3)
            self.netDecoder4 = DataParallel(self.netDecoder4)
        if self.is_train:
            self.netD_DDG = networks.define_D_DDG(opt).to(self.device)
            if opt['dist']:
                self.netD_DDG = DistributedDataParallel(self.netD_DDG,
                                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD_DDG = DataParallel(self.netD_DDG)

            self.netEncoder.train()
            self.netDecoder1.train()
            self.netDecoder2.train()
            self.netDecoder3.train()
            self.netDecoder4.train()
            self.netD_DDG.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                elif l_pix_type == 'l2_blur':
                    self.cri_pix = MSE_blur_loss(channels=3, kernel_size=5, sigma=1, dim=2).to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None
            
            # dde loss    
            if train_opt['dde_weight'] > 0:
                self.cri_dde = nn.MSELoss().to(self.device)
                self.l_dde_w = train_opt['dde_weight']
            else:
                logger.info('Remove dde loss.')
                self.cri_dde = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                logger.info('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                if opt['dist']:
                    self.netF = DistributedDataParallel(self.netF,
                                                        device_ids=[torch.cuda.current_device()])
                else:
                    self.netF = DataParallel(self.netF)

            # GD gan loss
            if train_opt['patch_gan']:
                print('Using PatchGAN Loss.')
                self.cri_gan = PatchGANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            else:
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            # optimizers
            # Encoder and Decoder
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            # Encoder
            for k, v in self.netEncoder.named_parameters():  # can optimize for a part of the model
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
            # Decoder2
            for k, v in self.netDecoder2.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            # Decoder3
            for k, v in self.netDecoder3.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            # Decoder4
            for k, v in self.netDecoder4.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))


            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
            self.optimizers.append(self.optimizer_G)
            
            
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD_DDG.parameters(), lr=train_opt['lr_D'],
                                                weight_decay=wd_D,
                                                betas=(train_opt['beta1_D'], train_opt['beta2_D']))
            self.optimizers.append(self.optimizer_D)

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
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

        self.Gaussian_blur = GaussianSmoothing_withPad(channels=3, kernel_size=21, sigma=2, dim=2, device=self.device)


    def feed_data(self, data, need_GT=True, syn_deg_type='random'):
        if self.is_train:

            self.var_HQ = data['HQ'].to(self.device)  # HQ
            # randomly choose degradation on input
            self.deg_choose_input = random.choice(['clean','noise','blur','blur_noise'])
            #print('111', deg_choose_input)
            if self.deg_choose_input == 'clean':
                self.var_input = self.var_HQ.clone()
            elif self.deg_choose_input == 'noise':
                self.var_input = add_gaussian_noise_pt(self.var_HQ.clone(),sigma=20)
            elif self.deg_choose_input == 'blur':
                self.var_input = self.Gaussian_blur(self.var_HQ.clone())
            elif self.deg_choose_input == 'blur_noise':
                self.var_input = self.Gaussian_blur(self.var_HQ.clone())
                self.var_input = add_gaussian_noise_pt(self.var_input,sigma=20)

            # randomly choose degradation on target
            self.deg_choose_target = random.choice(['clean','noise','blur','blur_noise'])
            #print('222', self.deg_choose_target)
            if self.deg_choose_target == 'clean':
                self.var_target = self.var_HQ.clone()
            elif self.deg_choose_target == 'noise':
                self.var_target = add_gaussian_noise_pt(self.var_HQ.clone(),sigma=20)
            elif self.deg_choose_target == 'blur':
                self.var_target = self.Gaussian_blur(self.var_HQ.clone())
            elif self.deg_choose_target == 'blur_noise':
                self.var_target = self.Gaussian_blur(self.var_HQ.clone())
                self.var_target = add_gaussian_noise_pt(self.var_target,sigma=20)

            self.var_ref1 = self.var_target
            
        else:
            self.var_HQ = data['HQ'].to(self.device)  # HQ

            # add degradation on input
            if syn_deg_type == 'random':
                deg_choose_input = random.choice(['clean','noise','blur','blur_noise'])
            else:
                deg_choose_input = None

            if deg_choose_input == 'clean':
                self.var_input = self.var_HQ.clone()
            elif deg_choose_input == 'noise':
                self.var_input = add_gaussian_noise_pt(self.var_HQ.clone(),sigma=20)
            elif deg_choose_input == 'blur':
                self.var_input = filter2D(self.var_HQ.clone(), blur_kernel)
            elif deg_choose_input == 'blur_noise':
                blur_kernel = random_bivariate_Gaussian(kernel_size=15,sigma_x_range=[2.0,2.0],isotropic=True)
                self.var_input = filter2D(self.var_HQ.clone(), blur_kernel)
                self.var_input = add_gaussian_noise_pt(self.var_input,sigma=20)
            elif deg_choose_input == None:
                self.var_input = self.var_HQ.clone()
            


    def optimize_parameters(self, step):
        # G
        for p in self.netD_DDG.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        
        # Pretrain Encoder
        self.encoded_feature = self.netEncoder(self.var_input)

        # Deep degradation substitution
        for p in self.netDecoder1.parameters():
            p.requires_grad = False
        for p in self.netDecoder2.parameters():
            p.requires_grad = False
        for p in self.netDecoder3.parameters():
            p.requires_grad = False
        for p in self.netDecoder4.parameters():
            p.requires_grad = False

        if self.deg_choose_target == 'clean':
            for p in self.netDecoder1.parameters():
                p.requires_grad = True
            self.fake_L1 = self.netDecoder1(self.encoded_feature)
        elif self.deg_choose_target == 'noise':
            for p in self.netDecoder2.parameters():
                p.requires_grad = True
            self.fake_L1 = self.netDecoder2(self.encoded_feature)
        elif self.deg_choose_target == 'blur':
            for p in self.netDecoder3.parameters():
                p.requires_grad = True
            self.fake_L1 = self.netDecoder3(self.encoded_feature)
        elif self.deg_choose_target == 'blur_noise':
            for p in self.netDecoder4.parameters():
                p.requires_grad = True
            self.fake_L1 = self.netDecoder4(self.encoded_feature)
        else:
            print('Wrong!')
            exit()

        # test
        # print('###### DDR Encoder ######')
        # for k, v in self.netEncoder.named_parameters():
        #     print(k, v.requires_grad)

        # print('###### DDR Generator ######')
        # for k, v in self.netGenerator.named_parameters():
        #     print(k, v.requires_grad)
        # exit()

        l_g_total = 0
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                l_g_pix = self.l_pix_w * (self.cri_pix(self.fake_L1, self.var_target))
                l_g_total += l_g_pix
                
            # if self.cri_dde: # deep degradation consistent loss
            #     l_g_dde = self.l_dde_w * self.cri_dde(self.embedding1_ref, self.embedding2_ref)
            #     l_g_total += l_g_dde
                
            if self.cri_fea:  # feature loss
                real_fea1 = self.netF(self.var_target).detach()
                fake_fea1 = self.netF(self.fake_L1)
                # real_fea2 = self.netF(self.var_L2).detach()
                # fake_fea2 = self.netF(self.fake_L2)
                l_g_fea = self.l_fea_w * (self.cri_fea(fake_fea1, real_fea1))
                l_g_total += l_g_fea

            pred_g_fake1 = self.netD_DDG(self.fake_L1)
            #pred_g_fake2 = self.netD_DDG(self.fake_L2)
            if self.opt['train']['gan_type'] in ['gan', 'lsgan']:
                l_g_gan1 = self.l_gan_w * self.cri_gan(pred_g_fake1, True)
                #l_g_gan2 = self.l_gan_w * self.cri_gan(pred_g_fake2, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real1 = self.netD_DDG(self.var_ref1).detach()
                #pred_d_real2 = self.netD_DDG(self.var_ref2).detach()
                l_g_gan1 = self.l_gan_w * (self.cri_gan(pred_d_real1 - torch.mean(pred_g_fake1), False)+self.cri_gan(pred_g_fake1 - torch.mean(pred_d_real1), True)) / 2
                #l_g_gan2 = self.l_gan_w * (self.cri_gan(pred_d_real2 - torch.mean(pred_g_fake2), False)+self.cri_gan(pred_g_fake2 - torch.mean(pred_d_real2), True)) / 2
            l_g_total = l_g_total + l_g_gan1 #+ l_g_gan2

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD_DDG.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real1 = self.netD_DDG(self.var_ref1)
        pred_d_fake1 = self.netD_DDG(self.fake_L1.detach())  # detach to avoid BP to G
        
        #pred_d_real2 = self.netD_DDG(self.var_ref2)
        #pred_d_fake2 = self.netD_DDG(self.fake_L2.detach())  # detach to avoid BP to G
        
        if self.opt['train']['gan_type'] in ['gan', 'lsgan']:
            l_d_real1 = self.cri_gan(pred_d_real1, True)
            l_d_fake1 = self.cri_gan(pred_d_fake1, False)
            #l_d_real2 = self.cri_gan(pred_d_real2, True)
            #l_d_fake2 = self.cri_gan(pred_d_fake2, False)
            
            l_d_total = l_d_real1 + l_d_fake1 #+ l_d_real2 + l_d_fake2
        elif self.opt['train']['gan_type'] == 'ragan':
            l_d_real1 = self.cri_gan(pred_d_real1 - torch.mean(pred_d_fake1), True)
            l_d_fake1 = self.cri_gan(pred_d_fake1 - torch.mean(pred_d_real1), False)
            #l_d_real2 = self.cri_gan(pred_d_real2 - torch.mean(pred_d_fake2), True)
            #l_d_fake2 = self.cri_gan(pred_d_fake2 - torch.mean(pred_d_real2), False)
            
            #l_d_total = (l_d_real1 + l_d_fake1 + l_d_real2 + l_d_fake2) / 2
            l_d_total = (l_d_real1 + l_d_fake1) / 2

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            # if self.cri_dde:
            #     self.log_dict['l_g_dde'] = l_g_dde.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan1'] = l_g_gan1.item()
            #self.log_dict['l_g_gan2'] = l_g_gan2.item()

        self.log_dict['l_d_real1'] = l_d_real1.item()
        self.log_dict['l_d_fake1'] = l_d_fake1.item()
        #self.log_dict['l_d_real2'] = l_d_real2.item()
        #self.log_dict['l_d_fake2'] = l_d_fake2.item()
        self.log_dict['D_real1'] = torch.mean(pred_d_real1.detach())
        self.log_dict['D_fake1'] = torch.mean(pred_d_fake1.detach())
        #self.log_dict['D_real2'] = torch.mean(pred_d_real2.detach())
        #self.log_dict['D_fake2'] = torch.mean(pred_d_fake2.detach())

    def test(self):
        self.netEncoder.eval()
        self.netDecoder1.eval()
        self.netDecoder2.eval()
        self.netDecoder3.eval()
        self.netDecoder4.eval()
        
        #print(self.var_input.shape)
        with torch.no_grad():
            self.encoded_feature = self.netEncoder(self.var_input)
            self.fake_D1 = self.netDecoder1(self.encoded_feature)
            self.fake_D2 = self.netDecoder2(self.encoded_feature)
            self.fake_D3 = self.netDecoder3(self.encoded_feature)
            self.fake_D4 = self.netDecoder4(self.encoded_feature)
        
        self.netEncoder.train()
        self.netDecoder1.train()
        self.netDecoder2.train()
        self.netDecoder3.train()
        self.netDecoder4.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        
        if self.is_train:
            out_dict['input'] = self.var_input.detach()[0].float().cpu()
            out_dict['output_clean'] = self.fake_D1.detach()[0].float().cpu()
            out_dict['output_noise'] = self.fake_D2.detach()[0].float().cpu()
            out_dict['output_blur'] = self.fake_D3.detach()[0].float().cpu()
            out_dict['output_blur_noise'] = self.fake_D4.detach()[0].float().cpu()
            
            out_dict['GT_target'] = self.var_target.detach()[0].float().cpu()
            out_dict['GT_deg'] = self.deg_choose_target
            out_dict['input_deg'] = self.deg_choose_input
            
        else:
            out_dict['input'] = self.var_input.detach()[0].float().cpu()
            out_dict['output_clean'] = self.fake_D1.detach()[0].float().cpu()
            out_dict['output_noise'] = self.fake_D2.detach()[0].float().cpu()
            out_dict['output_blur'] = self.fake_D3.detach()[0].float().cpu()
            out_dict['output_blur_noise'] = self.fake_D4.detach()[0].float().cpu()
            out_dict['FEA'] = self.encoded_feature.detach()[0].float().cpu()
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
        if self.is_train:
            # Discriminator
            s, n = self.get_network_description(self.netD_DDG)
            if isinstance(self.netD_DDG, nn.DataParallel) or isinstance(self.netD_DDG,
                                                                    DistributedDataParallel):
                net_struc_str = '{} - {}'.format(self.netD_DDG.__class__.__name__,
                                                 self.netD_DDG.module.__class__.__name__)
            else:
                net_struc_str = '{}'.format(self.netD_DDG.__class__.__name__)
            if self.rank <= 0:
                logger.info('Network D structure: {}, with parameters: {:,d}'.format(
                    net_struc_str, n))
                logger.info(s)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                if isinstance(self.netF, nn.DataParallel) or isinstance(
                        self.netF, DistributedDataParallel):
                    net_struc_str = '{} - {}'.format(self.netF.__class__.__name__,
                                                     self.netF.module.__class__.__name__)
                else:
                    net_struc_str = '{}'.format(self.netF.__class__.__name__)
                if self.rank <= 0:
                    logger.info('Network F structure: {}, with parameters: {:,d}'.format(
                        net_struc_str, n))
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

        load_path_G = self.opt['path']['pretrain_model_Decoder2']
        if load_path_G is not None:
            logger.info('Loading model for Decoder2 [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netDecoder2, self.opt['path']['strict_load'])

        load_path_G = self.opt['path']['pretrain_model_Decoder3']
        if load_path_G is not None:
            logger.info('Loading model for Decoder3 [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netDecoder3, self.opt['path']['strict_load'])

        load_path_G = self.opt['path']['pretrain_model_Decoder4']
        if load_path_G is not None:
            logger.info('Loading model for Decoder4 [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netDecoder4, self.opt['path']['strict_load'])
        
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for netD_DDG [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD_DDG, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netEncoder, 'E', iter_step)
        self.save_network(self.netDecoder1, 'De1', iter_step)
        self.save_network(self.netDecoder2, 'De2', iter_step)
        self.save_network(self.netDecoder3, 'De3', iter_step)
        self.save_network(self.netDecoder4, 'De4', iter_step)
        self.save_network(self.netD_DDG, 'D', iter_step)
