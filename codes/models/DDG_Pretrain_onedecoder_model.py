import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks_DDG as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss, PatchGANLoss, MSE_blur_loss

import math

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
        self.netDDG_Encoder = networks.define_DDE_Encoder(opt).to(self.device)
        self.netPretrainEncoder = networks.define_DDE(opt).to(self.device)
        self.netPretrainDecoder = networks.define_DDG(opt).to(self.device)
        
        if opt['dist']:
            self.netDDG_Encoder = DistributedDataParallel(self.netDDG_Encoder, device_ids=[torch.cuda.current_device()])
            self.netPretrainEncoder = DistributedDataParallel(self.netPretrainEncoder, device_ids=[torch.cuda.current_device()])
            self.netPretrainDecoder = DistributedDataParallel(self.netPretrainDecoder, device_ids=[torch.cuda.current_device()])
        else:
            self.netDDG_Encoder = DataParallel(self.netDDG_Encoder)
            self.netPretrainEncoder = DataParallel(self.netPretrainEncoder)
            self.netPretrainDecoder = DataParallel(self.netPretrainDecoder)
        if self.is_train:
            self.netD_DDG = networks.define_D_DDG(opt).to(self.device)
            if opt['dist']:
                self.netD_DDG = DistributedDataParallel(self.netD_DDG,
                                                    device_ids=[torch.cuda.current_device()])
            else:
                self.netD_DDG = DataParallel(self.netD_DDG)

            self.netDDG_Encoder.train()
            self.netPretrainEncoder.train()
            self.netPretrainDecoder.train()
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
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netDDG_Encoder.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            for k, v in self.netPretrainEncoder.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            for k, v in self.netPretrainDecoder.named_parameters():  # can optimize for a part of the model
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

        #self.print_network()  # print network
        self.load()  # load G and D if needed
        self.global_residual = opt['network_DDE']['GR']
        print('[global_residual: ]', self.global_residual)

    def feed_data(self, data, need_GT=True):
        if self.is_train:
            self.var_L1 = data['LQ1'].to(self.device)  # LQ1
            self.var_L2 = data['LQ2'].to(self.device)  # LQ2

            self.var_H1 = data['HQ1'].to(self.device)  # HQ1
            self.var_H2 = data['HQ2'].to(self.device)  # HQ2
            
            input_ref1 = data['HQ1']
            input_ref2 = data['HQ2']
            self.var_ref1 = input_ref1.to(self.device)
            self.var_ref2 = input_ref2.to(self.device)

            self.input_deg = data['input_deg_type']
            self.target_deg = data['target_deg_type']
            
        else:
            self.input_image = data['HQ1'].to(self.device) 
            self.ref_image = data['LQ2'].to(self.device)
            self.var_GTLQ = data['LQ1'].to(self.device)

            self.input_deg = data['input_deg_type']
            self.target_deg = data['target_deg_type']


    def optimize_parameters(self, step):
        # G
        for p in self.netD_DDG.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()
        
        # GT degradation embeddings
        self.embedding1_ref = self.netDDG_Encoder(self.var_L1)
        self.embedding2_ref = self.netDDG_Encoder(self.var_L2)
        
        # Deep degradation encoder
        if self.global_residual:
            self.H1_feature, _ = self.netPretrainEncoder(self.var_H1)
            self.H2_feature, _ = self.netPretrainEncoder(self.var_H2)
        else:
            self.H1_feature = self.netPretrainEncoder(self.var_H1)
            self.H2_feature = self.netPretrainEncoder(self.var_H2)

        self.fake_L1 = self.netPretrainDecoder(self.H1_feature, self.embedding2_ref)
        self.fake_L2 = self.netPretrainDecoder(self.H2_feature, self.embedding2_ref) # use the ref LQ2 embedding!

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
                l_g_pix = self.l_pix_w * (self.cri_pix(self.fake_L1, self.var_L1) + self.cri_pix(self.fake_L2, self.var_L2))
                l_g_total += l_g_pix
                
            if self.cri_dde: # deep degradation consistent loss
                l_g_dde = self.l_dde_w * self.cri_dde(self.embedding1_ref, self.embedding2_ref)
                l_g_total += l_g_dde
                
            if self.cri_fea:  # feature loss
                real_fea1 = self.netF(self.var_L1).detach()
                fake_fea1 = self.netF(self.fake_L1)
                real_fea2 = self.netF(self.var_L2).detach()
                fake_fea2 = self.netF(self.fake_L2)
                l_g_fea = self.l_fea_w * (self.cri_fea(fake_fea1, real_fea1) + self.cri_fea(fake_fea2, real_fea2))
                l_g_total += l_g_fea

            pred_g_fake1 = self.netD_DDG(self.fake_L1)
            pred_g_fake2 = self.netD_DDG(self.fake_L2)
            if self.opt['train']['gan_type'] in ['gan', 'lsgan']:
                l_g_gan1 = self.l_gan_w * self.cri_gan(pred_g_fake1, True)
                l_g_gan2 = self.l_gan_w * self.cri_gan(pred_g_fake2, True)
            elif self.opt['train']['gan_type'] == 'ragan':
                pred_d_real1 = self.netD_DDG(self.var_ref1).detach()
                pred_d_real2 = self.netD_DDG(self.var_ref2).detach()
                l_g_gan1 = self.l_gan_w * (self.cri_gan(pred_d_real1 - torch.mean(pred_g_fake1), False)+self.cri_gan(pred_g_fake1 - torch.mean(pred_d_real1), True)) / 2
                l_g_gan2 = self.l_gan_w * (self.cri_gan(pred_d_real2 - torch.mean(pred_g_fake2), False)+self.cri_gan(pred_g_fake2 - torch.mean(pred_d_real2), True)) / 2
            l_g_total = l_g_total + l_g_gan1 + l_g_gan2

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD_DDG.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real1 = self.netD_DDG(self.var_ref1)
        pred_d_fake1 = self.netD_DDG(self.fake_L1.detach())  # detach to avoid BP to G
        
        pred_d_real2 = self.netD_DDG(self.var_ref2)
        pred_d_fake2 = self.netD_DDG(self.fake_L2.detach())  # detach to avoid BP to G
        
        if self.opt['train']['gan_type'] in ['gan', 'lsgan']:
            l_d_real1 = self.cri_gan(pred_d_real1, True)
            l_d_fake1 = self.cri_gan(pred_d_fake1, False)
            l_d_real2 = self.cri_gan(pred_d_real2, True)
            l_d_fake2 = self.cri_gan(pred_d_fake2, False)
            
            l_d_total = l_d_real1 + l_d_fake1 + l_d_real2 + l_d_fake2
        elif self.opt['train']['gan_type'] == 'ragan':
            l_d_real1 = self.cri_gan(pred_d_real1 - torch.mean(pred_d_fake1), True)
            l_d_fake1 = self.cri_gan(pred_d_fake1 - torch.mean(pred_d_real1), False)
            l_d_real2 = self.cri_gan(pred_d_real2 - torch.mean(pred_d_fake2), True)
            l_d_fake2 = self.cri_gan(pred_d_fake2 - torch.mean(pred_d_real2), False)
            
            l_d_total = (l_d_real1 + l_d_fake1 + l_d_real2 + l_d_fake2) / 2

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:
                self.log_dict['l_g_pix'] = l_g_pix.item()
            if self.cri_dde:
                self.log_dict['l_g_dde'] = l_g_dde.item()
            if self.cri_fea:
                self.log_dict['l_g_fea'] = l_g_fea.item()
            self.log_dict['l_g_gan1'] = l_g_gan1.item()
            self.log_dict['l_g_gan2'] = l_g_gan2.item()

        self.log_dict['l_d_real1'] = l_d_real1.item()
        self.log_dict['l_d_fake1'] = l_d_fake1.item()
        self.log_dict['l_d_real2'] = l_d_real2.item()
        self.log_dict['l_d_fake2'] = l_d_fake2.item()
        self.log_dict['D_real1'] = torch.mean(pred_d_real1.detach())
        self.log_dict['D_fake1'] = torch.mean(pred_d_fake1.detach())
        self.log_dict['D_real2'] = torch.mean(pred_d_real2.detach())
        self.log_dict['D_fake2'] = torch.mean(pred_d_fake2.detach())

    def test(self):
        self.netDDG_Encoder.eval()
        self.netPretrainEncoder.eval()
        self.netPretrainDecoder.eval()
        
        with torch.no_grad():
            if self.is_train:
                self.embedding = self.netDDG_Encoder(self.var_L2)
                if self.global_residual:
                    self.HQ1_feature, self.HQ1_feature_res = self.netPretrainEncoder(self.var_H1)
                else:
                    self.HQ1_feature = self.netPretrainEncoder(self.var_H1)

                self.fake_L1 = self.netPretrainDecoder(self.HQ1_feature, self.embedding)            
            else:
                self.embedding = self.netDDG_Encoder(self.ref_image)
                if self.global_residual:
                    self.HQ_feature, self.HQ_feature_res = self.netPretrainEncoder(self.input_image)
                else:
                    self.HQ_feature = self.netPretrainEncoder(self.input_image)
                
                self.fake_LQ = self.netPretrainDecoder(self.HQ_feature, self.embedding)
        
        self.netDDG_Encoder.train()
        self.netPretrainEncoder.train()
        self.netPretrainDecoder.train()

    def test_Uformer(self):
        self.netDDG_Encoder.eval()
        self.netPretrainEncoder.eval()
        self.netPretrainDecoder.eval()
        
        with torch.no_grad():
            if self.is_train:
                self.embedding = self.netDDG_Encoder(self.var_L2)
                B, C, H, W = self.var_H1.shape
                #print('111', self.var_H1.shape)
                self.var_H1_pad, mask = expand2square(self.var_H1, factor=128)

                if self.global_residual:
                    self.HQ1_feature, self.HQ1_feature_res = self.netPretrainEncoder(self.var_H1_pad)
                else:
                    self.HQ1_feature = self.netPretrainEncoder(self.var_H1_pad)

                self.fake_L1 = self.netPretrainDecoder(self.HQ1_feature, self.embedding)
                self.fake_L1 = torch.masked_select(self.fake_L1, mask.bool()).reshape(1,3,H,W)
                #print('222', self.fake_L1.shape)
            else:
                self.embedding = self.netDDG_Encoder(self.ref_image)

                B, C, H, W = self.input_image.shape
                self.input_image_pad, mask = expand2square(self.input_image, factor=128)
                if self.global_residual:
                    self.HQ_feature, self.HQ_feature_res = self.netPretrainEncoder(self.input_image_pad)
                else:
                    self.HQ_feature = self.netPretrainEncoder(self.input_image_pad)
                
                self.fake_LQ = self.netPretrainDecoder(self.HQ_feature, self.embedding)

                self.fake_LQ = torch.masked_select(self.fake_LQ, mask.bool()).reshape(1,3,H,W)
        
        self.netDDG_Encoder.train()
        self.netPretrainEncoder.train()
        self.netPretrainDecoder.train()
    
    
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        
        if self.is_train:
            out_dict['L1'] = self.var_L1.detach()[0].float().cpu()
            out_dict['L2'] = self.var_L2.detach()[0].float().cpu()
            out_dict['H1'] = self.var_H1.detach()[0].float().cpu()
            out_dict['DDG_L1'] = self.fake_L1.detach()[0].float().cpu()
            out_dict['DDG_L2'] = self.fake_L2.detach()[0].float().cpu()
            out_dict['input_deg'] = self.input_deg[0]
            out_dict['target_deg'] = self.target_deg[0]
            
        else:
            out_dict['input_img'] = self.input_image.detach()[0].float().cpu()
            out_dict['ref_img'] = self.ref_image.detach()[0].float().cpu()
            out_dict['GT_LQ'] = self.var_GTLQ.detach()[0].float().cpu()
            out_dict['DDG_img'] = self.fake_LQ.detach()[0].float().cpu()
            out_dict['input_deg'] = self.input_deg[0]
            out_dict['target_deg'] = self.target_deg[0]
            

            if self.global_residual:
                out_dict['FEA_res_final'] = self.HQ_feature.detach()[0].float().cpu()
                out_dict['FEA_res'] = self.HQ_feature_res.detach()[0].float().cpu()
            else:
                out_dict['FEA'] = self.HQ_feature.detach()[0].float().cpu()
        
            
        return out_dict

    def print_network(self):
        # netDDG_Encoder
        s, n = self.get_network_description(self.netDDG_Encoder)
        if isinstance(self.netDDG_Encoder, nn.DataParallel) or isinstance(self.netDDG_Encoder, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netDDG_Encoder.__class__.__name__,
                                             self.netDDG_Encoder.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netDDG_Encoder.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network netDDG_Encoder structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
        
        # netPretrainEncoder
        s, n = self.get_network_description(self.netPretrainEncoder)
        if isinstance(self.netPretrainEncoder, nn.DataParallel) or isinstance(self.netPretrainEncoder, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netPretrainEncoder.__class__.__name__,
                                             self.netPretrainEncoder.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netPretrainEncoder.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network netPretrainEncoder structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

        # netPretrainDecoder
        s, n = self.get_network_description(self.netPretrainDecoder)
        if isinstance(self.netPretrainDecoder, nn.DataParallel) or isinstance(self.netPretrainDecoder, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netPretrainDecoder.__class__.__name__,
                                             self.netPretrainDecoder.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netPretrainDecoder.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network netPretrainDecoder structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
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
        load_path_G = self.opt['path']['pretrain_model_DDG_Encoder']
        if load_path_G is not None:
            logger.info('Loading model for netDDG_Encoder [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netDDG_Encoder, self.opt['path']['strict_load'])
        
        load_path_G = self.opt['path']['pretrain_model_PretrainEncoder']
        if load_path_G is not None:
            logger.info('Loading model for netPretrainEncoder [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netPretrainEncoder, self.opt['path']['strict_load'])

        load_path_G = self.opt['path']['pretrain_model_PretrainDecoder']
        if load_path_G is not None:
            logger.info('Loading model for netPretrainDecoder [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netPretrainDecoder, self.opt['path']['strict_load'])
        
        load_path_D = self.opt['path']['pretrain_model_D']
        if self.opt['is_train'] and load_path_D is not None:
            logger.info('Loading model for netD_DDG [{:s}] ...'.format(load_path_D))
            self.load_network(load_path_D, self.netD_DDG, self.opt['path']['strict_load'])

    def save(self, iter_step):
        self.save_network(self.netDDG_Encoder, 'DDG_Encoder', iter_step)
        self.save_network(self.netPretrainEncoder, 'PretrainEncoder', iter_step)
        self.save_network(self.netPretrainDecoder, 'PretrainDecoder', iter_step)
        self.save_network(self.netD_DDG, 'D', iter_step)
