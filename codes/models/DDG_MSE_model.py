import logging
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks_DDG as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import GANLoss

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
        self.netGenerator = networks.define_DDG(opt).to(self.device)
        
        if opt['dist']:
            self.netEncoder = DistributedDataParallel(self.netEncoder, device_ids=[torch.cuda.current_device()])
            self.netGenerator = DistributedDataParallel(self.netGenerator, device_ids=[torch.cuda.current_device()])
        else:
            self.netEncoder = DataParallel(self.netEncoder)
            self.netGenerator = DataParallel(self.netGenerator)
        
        if self.is_train:
            self.netEncoder.train()
            self.netGenerator.train()

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                logger.info('Remove pixel loss.')
                self.cri_pix = None
            
            # dde loss             
            if train_opt['dde_weight'] >= 0:
                self.cri_dde = nn.MSELoss().to(self.device)
                self.l_dde_w = train_opt['dde_weight']
            else:
                logger.info('Remove dde loss.')
                self.cri_dde = None
                  


            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netEncoder.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            for k, v in self.netGenerator.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1_G'], train_opt['beta2_G']))
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
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

        self.print_network()  # print network
        self.load()  # load G and D if needed

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
            
        else:
            self.var_LQ = data['ref_LQ'].to(self.device)  # ref_LQ
            self.var_HQ = data['ref_HQ'].to(self.device)  # ref_HQ
            self.var_GTLQ = data['GT_LQ'].to(self.device)  # ref_HQ
            


    def optimize_parameters(self, step):
        self.optimizer_G.zero_grad()
        
        # GT degradation embeddings
        self.embedding1_ref = self.netEncoder(self.var_L1)
        self.embedding2_ref = self.netEncoder(self.var_L2)
        
        # Deep degradation generation
        self.fake_L1 = self.netGenerator(self.var_H1, self.embedding1_ref)
        self.fake_L2 = self.netGenerator(self.var_H2, self.embedding1_ref)

        l_g_total = 0
        if self.cri_pix:  # pixel loss
            l_g_pix = self.l_pix_w * (self.cri_pix(self.fake_L1, self.var_L1) + self.cri_pix(self.fake_L2, self.var_L2))
            l_g_total += l_g_pix
        
         
        # deep degradation consistent loss
        l_g_dde = self.cri_dde(self.embedding1_ref, self.embedding2_ref)               
        l_g_total += self.l_dde_w * l_g_dde
 
                


        l_g_total.backward()
        self.optimizer_G.step()



        # set log
        self.log_dict['l_g_pix'] = l_g_pix.item()
        self.log_dict['l_g_dde'] = l_g_dde.item()
            

    def test(self):
        self.netEncoder.eval()
        self.netGenerator.eval()
        
        with torch.no_grad():
            if self.is_train:
                self.embedding = self.netEncoder(self.var_L1)
                self.fake_L1 = self.netGenerator(self.var_H1, self.embedding)            
            else:
                self.embedding = self.netEncoder(self.var_LQ)
                self.fake_LQ = self.netGenerator(self.var_HQ, self.embedding)
        
        self.netEncoder.train()
        self.netGenerator.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        
        if self.is_train:
            out_dict['L1'] = self.var_L1.detach()[0].float().cpu()
            out_dict['L2'] = self.var_L2.detach()[0].float().cpu()
            out_dict['DDG_L1'] = self.fake_L1.detach()[0].float().cpu()
            out_dict['DDG_L2'] = self.fake_L2.detach()[0].float().cpu()
            
        else:
            out_dict['LQ'] = self.var_LQ.detach()[0].float().cpu()
            out_dict['DDG_LQ'] = self.fake_LQ.detach()[0].float().cpu()
            out_dict['GT_LQ'] = self.var_GTLQ.detach()[0].float().cpu()
            
        
            
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netGenerator)
        if isinstance(self.netGenerator, nn.DataParallel) or isinstance(self.netGenerator, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netGenerator.__class__.__name__,
                                             self.netGenerator.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netGenerator.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network netGenerator structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)
            

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_Generator']
        if load_path_G is not None:
            logger.info('Loading model for Generator [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netGenerator, self.opt['path']['strict_load'])
        
        load_path_G = self.opt['path']['pretrain_model_Encoder']
        if load_path_G is not None:
            logger.info('Loading model for Encoder [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netEncoder, self.opt['path']['strict_load'])
        

    def save(self, iter_step):
        self.save_network(self.netGenerator, 'G', iter_step)
        self.save_network(self.netEncoder, 'E', iter_step)
