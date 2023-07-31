import torch
import logging

import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.DDG_arch as DDG_arch
import models.modules.SwinIR_arch as SwinIR_arch
import models.modules.FFANet_arch as FFANet_arch
import models.modules.Uformer_arch as Uformer_arch
import models.modules.EDT_arch as EDT_arch
import models.modules.Restormer_arch as Restormer_arch

logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_noGR':
        netG = SRResNet_arch.MSRResNet_noGR(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_noGR_fea':
        netG = SRResNet_arch.MSRResNet_noGR_fea(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], rfea_layer=opt_net['rfea_layer'], upscale=opt_net['scale'])
    elif which_model == 'FFANet':
        netG = FFANet_arch.FFA(gps=opt_net['gps'],blocks=opt_net['blocks'])

    elif which_model == 'MSRResNet_wGR_i':
        netG = SRResNet_arch.MSRResNet_wGR_i(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'MSRResNet_wGR_i_fea':
        netG = SRResNet_arch.MSRResNet_wGR_i_fea(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], rfea_layer=opt_net['rfea_layer'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'])
    elif which_model == 'RRDBNet_wGR_i':
        netG = RRDBNet_arch.RRDBNet_wGR_i(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'])
    elif which_model == 'RRDBNet_wGR_i_dropoutlast_channel05':
        netG = RRDBNet_arch.RRDBNet_wGR_i_dropoutlast_channel05(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'])
    elif which_model == 'RRDBNet_dropoutlast_channel05':
        netG = RRDBNet_arch.RRDBNet_dropoutlast_channel05(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'])
    elif which_model == 'RRDBNet_wGR_i_fea':
        netG = RRDBNet_arch.RRDBNet_wGR_i_fea(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'])
    elif which_model == 'RRDBNet_wGR_i_modulation':
        netG = RRDBNet_arch.RRDBNet_wGR_i_modulation(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], modul_method=opt_net['modul_method'])
    elif which_model == 'RRDBNet_wGR_i_modulation_fea':
        netG = RRDBNet_arch.RRDBNet_wGR_i_modulation_fea(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], scale=opt_net['upscale'], modul_method=opt_net['modul_method'])
    elif which_model == 'AttentionNet':
        netG = SRResNet_arch.AttentionNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])
    elif which_model == 'RDANet':
        netG = RDANet_arch.RDANet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'])       

    elif which_model == 'Uformer_B':
        netG = Uformer_arch.Uformer(in_chans=opt_net['in_nc'], img_size=opt_net['img_size'], embed_dim=32, win_size=8, token_projection='linear', token_mlp='leff',
                                    depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], modulator=True, dd_in=opt_net['dd_in'])
    
    elif which_model == 'Restormer':
        netG = Restormer_arch.Restormer(
        inp_channels = opt_net['inp_channels'], 
        out_channels = opt_net['out_channels'], 
        dim = opt_net['dim'],
        num_blocks = opt_net['num_blocks'], 
        num_refinement_blocks = opt_net['num_refinement_blocks'],
        heads = opt_net['heads'],
        ffn_expansion_factor = opt_net['ffn_expansion_factor'],
        bias = opt_net['bias'],
        global_residual = opt_net['global_residual'],LayerNorm_type = opt_net['LayerNorm_type'],   ## Other option 'BiasFree'
        dual_pixel_task = opt_net['dual_pixel_task']        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        )
    
    elif which_model == 'EDT_B':
        from easydict import EasyDict as edict
        res=opt_net['img_size']
        class Config:
            MODEL = edict()
            MODEL.IN_CHANNEL = opt_net['in_nc']
            MODEL.DEPTH = 2
            MODEL.IMAGE_SIZE = res // (2 ** MODEL.DEPTH)
            MODEL.SCALES = []
            MODEL.NOISE_LEVELS = [1]
            MODEL.RAIN_LEVELS = []
            MODEL.WINDOW_SIZE = (6, 24)
            MODEL.IMAGE_RANGE = 1.0
            MODEL.NUM_FEAT = 32
            MODEL.DEPTHS = [6, 6, 6, 6, 6, 6]
            MODEL.EMBED_DIM = 180
            MODEL.NUM_HEADS = [6, 6, 6, 6, 6, 6]
            MODEL.MLP_RATIO = 2
            MODEL.UPSAMPLER = 'pixelshuffle'
            #MODEL.UPSAMPLER = 'pixelshuffledirect'
            MODEL.RESI_CONNECTION = '1conv'

        config = Config()
        netG = EDT_arch.Network(config)

    elif which_model == 'DASR':
        netG = DASR_arch.DASR()                                 

    elif which_model == 'RCAN':
        netG = RCAN_arch.RCAN()
        
    elif which_model == 'SwinIR':
        netG = SwinIR_arch.SwinIR(upscale=opt_net['upscale'], 
                   window_size=opt_net['window_size'], img_range=1., depths=opt_net['depths'],
                   embed_dim=opt_net['nf'], num_heads=opt_net['num_heads'], mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsampler'], resi_connection=opt_net['resi_connection'],
                   global_residual=opt_net['GR'], return_fea=opt_net['rfea_layer'])
                   

                                    
    elif which_model == 'TBSRGAN':
        netG = TBSRGAN_arch.TBSRGAN(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], 
                                    LF_nf=opt_net['LF_nf'], LF_nb=opt_net['LF_nb'], 
                                    HF_nf=opt_net['HF_nf'], HF_nb=opt_net['HF_nb'], upscale=opt_net['scale'], concat=opt_net['concat'])
    elif which_model == 'TBSRGAN_SHARE':
        netG = TBSRGAN_arch.TBSRGAN_SHARE(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], 
                                    LF_nf=opt_net['LF_nf'], LF_nb=opt_net['LF_nb'], 
                                    HF_nf=opt_net['HF_nf'], HF_nb=opt_net['HF_nb'], upscale=opt_net['scale'], concat=opt_net['concat'])
    elif which_model == 'TBSRGAN_CROSS_G2':
        netG = TBSRGAN_arch.TBSRGAN_CROSS_G2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], 
                                    LF_nf=opt_net['LF_nf'], LF_nb=opt_net['LF_nb'], 
                                    HF_nf=opt_net['HF_nf'], HF_nb=opt_net['HF_nb'], upscale=opt_net['scale'], concat=opt_net['concat'])
    elif which_model == 'TBSRGAN_CROSS_G4':
        netG = TBSRGAN_arch.TBSRGAN_CROSS_G4(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], 
                                    LF_nf=opt_net['LF_nf'], LF_nb=opt_net['LF_nb'], 
                                    HF_nf=opt_net['HF_nf'], HF_nb=opt_net['HF_nb'], upscale=opt_net['scale'], concat=opt_net['concat'])
                                  
    
    # elif which_model == 'sft_arch':  # SFT-GAN
    #     netG = sft_arch.SFT_Net()
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif which_model == 'NLayerDiscriminator':
        netD = DDG_arch.NLayerDiscriminator()
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


#### Define Network used for Perceptual Loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # PyTorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,
                                          use_input_norm=True, device=device)
    netF.eval()  # No need to train
    return netF
