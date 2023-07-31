import torch
import logging
import models.modules.DDG_arch as DDG_arch
import models.modules.SwinIR_arch as SwinIR_arch
import models.modules.SwinIR_pretrain_arch as SwinIR_pretrain_arch
import models.modules.discriminator_vgg_arch as SRGAN_arch
import models.modules.Uformer_pretrain_arch as Uformer_pretrain_arch

import models.modules.SwinIR_Backbone_arch as SwinIR_Backbone_arch
import models.modules.Uformer_Backbone_arch as Uformer_Backbone_arch
import models.modules.RCAN_Pretrain_Head_arch as RCAN_Pretrain_Head_arch

#import models.modules.DehazeFormer_Backbone as DehazeFormer_Backbone
import models.modules.Restormer_Backbone_arch as Restormer_Backbone_arch

logger = logging.getLogger('base')


####################
# define network
####################

#### Deep Degradation Encoder
def define_DDE(opt):
    opt_net = opt['network_DDE']
    which_model = opt_net['which_model_DDE']

    if which_model == 'DeepDegradationEncoder_v1':
        netG = DDG_arch.DeepDegradationEncoder_v1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'DeepDegradationEncoder_v2':
        netG = DDG_arch.DeepDegradationEncoder_v2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'], checkpoint=opt_net['pretrained_DDR_model'])
    elif which_model == 'DeepDegradationEncoder_v3':
        netG = DDG_arch.DeepDegradationEncoder_v3(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'ConditionNet_GFM':
        netG = DDG_arch.ConditionNet_GFM()
    elif which_model == 'SwinIR_Encoder':
        netG = SwinIR_pretrain_arch.SwinIR_Encoder(window_size=opt_net['window_size'], img_range=1., depths=opt_net['depths'],
                   embed_dim=opt_net['nf'], num_heads=opt_net['num_heads'], mlp_ratio=opt_net['mlp_ratio'],
                   resi_connection=opt_net['resi_connection'], 
                   global_residual=opt_net['GR']) 

    elif which_model == 'SwinIR_Backbone':
        netG = SwinIR_Backbone_arch.SwinIR_Backbone(window_size=opt_net['window_size'], img_range=1., depths=opt_net['depths'],
                   embed_dim=opt_net['nf'], num_heads=opt_net['num_heads'], mlp_ratio=opt_net['mlp_ratio'],
                   resi_connection=opt_net['resi_connection'], 
                   global_residual=opt_net['GR']) 

    elif which_model == 'Uformer_Backbone':
        netG = Uformer_Backbone_arch.Uformer_Backbone(img_size=opt_net['img_size'], embed_dim=32, depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
                 win_size=8, mlp_ratio=4., out_chans=64, token_projection='linear', token_mlp='leff', modulator=True)
        
    elif which_model == 'Restormer_Backbone':
        netG = Restormer_Backbone_arch.Restormer_Backbone(
        inp_channels = opt_net['inp_channels'], 
        out_channels = opt_net['out_channels'], 
        dim = opt_net['dim'],
        num_blocks = opt_net['num_blocks'], 
        num_refinement_blocks = opt_net['num_refinement_blocks'],
        heads = opt_net['heads'],
        ffn_expansion_factor = opt_net['ffn_expansion_factor'],
        bias = opt_net['bias'],
        global_residual = opt_net['global_residual'],
        LayerNorm_type = opt_net['LayerNorm_type'],   ## Other option 'BiasFree'
        dual_pixel_task = opt_net['dual_pixel_task']        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        )

    elif which_model == 'Uformer_Encoder':
        netG = Uformer_pretrain_arch.Uformer_Encoder(img_size=opt_net['img_size'], embed_dim=32, depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True)
    else:
        raise NotImplementedError('Encoder model [{:s}] not recognized'.format(which_model))
    return netG

#### Deep Degradation Generator    
def define_DDG(opt):
    opt_net = opt['network_DDG']
    which_model = opt_net['which_model_DDG']

    if which_model == 'CSRNet_GFM_7':
        netG = DDG_arch.CSRNet_GFM_7(noise=opt_net['noise'], cond_dim=opt_net['modul_channels'])
    elif which_model == 'UNet4':
        netG = DDG_arch.UNet4(noise=opt_net['noise'], modul_channels=opt_net['modul_channels'])
    elif which_model == 'CSRNet_GFM_7_StyleGAN2':
        netG = DDG_arch.CSRNet_GFM_7_StyleGAN2(nf=opt_net['nf'], embed_ch=opt_net['modul_channels'])
    elif which_model == 'SwinIR_modulation':
        netG = SwinIR_arch.SwinIR_modulation(depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=8, embed_dim=120, mlp_ratio=2, degradation_embed_dim=512, resi_connection='3conv', 
                 require_modulation=opt_net['require_modulation'])
    elif which_model == 'RRDBNet_2':
        netG = RRDBNet_arch.RRDBNet(in_nc=120, out_nc=3, nf=64, nb=2, scale=1)
    elif which_model == 'SwinIR_modulation_Decoder':
        netG = SwinIR_pretrain_arch.SwinIR_modulation_Decoder(
                   window_size=opt_net['window_size'], img_range=1., depths=opt_net['depths'],
                   embed_dim=opt_net['nf'], num_heads=opt_net['num_heads'], mlp_ratio=opt_net['mlp_ratio'],
                   degradation_embed_dim=512, upscale=opt_net['upscale'], upsampler=opt_net['upsampler'],
                   require_modulation=opt_net['require_modulation'], 
                   resi_connection=opt_net['resi_connection'])
    
    elif which_model == 'Uformer_Decoder_modulation':
        netG = Uformer_pretrain_arch.Uformer_Decoder_modulation(img_size=opt_net['img_size'], embed_dim=16,depths=[1, 2, 8, 8, 2, 8, 8, 2, 1],
                 win_size=8, mlp_ratio=4., token_projection='linear', token_mlp='leff', modulator=True, shift_flag=False,
                 degradation_embed_dim = 512, require_modulation=opt_net['require_modulation'])
    
    elif which_model == 'CNN_Linear_Probing':
        netG = SwinIR_pretrain_arch.CNN_Linear_Probing(
            in_channels=opt_net['in_nc'], out_channels=opt_net['out_nc'],
            upscale=opt_net['upscale'], upsampler=opt_net['upsampler'])
            
    elif which_model == 'RCAN_Head':
        netG = RCAN_Pretrain_Head_arch.RCAN_Head(in_c=opt_net['in_nc'], out_c=opt_net['out_nc'], scale=opt_net['upscale'], require_modulation=opt_net['require_modulation'])

    elif which_model == 'MSRResNet_Head':
        netG = RCAN_Pretrain_Head_arch.MSRResNet_Head(in_c=opt_net['in_nc'], out_c=opt_net['out_nc'], scale=opt_net['upscale'], require_modulation=opt_net['require_modulation'])

    elif which_model == 'MSRResNet_Head_1res':
        netG = RCAN_Pretrain_Head_arch.MSRResNet_Head_1res(in_c=opt_net['in_nc'], out_c=opt_net['out_nc'], scale=opt_net['upscale'], require_modulation=opt_net['require_modulation'])
    elif which_model == 'Simple_Head':
        netG = RCAN_Pretrain_Head_arch.Simple_Head(in_c=opt_net['in_nc'], out_c=opt_net['out_nc'], scale=opt_net['upscale'], require_modulation=opt_net['require_modulation'])
    elif which_model == 'One_Conv_Head':
        netG = RCAN_Pretrain_Head_arch.One_Conv_Head(in_c=opt_net['in_nc'], out_c=opt_net['out_nc'], scale=opt_net['upscale'], require_modulation=opt_net['require_modulation'])

    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    return netG


#### Deep Degradation Embedding Encoder
def define_DDE_Encoder(opt):
    opt_net = opt['network_DDE_Encoder']
    which_model = opt_net['which_model_DDE_Encoder']

    if which_model == 'DeepDegradationEncoder_v1':
        netG = DDG_arch.DeepDegradationEncoder_v1(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'DeepDegradationEncoder_v2':
        netG = DDG_arch.DeepDegradationEncoder_v2(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'],
                                                  checkpoint=opt_net['pretrained_DDR_model'])
    elif which_model == 'DeepDegradationEncoder_v3':
        netG = DDG_arch.DeepDegradationEncoder_v3(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'],
                                                  checkpoint=opt_net['pretrained_DDR_model'])
    elif which_model == 'DeepDegradationEncoder_v4':
        netG = DDG_arch.DeepDegradationEncoder_v4(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'],
                                                  checkpoint=opt_net['pretrained_DDR_model'])
    elif which_model == 'DeepDegradationEncoder_v5':
        netG = DDG_arch.DeepDegradationEncoder_v5(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'],
                                                  checkpoint=opt_net['pretrained_DDR_model'])
    elif which_model == 'DeepDegradationEncoder_v6':
        netG = DDG_arch.DeepDegradationEncoder_v6(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'],
                                                  checkpoint=opt_net['pretrained_DDR_model'])
    elif which_model == 'DeepDegradationEncoder_v7':
        netG = DDG_arch.DeepDegradationEncoder_v7(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'],
                                                  checkpoint=opt_net['pretrained_DDR_model'])
    elif which_model == 'ConditionNet_GFM':
        netG = DDG_arch.ConditionNet_GFM()
    elif which_model == 'SwinIR_Encoder':
        netG = SwinIR_pretrain_arch.SwinIR_Encoder(window_size=8, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=120, num_heads=[6, 6, 6, 6], mlp_ratio=2)                               
    
    

    else:
        raise NotImplementedError('Encoder model [{:s}] not recognized'.format(which_model))
    return netG

#### Discriminator
def define_D_DDG(opt):
    opt_net = opt['network_D_DDG']
    which_model = opt_net['which_model_D_DDG']

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
