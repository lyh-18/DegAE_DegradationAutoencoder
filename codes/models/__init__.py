import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'sr_modulation':
        from .SR_modulation_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
        
    elif model == 'sr_fea':
        from .SR_model_fea import SRModel as M
    elif model == 'sr_modulation_fea':
        from .SR_modulation_model_fea import SRModel as M
        
    elif model == 'ddg':
        from .DDG_model import DDGModel as M
    elif model == 'ddg_mse':
        from .DDG_MSE_model import DDGModel as M
    elif model == 'ddg_srgan':
        from .DDG_SRGAN_model import DDG_SRGAN_Model as M
    elif model == 'ddg_4decoder':
        from .DDG_substitution_model import DDGModel as M
    elif model == 'ddg_1decoder':
        from .DDG_Pretrain_onedecoder_model import DDGModel as M
    elif model == 'ddg_encoder_decoder':
        from .DDG_Encoder_Decoder_model import DDGModel as M
    elif model == 'ddg_encoder_decoder_mse':
        from .DDG_Encoder_Decoder_model_MSE import DDGModel as M
        
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'tbsrgan':
        from .TBSRGAN_model import TBSRGANModel as M
    elif model == 'tbsrgan_psnr':
        from .TBSRGAN_PSNR_model import TBSRGAN_PSNR_Model as M
        
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
