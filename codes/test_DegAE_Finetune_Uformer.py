import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import numpy as np

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

from niqe import calculate_niqe

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['niqe'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    crop_border = opt['crop_border'] if opt['crop_border'] is not None else opt['scale']
    print('crop_border: ', crop_border)
    
    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        model.feed_data(data, need_GT=need_GT)
        img_path = data['LQ_path'][0] if need_GT else data['LQ_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test_Uformer()
        visuals = model.get_current_visuals(need_GT=need_GT)

        sr_img = util.tensor2img(visuals['SR'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix:
            save_img_path = osp.join(dataset_dir, img_name + suffix + '.png')
        else:
            save_img_path = osp.join(dataset_dir, img_name + '.png')
        util.save_img(sr_img, save_img_path)

        # calculate PSNR and SSIM
        
        if need_GT:
            gt_img = util.tensor2img(visuals['GT'])
            
            if crop_border == 0 or crop_border == 1:
                cropped_sr_img = sr_img
                cropped_gt_img = gt_img
            else:
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = util.calculate_psnr(cropped_sr_img, cropped_gt_img)
            ssim = util.calculate_ssim(cropped_sr_img, cropped_gt_img)
            try:
                niqe_score = float(calculate_niqe(cropped_sr_img, crop_border=crop_border))
                test_results['niqe'].append(niqe_score)
            except:
                print('skip {}'.format(img_path))
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            

            if gt_img.shape[2] == 3:  # RGB image
                sr_img_y = bgr2ycbcr(sr_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                
                sr_img_y = np.clip((sr_img_y*255).round(), 0, 255).astype(np.uint8)
                gt_img_y = np.clip((gt_img_y*255).round(), 0, 255).astype(np.uint8)
                
                if crop_border == 0:
                    cropped_sr_img_y = sr_img_y
                    cropped_gt_img_y = gt_img_y
                else:
                    cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
                    cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                psnr_y = util.calculate_psnr(cropped_sr_img_y, cropped_gt_img_y)
                ssim_y = util.calculate_ssim(cropped_sr_img_y, cropped_gt_img_y)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                #logger.info(
                #    '{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; NIQE: {:.4f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
                #    format(img_name, psnr, ssim, niqe_score, psnr_y, ssim_y))
            else:
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
        else:
            logger.info(img_name)
            sr_img = sr_img / 255.

            crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
            if crop_border == 0:
                cropped_sr_img = sr_img
            else:
                cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
                
            try:
                niqe_score = float(calculate_niqe(cropped_sr_img * 255, crop_border=crop_border))
                test_results['niqe'].append(niqe_score)
                logger.info('{:20s} - NIQE: {:.6f} dB.'.format(img_name, niqe_score))
            except:
                print('skip {}'.format(img_path))
            
            

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        ave_niqe = sum(test_results['niqe']) / len(test_results['niqe'])
        logger.info(
            '----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; NIQE: {:.4f}\n'.format(
                test_set_name, ave_psnr, ave_ssim, ave_niqe))
        if test_results['psnr_y'] and test_results['ssim_y']:
            ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
            ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
            logger.info(
                '----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'.
                format(ave_psnr_y, ave_ssim_y))
                
    else:
        ave_niqe = sum(test_results['niqe']) / len(test_results['niqe'])
        logger.info(
            '----Average NIQE results for {}----\n\t NIQE: {:.4f}\n'.format(
                test_set_name, ave_niqe))
