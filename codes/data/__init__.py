'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'LQ':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    elif mode == 'LQGT_RESIDE':
        from data.LQGT_RESIDE_dataset import LQGTDataset as D
    elif mode == 'LQGT_degrade':
        from data.LQGT_degrade_dataset import LQGTDataset as D
    elif mode == 'LQGT_degrade_val':
        from data.LQGT_degrade_val_dataset import LQGTDataset as D
    elif mode == 'triangle_dataset':
        from data.LQGT_tri_dataset import LQGTDataset as D
    elif mode == 'LQGT_rain':
        from data.LQGT_rain_dataset import LQGTDataset as D
    elif mode == 'LQGT_GoPro':
        from data.LQGT_GoPro_dataset import LQGTDataset as D
        
    elif mode == 'DDG_train':
        from data.DDG_train_dataset import DDGTrainDataset as D
    elif mode == 'DDG_train_blur':
        from data.DDG_train_blur_dataset import DDGTrainDataset as D
    elif mode == 'DDG_train_noise':
        from data.DDG_train_noise_dataset import DDGTrainDataset as D
    elif mode == 'DDG_train_jpeg':
        from data.DDG_train_jpeg_dataset import DDGTrainDataset as D
    elif mode == 'DDG_train_pretrain':
        from data.DDG_train_Pretrain_dataset import DDGTrainDataset as D
    elif mode == 'DDG_train_pretrain_complex':
        from data.DDG_train_Pretrain_Complex_dataset import DDGTrainDataset as D
        
    elif mode == 'DDG_val':
        from data.DDG_val_dataset import DDGValDataset as D
    elif mode == 'DDG_val_blur':
        from data.DDG_val_blur_dataset import DDGValDataset as D
    elif mode == 'DDG_val_noise':
        from data.DDG_val_noise_dataset import DDGValDataset as D
    elif mode == 'DDG_val_jpeg':
        from data.DDG_val_jpeg_dataset import DDGValDataset as D
        
    elif mode == 'DDG_test':
        from data.DDG_test_dataset import DDGTestDataset as D

    elif mode == 'HQ':
        from data.HQ_dataset import HQDataset as D
        
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
