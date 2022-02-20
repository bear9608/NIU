import argparse
import os
import yaml
import h5py
from torch.autograd import Variable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter


import datasets
import models
import utils
import utils.few_shot as fs
from datasets.samplers import CategoriesSampler


def main(config):
    svname = args.name
    if svname is None:
        svname = 'classifier_{}'.format(config['train_dataset'])
        svname += '_savefeature_' + config['model_args']['encoder']
    if args.tag is not None:
        svname += '_' + args.tag
    save_path = os.path.join('./save', svname)
    utils.ensure_path(save_path)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    #### Dataset ####

    # train
    train_dataset = datasets.make(config['train_dataset'],
                                  **config['train_dataset_args'])
    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=False,
                              num_workers=6, pin_memory=True)
    utils.log('train dataset: {} (x{}), {}'.format(
        train_dataset[0][0].shape, len(train_dataset),
        train_dataset.n_classes))
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_dataset, 'train_dataset', writer)

    # tval
    if config.get('tval_dataset'):
        tval_dataset = datasets.make(config['tval_dataset'],
                                     **config['tval_dataset_args'])
        utils.log('tval dataset: {} (x{}), {}'.format(
                tval_dataset[0][0].shape, len(tval_dataset),
                tval_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(tval_dataset, 'tval_dataset', writer)
        tval_loader = DataLoader(tval_dataset, config['batch_size'],shuffle=False,
                                 num_workers=4, pin_memory=True)
    else:
        tval_loader = None

    # val
    if config.get('val_dataset'):
        eval_val = True
        val_dataset = datasets.make(config['val_dataset'],
                                    **config['val_dataset_args'])
        val_loader = DataLoader(val_dataset, config['batch_size'],
                                num_workers=6, pin_memory=True)
        utils.log('val dataset: {} (x{}), {}'.format(
            val_dataset[0][0].shape, len(val_dataset),
            val_dataset.n_classes))
        if config.get('visualize_datasets'):
            utils.visualize_dataset(val_dataset, 'val_dataset', writer)
    else:
        eval_val = False


    ########

    #### Model and Optimizer ####

    if config.get('load'):
        model_sv = torch.load(config['load'])
        model = models.load(model_sv)
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
            model.parameters(),
            config['optimizer'], **config['optimizer_args'])



    ########

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_va = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()


    timer_epoch.s()
    aves_keys = ['tl', 'ta', 'vl', 'va']
    aves = {k: utils.Averager() for k in aves_keys}


    if eval_val:
        model.eval()
        f = h5py.File('savefeature_base.hdf5', 'w')
        data_Classify = train_dataset.n_classes
        max_count = len(train_loader) * train_loader.batch_size
        all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
        all_feats = None
        data_feature = None

        count = 0
        i = 0
        for data, label in tqdm(train_loader, desc='save_feature', leave=False):
            data, label = data.cuda(), label.cuda()
            with torch.no_grad():
                data = Variable(data)
                features = model(data)
                if all_feats is None:
                    all_feats = f.create_dataset('all_feats', [max_count] + list(features.size()[1:]), dtype='f')
                all_feats[count:count + features.size(0)] = features.data.cpu().numpy()
                all_labels[count:count + features.size(0)] = label.cpu().numpy()
                count = count + features.size(0)
        count_var = f.create_dataset('count', (1,), dtype='i')
        count_var[0] = count
        if data_feature is None:
            data_feature = f.create_dataset('data_feature',[data_Classify]+ list(features.size()[1:]),dtype='f')
        for i in range(data_Classify):
            a = all_feats[i*600]
            countdemo = 1
            for h in range(1,600):
                a = all_feats[h+i*600]+a
                countdemo += 1
            data_feature[i] = a/600
            print(len(data_feature))
        f.close()

        # post
    if lr_scheduler is not None:
        lr_scheduler.step()

    for k, v in aves.items():
        aves[k] = v.item()

    t_epoch = utils.time_str(timer_epoch.t())
    t_used = utils.time_str(timer_used.t())

    writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/save_feature.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True
        config['_gpu'] = args.gpu

    utils.set_gpu(args.gpu)
    main(config)




