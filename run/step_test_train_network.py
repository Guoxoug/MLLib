#! /usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from MLLib.models.densenet import DenseNet
from MLLib.models.model_generator import get_model
from MLLib.utils.utils import Meter, move_to_device
from MLLib.utils.temperature_scaling import ModelWithTemperature 
from MLLib.data_importer.ds_cifar import DS_cifar100, DS_cifar10
from MLLib.utils.loss import ECELoss, Error_topk, get_ece

commandLineParser = argparse.ArgumentParser(description='Train Models')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path to directory location where to setup')
commandLineParser.add_argument('--device_type', type=str, choices=['cpu', 'cuda'], default='cuda',
                               help='choose to run on gpu or cpu')


def test_train(dataset, net_arch, trn_para, save, device_type, model_filename='model.pth', batch_size=256):
    """
    A function to test a DenseNet-BC on data.

    Args:
        data (class Data) - data instance
        save (str) - path to save the model to (default /outputs)
        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        n_epochs (int) - number of epochs for training (default 300)
        lr (float) - initial learning rate
        wd (float) - weight decay
        momentum (float) - momentum
    """
    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    model = get_model(net_arch)
    model_uncalibrated = move_to_device(model, device_type)
    test_model = model_uncalibrated
    # Load model state dict
    model_state_filename = os.path.join(save, model_filename)
    if not os.path.exists(model_state_filename):
        raise RuntimeError('Cannot find file %s to load' % model_state_filename)
    state_dict = torch.load(model_state_filename)
    test_model.load_state_dict(state_dict)
    test_model.eval()

    # Make dataloader
    test_loader = dataset.test_train_loader
    
    nll_criterion = move_to_device(nn.CrossEntropyLoss(), device_type)
    ece_criterion = move_to_device(ECELoss(), device_type)
    # First: collect all the logits and labels for the test set
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in test_loader:
            input = move_to_device(input, device_type)
            logits = test_model(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = move_to_device(torch.cat(logits_list), device_type)
        labels = move_to_device(torch.cat(labels_list), device_type)
    # Calculate Error
    error_1 = Error_topk(logits, labels, 1).item()
    error_5 = Error_topk(logits, labels, 5).item()
    # Calculate NLL and ECE before temperature scaling
    result_nll = nll_criterion(logits, labels).item()
    result_ece = ece_criterion(logits, labels).item()
    #result_ece_2 = get_ece(logits, labels)
    torch.save(logits, os.path.join(save, 'test_train_logits.pth'))
    torch.save(labels, os.path.join(save, 'test_train_labels.pth'))
    np.save(os.path.join(save, 'test_train_logits.npy'), logits.cpu().numpy())
    np.save(os.path.join(save, 'test_train_labels.npy'), labels.cpu().numpy())
        
if __name__ == '__main__':
    """
    Test a 40-layer DenseNet-BC on CIFAR-100

    Args:
        --data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        --save (str) - path to save the model to (default /tmp)

        --valid_size (int) - size of validation set
        --seed (int) - manually set the random seed (default None)
    """
    args = commandLineParser.parse_args()

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_test_train_network.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if os.path.isdir(args.destination_dir):
        print('destination directory exists.')
    else:
        os.makedirs(args.destination_dir)

    path = os.path.join(args.destination_dir, 'cfg/net_arch.pickle')                     
    with open(path, 'rb') as handle:
        network_architecture = pickle.load(handle)
    
    path = os.path.join(args.destination_dir, 'cfg/trn_para.pickle')                     
    with open(path, 'rb') as handle:                 
        tps = pickle.load(handle) 

    # initialize loaders
    if tps['data_name'] == 'cifar100':
        ds = DS_cifar100(tps['data_name'], tps['data_path'], tps['batch_size'], tps['valid_size'], 'cfg/data', tps['data_indices_path']) 
    elif tps['data_name'] == 'cifar10':
        ds = DS_cifar10(tps['data_name'], tps['data_path'], tps['batch_size'], tps['valid_size'], 'cfg/data', tps['data_indices_path']) 
    test_train(ds, network_architecture, tps, args.destination_dir+'/outputs', args.device_type, model_filename='model.pth') 
