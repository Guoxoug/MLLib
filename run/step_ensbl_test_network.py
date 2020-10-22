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
from MLLib.utils.utils import Meter, move_to_device
from MLLib.utils.temperature_scaling import ModelWithTemperature 
from MLLib.data_importer.ds_cifar100 import DS_cifar100
from MLLib.utils.loss import ECELoss, Error_topk

commandLineParser = argparse.ArgumentParser(description='Train Models')
commandLineParser.add_argument('source_dir', type=str,
                               help='absolute path to directory where models stored, e.g. parent dir of DSN1')
commandLineParser.add_argument('model_label', type=str,
                               help='model label, e.g. DSN')
commandLineParser.add_argument('num_models', type=int,
                               help='number of ensbl model')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path to directory location where to setup')
commandLineParser.add_argument('--device_type', type=str, choices=['cpu', 'cuda'], default='cuda',
                               help='choose to run on gpu or cpu')

def ensbl_test(save_dirs, target_save_dir, test_cal=False, logit_filename='uncal_logits.pth', label_filename='uncal_labels.pth', device_type='cpu'):
    """
    A function to test ensemble of DenseNet-BC on data.

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
    # Individual Models
    nll_criterion = move_to_device(nn.CrossEntropyLoss(), device_type)
    ece_criterion = move_to_device(ECELoss(), device_type)
    logits_list, labels_list, probs_list, ERR_1_list, ERR_5_list, NLL_list, ECE_list = [], [], [], [], [], [], []
    for sd in save_dirs:
        lg = torch.load(os.path.join(sd, logit_filename), map_location=torch.device(device_type))
        lb = torch.load(os.path.join(sd, label_filename), map_location=torch.device(device_type))
        #logits_list.append(torch.unsqueeze(lg, dim=1))
        labels_list.append(torch.unsqueeze(lb, dim=1))
        probs_list.append(torch.unsqueeze(torch.nn.Softmax(dim=-1)(lg),dim=1))
        ERR_1_list.append(torch.unsqueeze(Error_topk(lg, lb, 1),0))
        ERR_5_list.append(torch.unsqueeze(Error_topk(lg, lb, 5),0))
        NLL_list.append(torch.unsqueeze(nll_criterion(lg, lb),0))
        ECE_list.append(torch.unsqueeze(ece_criterion(lg, lb),0))
    ERR_1_list=torch.cat(ERR_1_list)
    ERR_5_list=torch.cat(ERR_5_list)
    print(ERR_1_list)
    print(ERR_5_list)
    NLL_list=torch.cat(NLL_list)
    ECE_list=torch.cat(ECE_list)*100
    labels_list=torch.cat(labels_list, dim=1)
    probs_list=torch.cat(probs_list, dim=1)
    avg_prob=torch.mean(probs_list, dim=1)
    pseudo_avg_logit=torch.log(avg_prob)
    lb=labels_list[:,0]
    avg_ERR_1 = Error_topk(pseudo_avg_logit, lb, 1)
    avg_ERR_5 = Error_topk(pseudo_avg_logit, lb, 5)
    avg_NLL = nll_criterion(pseudo_avg_logit, lb)
    avg_ECE = ece_criterion(pseudo_avg_logit, lb)
    cal_str = 'Calibrated' if test_cal else 'Uncalibrated'    
    with open(os.path.join(target_save_dir, 'ensbl_results.txt'), 'a') as f:
        f.write('+++++++'+cal_str+'+++++++\n')
        f.write('==Individual Models==\n')
        f.write(' ERR_1: %.4f \scriptsize{$\pm$ %.4f}\n ERR_5: %.4f \scriptsize{$\pm$ %.4f}\n NLL: %.3f \scriptsize{$\pm$ %.3f}\n ECE: %.4f \scriptsize{$\pm$ %.4f}\n' % ( torch.mean(ERR_1_list).item(), torch.std(ERR_1_list).item(),
                                                       torch.mean(ERR_5_list).item(), torch.std(ERR_5_list).item(),
                                                       torch.mean(NLL_list).item(), torch.std(NLL_list).item(),
                                                       torch.mean(ECE_list).item(), torch.std(ECE_list).item()))
        f.write('==Ensemble==\n')
        f.write(' ERR_1: %.4f \n ERR_5: %.4f \n NLL: %.3f \n ECE: %.4f \n' % ( avg_ERR_1.item(), avg_ERR_5.item(), avg_NLL.item(), avg_ECE.item()))
     
    
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
    with open('CMDs/step_ensbl_test_network.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if os.path.isdir(args.destination_dir):
        print('destination directory exists. ')
    else:
        os.makedirs(args.destination_dir)

    path = os.path.join(args.source_dir, args.model_label+'1', 'cfg/net_arch.pickle')                     
    with open(path, 'rb') as handle:
        net_arch = pickle.load(handle)
    
    path = os.path.join(args.source_dir, args.model_label+'1', 'cfg/trn_para.pickle')                     
    with open(path, 'rb') as handle:                 
        tps = pickle.load(handle) 

    # initialize loaders
    #ds_cifar100 = DS_cifar100(tps['data_name'], tps['data_path'], tps['batch_size'], tps['valid_size'], 'cfg/data', tps['data_indices_path']) 
    # Begin test    
    with open(os.path.join(args.destination_dir, 'ensbl_results.txt'), 'w') as f:
        f.write('Ensbl Result of '+net_arch['model_name']+' on Data '+net_arch['data_name']+'\n')
    save_dirs = [os.path.join(args.source_dir, args.model_label+str(i+1), 'outputs') for i in range(args.num_models)]
    ensbl_test(save_dirs, args.destination_dir, False, 'uncal_logits.pth', 'uncal_labels.pth')
    ensbl_test(save_dirs, args.destination_dir, True, 'cal_logits.pth', 'cal_labels.pth')
