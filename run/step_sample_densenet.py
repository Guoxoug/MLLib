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
from MLLib.data_importer.ds_cifar100 import DS_cifar100
from MLLib.utils.loss import ECELoss, Error_topk, get_ece

commandLineParser = argparse.ArgumentParser(description='Train Models')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path to directory location where to setup')
commandLineParser.add_argument('--device_type', type=str, choices=['cpu', 'cuda'], default='cuda',
                               help='choose to run on gpu or cpu')

def sample(dataset, net_arch, trn_para, save, device_type, model_filename='model.pth', batch_size=256):
    model = get_model(net_arch)
    sample_model = move_to_device(model, device_type) 
    model_state_filename = os.path.join(save, model_filename)
    state_dict = torch.load(model_state_filename)
    sample_model.load_state_dict(state_dict)
    sample_model.eval() 
    train_loader = dataset.train_loader
    sample_num = 20
    perturb_degree = 5
    inputs_list = []
    perturbations_list = []
    logits_list = []
    labels_list = []
    with torch.no_grad():
        cnt = 0
        for input, label in train_loader:
            if cnt>5:
                break
            cnt+=1
            cur_logits_list = []
            cur_perturbations_list = []
            input = move_to_device(input, device_type)   
            print('input:'+str(input.size()))
            logits = sample_model(input)
            print('logits:'+str(logits.size()))
            cur_logits_list.append(logits.unsqueeze_(1))
            for i in range(sample_num):
                perturbation = torch.randn_like(input[0])
                perturbation_flatten = torch.flatten(perturbation)
                p_n = torch.norm(perturbation_flatten, p=2)
                print('p_n:'+str(p_n.size()))
                perturbation = perturbation.div(p_n)
                perturbation = perturbation.unsqueeze(0).expand_as(input) * perturb_degree
                perturbation = move_to_device(perturbation, device_type)
                print('perturbation:'+str(perturbation.size()))
                cur_perturbations_list.append(perturbation.unsqueeze(1))
                perturb_input = input+perturbation 
                p_logits = sample_model(perturb_input)
                cur_logits_list.append(p_logits.unsqueeze_(1))
                
            cur_logits_list = torch.cat(cur_logits_list, dim=1)
            cur_perturbations_list = torch.cat(cur_perturbations_list, dim=1)
            logits_list.append(cur_logits_list)
            labels_list.append(label)
            inputs_list.append(input)
            perturbations_list.append(cur_perturbations_list)
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list) 
    inputs = torch.cat(inputs_list)
    perturbations = torch.cat(perturbations_list)
    np.save(os.path.join(save, 'sample_input_pdg'+str(perturb_degree)+'.npy'), inputs.cpu().numpy()) 
    np.save(os.path.join(save, 'sample_logits_pdg'+str(perturb_degree)+'.npy'), logits.cpu().numpy()) 
    np.save(os.path.join(save, 'sample_perturbations_pdg'+str(perturb_degree)+'.npy'),perturbations.cpu().numpy()) 
    np.save(os.path.join(save, 'sample_labels_pdg'+str(perturb_degree)+'.npy'), labels.cpu().numpy()) 

if __name__ == '__main__':
    """
    Sample a 40-layer DenseNet-BC on CIFAR-100

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
    with open('CMDs/step_sample_densenet.cmd', 'a') as f:
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
    ds_cifar100 = DS_cifar100(tps['data_name'], tps['data_path'], 128, tps['valid_size'], 'cfg/data', tps['data_indices_path']) 
    # Begin sample
    sample(ds_cifar100, network_architecture, tps, args.destination_dir+'/outputs', args.device_type, model_filename='model.pth') 
