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
from torch.nn import functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from MLLib.models.densenet import DenseNet
from MLLib.models.model_generator import get_model
from MLLib.utils.utils import *
from MLLib.utils.temperature_scaling import ModelWithTemperature 
from MLLib.utils.uncertainties import *
from MLLib.data_importer.ds_cifar import DS_cifar100, DS_cifar10, get_dataset
from MLLib.utils.loss import TCELoss, ECELoss, Error_topk, get_ece
from pytorch_model_summary import summary
from MLLib.models.calibrator_cpu import Calibrator

commandLineParser = argparse.ArgumentParser(description='Calibration')
commandLineParser.add_argument('ensbl_dir', type=str,
                               help='absolute path to directory location where to ensbl members are stored')
commandLineParser.add_argument('model_name', type=str,
                               help='e.g DSN, RSN')
commandLineParser.add_argument('num_members', type=int,
                               help='number of members in ensbl')
commandLineParser.add_argument('calibration_method', type=int,
                               help='index of calibration method, 0 for sig_temp, 1 for ens_temp_bef, 2 for ens_temp_aft')
commandLineParser.add_argument('--device_type', type=str, choices=['cpu', 'cuda'], default='cuda',
                               help='choose to run on gpu or cpu')
commandLineParser.add_argument('--dyn_T_factor', type=str, choices=['confidence','expected_entropy','entropy_of_expected','mutual_information','EPKL'], default='confidence',
                               help='the factor that dynamically determine the temperature')
commandLineParser.add_argument('--num_factor_bins', type=int, default='4',
                               help='number of bins for dynamic temperature factors')
commandLineParser.add_argument('--num_iters', type=int, default='50',
                               help='number of iters in optimization')
commandLineParser.add_argument('--opt_target', type=str, choices=['NLL','ECE', 'TCE', 'ACCE', 'ECCE'], default='NLL',
                               help='loss function for optimization, NLL, ECE, or TCE')
commandLineParser.add_argument('--opt_method', type=str, choices=['LBFGS','SGD'], default='SGD',
                               help='optimizer used, LBFGS or SGD')
commandLineParser.add_argument('--learning_rate', type=float, default=0.01,
                               help='learning rate for the optimizer')

def obtain_uncertainty_bins(factors, num_factor_bins, epsilon=1e-7):
    sorted_factors, _ = torch.sort(factors) 
    tot = sorted_factors.shape[0]
    ind = torch.arange(0, tot, tot/num_factor_bins)
    return sorted_factors[ind[1:].long()] #not include the lower and upper boundaries

def obtain_uncertainty_mask(factors, factor_bins, num_factor_bins, epsilon=1e-7):
    factor_bins_ = torch.cat([torch.min(factors).unsqueeze(0)-epsilon, factor_bins, torch.max(factors).unsqueeze(0)+epsilon])
    
    ind_list = []
    for i in range(factor_bins_.shape[0]-1):
        low = factor_bins_[i]
        high = factor_bins_[i+1]
        ind = torch.bitwise_and(torch.ge(factors, low), torch.lt(factors, high))
        ind_list.append(ind.unsqueeze(1))
    '''
    for i in range(factor_bins.shape[0]):
        if i == 0:
            low = torch.min(factors)-epsilon
            high = factor_bins[0]
        elif i == factor_bins.shape[0]-1:
            low = factor_bins[i-1]
            high = torch.max(factors)+epsilon
        else:
            low = factor_bins[i-1]
            high = factor_bins[i]
        ind = torch.bitewise_and(torch.ge(factors, low), torch.lt(factors, high))
        ind_list.append(ind.unsqueeze(1))
    '''
    return torch.cat(ind_list, dim=1)
    

def calibrate_on_val(valid_logits, valid_labels, valid_masks, test_logits, test_labels, test_masks, calibration_method, save, device_type, opt_target, opt_method, num_iters, learning_rate, dyn_T_factor, num_factor_bins):
    """
    A function to calibrate model/ensemble on validation data.

    Args:
        valid_logits - model/ensemble-generated outputs on validation set
        valid_labels - labels on validation set
        test_logits - model/ensemble-generated outputs on test set
        test_labels - labels on test set
        calibration_method - 0: sig_temp, 1: ens_temp_bef, 2: ens_temp_aft
        save - save path of optimized calibrator
    """
    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)
    
    #valid_logits = move_to_device(valid_logits, device_type)
    #valid_labels = move_to_device(valid_labels, device_type)
    num_members = valid_logits.size()[1]
    calibrator = Calibrator(calibration_method, num_members, num_factor_bins) 
    with open(os.path.join(save, 'ens_cal', 'cal_results.txt'), 'a') as f:
        f.write('============'+ calibrator.calibration_method+' '+str(num_factor_bins)+' bins on '+args.dyn_T_factor+'============\n')
        if calibrator.calibration_method == 'ens_temp_aft':
            valid_logits = prob_to_logit(torch.mean(F.softmax(valid_logits, dim=2), dim=1))
            test_logits = prob_to_logit(torch.mean(F.softmax(test_logits, dim=2), dim=1))
        calibrator.opt_calibration(valid_logits, valid_labels, valid_masks, opt_target, opt_method, num_iters, learning_rate, True)  
        #test_logits = move_to_device(test_logits, device_type)
        #test_labels = move_to_device(test_labels, device_type)
        result = calibrator.eval_(test_logits, test_labels)
        f.write('Test Before Calibration - '+result+'\n')
        cal_test_logits = calibrator.cal_logits(test_logits, test_masks)
        result = calibrator.eval_(cal_test_logits, test_labels)
        f.write('Test After Calibration - '+result+'\n')
    if calibrator.calibration_method in ['sig_temp', 'ens_temp_bef', 'ens_temp_aft']:
        cal_test_probs = F.softmax(cal_test_logits, dim=1)
        cal_valid_logits = calibrator.cal_logits(valid_logits, valid_masks)
        cal_valid_probs = F.softmax(cal_valid_logits, dim=1)
        name_str = calibrator.calibration_method+'_'+str(num_factor_bins)+'_'+dyn_T_factor
        torch.save(calibrator.temperature.cpu(), os.path.join(save, 'ens_cal', name_str+'_test_T.pth'))
        cal_temperature = calibrator.temperature.detach().numpy()
        cal_test_logits = cal_test_logits.detach().numpy()
        cal_test_probs = cal_test_probs.detach().numpy()
        cal_valid_logits = cal_valid_logits.detach().numpy()
        cal_valid_probs = cal_valid_probs.detach().numpy()
        np.save(os.path.join(save, 'ens_cal', name_str+'_test_T.npy'), cal_temperature)
        np.save(os.path.join(save, 'ens_cal', name_str+'_test_logits.npy'), cal_test_logits)
        np.save(os.path.join(save, 'ens_cal', name_str+'_test_probs.npy'), cal_test_probs)
        np.save(os.path.join(save, 'ens_cal', name_str+'_valid_logits.npy'), cal_valid_logits)
        np.save(os.path.join(save, 'ens_cal', name_str+'_valid_probs.npy'), cal_valid_probs)
        if calibrator.calibration_method  == 'ens_temp_bef':
            uncal_test_probs = F.softmax(test_logits, dim=2).detach().numpy()
            uncal_valid_probs = F.softmax(valid_logits, dim=2).detach().numpy()
            test_labels = test_labels.detach().numpy()
            valid_labels = valid_labels.detach().numpy()
            np.save(os.path.join(save, 'ens_cal', 'uncal_valid_probs.npy'), uncal_valid_probs)
            np.save(os.path.join(save, 'ens_cal', 'uncal_test_probs.npy'), uncal_test_probs)
            np.save(os.path.join(save, 'ens_cal', 'valid_labels.npy'), valid_labels)
            np.save(os.path.join(save, 'ens_cal', 'test_labels.npy'), test_labels)
        #plot figures
        
        
        

if __name__ == '__main__':
    """
    Calibration

    Args:
        --ens_path (str) - path to directory where ensemble members are stored
        --save (str) - sub director where valid data are stored

    """
    args = commandLineParser.parse_args()

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_calibrate_network.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if not os.path.isdir(os.path.join(args.ensbl_dir, 'ens_cal')):
        os.makedirs(os.path.join(args.ensbl_dir, 'ens_cal'))
    
    #Aggregating logits and labels
    valid_logits = []
    valid_labels = []
    test_logits = []
    test_labels = []
    for i in range(args.num_members):
        output_dir = os.path.join(args.ensbl_dir, args.model_name+str(i+1), 'outputs')
        #v_logits = torch.load(output_dir+'/test_valid_logits.pth')
        #v_labels = torch.load(output_dir+'/test_valid_labels.pth')
        #t_logits = torch.load(output_dir+'/uncal_logits.pth')
        #t_labels = torch.load(output_dir+'/uncal_labels.pth')
        v_logits = torch.from_numpy(np.load(output_dir+'/test_valid_logits.npy'))
        v_labels = torch.from_numpy(np.load(output_dir+'/test_valid_labels.npy'))
        t_logits = torch.from_numpy(np.load(output_dir+'/uncal_logits.npy'))
        t_labels = torch.from_numpy(np.load(output_dir+'/uncal_labels.npy'))
        valid_logits.append(v_logits.unsqueeze(1))
        valid_labels.append(v_labels)
        test_logits.append(t_logits.unsqueeze(1))
        test_labels.append(t_labels)
    valid_logits = torch.cat(valid_logits,dim=1)
    valid_labels = (valid_labels[0])
    test_logits = torch.cat(test_logits, dim=1)
    test_labels = (test_labels[0])
    valid_uncertainties = ensemble_uncertainties_torch(F.softmax(valid_logits, dim=2))
    test_uncertainties = ensemble_uncertainties_torch(F.softmax(test_logits, dim=2))
    
    print(valid_uncertainties[args.dyn_T_factor].shape)
    print(valid_uncertainties[args.dyn_T_factor])
    factor_bins = obtain_uncertainty_bins(valid_uncertainties[args.dyn_T_factor], args.num_factor_bins) 
    print(factor_bins)
    valid_masks = obtain_uncertainty_mask(valid_uncertainties[args.dyn_T_factor], factor_bins, args.num_factor_bins).float()
    test_masks = obtain_uncertainty_mask(test_uncertainties[args.dyn_T_factor], factor_bins, args.num_factor_bins).float()
    
    lr = args.learning_rate if args.calibration_method == 2 else args.learning_rate*args.num_factor_bins
    #lr = args.learning_rate
    calibrate_on_val(valid_logits, valid_labels, valid_masks, test_logits, test_labels, test_masks, args.calibration_method, args.ensbl_dir, args.device_type, args.opt_target, args.opt_method, args.num_iters, lr, args.dyn_T_factor, args.num_factor_bins)
