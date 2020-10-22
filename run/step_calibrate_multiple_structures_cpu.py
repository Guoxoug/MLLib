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
from sklearn.metrics import roc_auc_score, roc_curve

commandLineParser = argparse.ArgumentParser(description='Calibration')
commandLineParser.add_argument('dest_dir', type=str,
                               help='absolute path to directory for storing results')
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
commandLineParser.add_argument('--opt_target', type=str, choices=['NLL','ECE', 'TCE'], default='NLL',
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
    
def structure_combination_calibration_on_val(valid_logits, valid_labels, valid_masks, test_logits, test_labels, test_masks, calibration_method, save, device_type, opt_target, opt_method, num_iters, learning_rate, dyn_T_factor, num_factor_bins, model_names, use_cal_structure=True, weighted_combine=0):
    """
    A function to obtain likelihood bayesian weights and calibrate the weighted combination on validation data.
    Args:
        valid_logits - ensemble-generated outputs on validation set [structure_num, sample_num, member_num, class_num]
        valid_labels - labels on validation set
        calibration_method - 0: sig_temp, 1: ens_temp_bef, 2: ens_temp_aft
    """
    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)
     
    #valid_logits = move_to_device(valid_logits, device_type)
    #valid_labels = move_to_device(valid_labels, device_type)
    nll_criterion = nn.CrossEntropyLoss()
    ece_criterion = ECELoss()
    tce_criterion = TCELoss()
    NLL_criterion = nn.NLLLoss()
    num_structures = valid_logits.size()[0]
    num_sample = valid_logits.size()[1]
    num_members = valid_logits.size()[2]
    structure_mean_lls = []
    struct_valid_logits_list = []
    struct_test_logits_list = []
    with open(os.path.join(save, 'ens_multi_structure_cal', 'cal_results.txt'), 'a') as f:
        f.write('============'+ str(num_factor_bins)+' bins on '+args.dyn_T_factor+'============\n')
        name_str = 'calibration_method-'+str(calibration_method)+'_'+str(num_factor_bins)+'bin_confidence_'+ \
            ('calStruct_' if use_cal_structure else 'uncalStruct_') + ('weightComb' if weighted_combine>0 else 'unweightedComb')  
        f.write(name_str+'\n')
        for i in range(num_structures):
            calibrator = Calibrator(calibration_method, num_members, num_factor_bins) 
            valid_logits_ = prob_to_logit(torch.mean(F.softmax(valid_logits[i], dim=2), dim=1))
            test_logits_ = prob_to_logit(torch.mean(F.softmax(test_logits[i], dim=2), dim=1))
            valid_masks_ = valid_masks[i]
            test_masks_ = test_masks[i]
            calibrator.opt_calibration(valid_logits_, valid_labels, valid_masks_, opt_target, opt_method, num_iters, learning_rate, True)  
            result_bef = calibrator.eval_(test_logits_, test_labels)
            cal_test_logits = calibrator.cal_logits(test_logits_, test_masks_)
            result_aft = calibrator.eval_(cal_test_logits, test_labels)
            cal_valid_logits = calibrator.cal_logits(valid_logits_, valid_masks_)
            structure_mean_lls.append(nll_criterion(cal_valid_logits, valid_labels).unsqueeze(0))
            if use_cal_structure:
                f.write('\tSTRUCT '+str(i)+' Test Before Calibration - '+result_bef+'\n')
                f.write('\tSTRUCT '+str(i)+' Test After Calibration - '+result_aft+' \n\ttemperature:'+str(calibrator.temperature.detach().numpy())+'\n')
                struct_valid_logits_list.append(cal_valid_logits.unsqueeze(1))
                struct_test_logits_list.append(cal_test_logits.unsqueeze(1))
            else:
                struct_valid_logits_list.append(valid_logits_.unsqueeze(1))
                struct_test_logits_list.append(test_logits_.unsqueeze(1))
        struct_valid_logits_list = torch.cat(struct_valid_logits_list, dim=1)
        struct_test_logits_list = torch.cat(struct_test_logits_list, dim=1)
        
        if weighted_combine == 0:
            structure_ens_valid_probs = torch.mean(F.softmax(struct_valid_logits_list,dim=2),dim=1)
            structure_ens_test_probs = torch.mean(F.softmax(struct_test_logits_list,dim=2),dim=1)
        elif weighted_combine == 1: # weight estimated by Max LL 
            # estimate combination weights by max LL
            log_alpha = nn.Parameter(torch.ones([1, num_structures]))
            optimizer = optim.SGD([log_alpha], lr=0.3)
            for tit in range(400):
                c_w = F.softmax(log_alpha,dim=1).unsqueeze(2).expand(struct_valid_logits_list.size())
                predicted_logits = prob_to_logit(torch.sum(F.softmax(struct_valid_logits_list.detach(),dim=2)*c_w,dim=1))
                optimizer.zero_grad()
                loss = nll_criterion(predicted_logits, valid_labels) 
                loss.backward()
                #print('c_w:'+str(loss))
                optimizer.step()
            c_w_valid = F.softmax(log_alpha,dim=1).unsqueeze(2).expand(struct_valid_logits_list.size())
            c_w_test = F.softmax(log_alpha,dim=1).unsqueeze(2).expand(struct_test_logits_list.size())
            f.write('\tcombine weight'+str(model_names)+' '+str(F.softmax(log_alpha, dim=1).detach().numpy())+'\n')
            structure_ens_valid_probs = torch.sum(F.softmax(struct_valid_logits_list,dim=2)*c_w_valid,dim=1) 
            structure_ens_test_probs = torch.sum(F.softmax(struct_test_logits_list,dim=2)*c_w_test,dim=1)
        elif weighted_combine == 2: # weight estimated by AUC
            # According to Zhong et al., Accurate Probability Calibration for Multiple Classifiers, IJCAI'13, https://www.ijcai.org/Proceedings/13/Papers/286.pdf
            confidences, predictions = torch.max(F.softmax(struct_valid_logits_list,dim=2),dim=2)
            conf_ = confidences.detach().numpy()
            pred_ = predictions.detach().numpy()
            label_=valid_labels.detach().numpy()
            num_structs = struct_valid_logits_list.size()[1]
            aucs = np.zeros(num_structs)
            for s_i in range(num_structs):
                roc_auc = roc_auc_score((pred_[:,s_i] == label_), conf_[:,s_i]) 
                aucs[s_i] = roc_auc
            print('AUCs: '+str(aucs))
            mu = np.mean(1.-aucs)
            log_alpha = torch.from_numpy(-(1.-aucs)/(2*mu))
            auc_w = F.softmax(log_alpha.unsqueeze(0),dim=1)
            f.write('\tcombine weight'+str(model_names)+' '+str(auc_w.detach().numpy())+'\n')
            a_w_valid = auc_w.unsqueeze(2).expand(struct_valid_logits_list.size())
            a_w_test = auc_w.unsqueeze(2).expand(struct_test_logits_list.size())
            structure_ens_valid_probs = torch.sum(F.softmax(struct_valid_logits_list,dim=2)*a_w_valid,dim=1)
            structure_ens_test_probs = torch.sum(F.softmax(struct_test_logits_list,dim=2)*a_w_test,dim=1)
        #Calibration after combining calibrated structures
        calibrator = Calibrator(calibration_method, num_members, num_factor_bins)
        ens_valid_logits = prob_to_logit(structure_ens_valid_probs)
        ens_test_logits = prob_to_logit(structure_ens_test_probs)
        ens_valid_uncertainties, _ = torch.max(structure_ens_valid_probs, dim=1)
        ens_test_uncertainties, _ = torch.max(structure_ens_test_probs, dim=1)
        ens_factor_bins = obtain_uncertainty_bins(ens_valid_uncertainties, args.num_factor_bins)
        ens_valid_masks = obtain_uncertainty_mask(ens_valid_uncertainties, ens_factor_bins, args.num_factor_bins).float()
        ens_test_masks = obtain_uncertainty_mask(ens_test_uncertainties, ens_factor_bins, args.num_factor_bins).float()
    
        calibrator.opt_calibration(ens_valid_logits.detach(), valid_labels, ens_valid_masks, opt_target, opt_method, num_iters, learning_rate, True)
        result = calibrator.eval_(ens_test_logits, test_labels)
        f.write('\tENS Test Before Calibration - '+result+'\n')
        cal_test_logits = calibrator.cal_logits(ens_test_logits, ens_test_masks)
        result = calibrator.eval_(cal_test_logits, test_labels)
        f.write('\tENS Test After Calibration - '+result+' \n\ttemperature:'+str(calibrator.temperature.detach().numpy())+'\n')
        cal_valid_logits = calibrator.cal_logits(ens_valid_logits, ens_valid_masks)
    cal_test_probs = F.softmax(cal_test_logits, dim=1)
    cal_valid_probs = F.softmax(cal_valid_logits, dim=1)
    torch.save(calibrator.temperature.cpu(), os.path.join(save, 'ens_multi_structure_cal', name_str+'_test_T.pth'))
    cal_temperature = calibrator.temperature.detach().numpy()
    cal_test_logits = cal_test_logits.detach().numpy()
    cal_test_probs = cal_test_probs.detach().numpy()
    cal_valid_logits = cal_valid_logits.detach().numpy()
    cal_valid_probs = cal_valid_probs.detach().numpy()
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_test_T.npy'), cal_temperature)
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_test_logits.npy'), cal_test_logits)
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_test_probs.npy'), cal_test_probs)
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_valid_logits.npy'), cal_valid_logits)
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_valid_probs.npy'), cal_valid_probs)
    uncal_test_probs = structure_ens_test_probs.detach().numpy()
    uncal_valid_probs = structure_ens_valid_probs.detach().numpy()
    test_labels = test_labels.detach().numpy()
    valid_labels = valid_labels.detach().numpy()
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_uncal_valid_probs.npy'), uncal_valid_probs)
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_uncal_test_probs.npy'), uncal_test_probs)
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_valid_labels.npy'), valid_labels)
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_test_labels.npy'), test_labels)
    
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_structure_test_logits.npy'), struct_test_logits_list.detach().numpy())
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_structure_test_probs.npy'), F.softmax(struct_test_logits_list,dim=2).detach().numpy())
    np.save(os.path.join(save, 'ens_multi_structure_cal', name_str+'_structure_valid_logits.npy'), struct_valid_logits_list.detach().numpy())
     
 
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
    with open('CMDs/step_calibrate_multiple_structures.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if not os.path.isdir(os.path.join(args.dest_dir, 'ens_multi_structure_cal')):
        os.makedirs(os.path.join(args.dest_dir, 'ens_multi_structure_cal'))
    
    #Aggregating logits and labels
    basic_path = '/home/mifs/xw369/calibration/baseline'
    DSN_100_dir = basic_path+'/densenetK12L100_ts_v4' 
    DSN_121_dir = basic_path+'/densenet121_ts_v4'
    DSN_100_2_dir = basic_path+'/densenetK12L100_ts_v2'
    RSN_wide_dir = basic_path+'/resnet_wide_28_10_ts_v4'
    RSN_wide_2_dir = basic_path+'/resnet_wide_28_10_ts_v2'
    LEN_dir = basic_path+'/lenet5_ts_v2'
    ensbl_dirs = [LEN_dir, DSN_100_dir, DSN_121_dir, RSN_wide_dir]   
    model_names = ['LEN', 'DSN', 'DSN', 'RSN']
    #ensbl_dirs = [DSN_100_dir, DSN_121_dir, RSN_wide_dir]   
    #model_names = ['DSN', 'DSN', 'RSN']
    #ensbl_dirs = [LEN_dir, DSN_100_dir, DSN_100_2_dir, DSN_121_dir, RSN_wide_dir, RSN_wide_2_dir]
    #model_names = ['LEN', 'DSN', 'DSN', 'DSN', 'RSN', 'RSN']
    structure_valid_logits = []
    structure_valid_labels = []
    structure_test_logits = []
    structure_test_labels = []
    structure_valid_masks = []
    structure_test_masks = []
    for j in range(len(ensbl_dirs)):
        ensbl_dir = ensbl_dirs[j]         
        model_name = model_names[j]
        valid_logits = []
        valid_labels = []
        test_logits = []
        test_labels = []
        for i in range(args.num_members):
            output_dir = os.path.join(ensbl_dir, model_name+str(i+1), 'outputs')
            v_logits = torch.from_numpy(np.load(output_dir+'/test_valid_logits.npy'))
            v_labels = torch.from_numpy(np.load(output_dir+'/test_valid_labels.npy'))
            t_logits = torch.from_numpy(np.load(output_dir+'/uncal_logits.npy'))
            t_labels = torch.from_numpy(np.load(output_dir+'/uncal_labels.npy'))
            valid_logits.append(v_logits.unsqueeze(1))
            valid_labels.append(v_labels)
            test_logits.append(t_logits.unsqueeze(1))
            test_labels.append(t_labels)
        valid_logits = torch.cat(valid_logits,dim=1)
        structure_valid_logits.append(valid_logits.unsqueeze(0))
        valid_labels = (valid_labels[0])
        test_logits = torch.cat(test_logits, dim=1)
        structure_test_logits.append(test_logits.unsqueeze(0))
        test_labels = (test_labels[0])
        valid_uncertainties = ensemble_uncertainties_torch(F.softmax(valid_logits, dim=2))
        test_uncertainties = ensemble_uncertainties_torch(F.softmax(test_logits, dim=2))
        factor_bins = obtain_uncertainty_bins(valid_uncertainties[args.dyn_T_factor], args.num_factor_bins) 
        valid_masks = obtain_uncertainty_mask(valid_uncertainties[args.dyn_T_factor], factor_bins, args.num_factor_bins).float()
        test_masks = obtain_uncertainty_mask(test_uncertainties[args.dyn_T_factor], factor_bins, args.num_factor_bins).float()
        structure_valid_masks.append(valid_masks.unsqueeze(0))
        structure_test_masks.append(test_masks.unsqueeze(0))
    structure_valid_labels = valid_labels
    structure_test_labels = test_labels
    structure_valid_logits = torch.cat(structure_valid_logits,dim=0)
    structure_test_logits = torch.cat(structure_test_logits,dim=0)
    structure_valid_masks = torch.cat(structure_valid_masks,dim=0)
    structure_test_masks = torch.cat(structure_test_masks,dim=0)
     
    lr = args.learning_rate if args.calibration_method == 2 else args.learning_rate*args.num_factor_bins
    structure_combination_calibration_on_val(structure_valid_logits, structure_valid_labels, structure_valid_masks, structure_test_logits, structure_test_labels, structure_test_masks, args.calibration_method, args.dest_dir, args.device_type, args.opt_target, args.opt_method, args.num_iters, lr, args.dyn_T_factor, args.num_factor_bins, model_names, use_cal_structure=True, weighted_combine=2)
# weighted_combine 0:mean, 1 LL 2 AUC

