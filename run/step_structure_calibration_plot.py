#! /usr/bin/env python
import os
import argparse
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
import seaborn as sns
from MLLib.models.calibrator_cpu import __calibration_methods__
from MLLib.utils.loss import cal_multiple_ECE
sns.set(color_codes=True)

commandLineParser = argparse.ArgumentParser(description='Calibration_Plot')
commandLineParser.add_argument('ensbl_dir', type=str,
                               help='absolute path to directory location where to ensbl members are stored')
commandLineParser.add_argument('calibration_method', type=int,
                               help='index of calibration method, 0 for sig_temp, 1 for ens_temp_bef, 2 for ens_temp_aft')
commandLineParser.add_argument('dyn_T_factor', type=str, choices=['confidence','expected_entropy','entropy_of_expected','mutual_information','EPKL'], default='confidence',
                               help='the factor that dynamically determine the temperature')

def plot_reliability_curve(probs_, targets_, file_path=None, n_bins=15):
    plt.plot([0, 1], [0, 1], linestyle='--')
    fop, mpv = calibration_curve((targets_==np.argmax(probs_,axis=1)), np.max(probs_, axis=1), n_bins=n_bins)
    plt.plot(mpv, fop, marker='v')
    plt.ylabel('Fraction of corrects')
    plt.xlabel('Max Probability')
    plt.title('Top-label Reliability Curve')
    plt.legend()
    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight')
    else:
        plt.show()
    #plt.close()

if __name__=='__main__':
    args = commandLineParser.parse_args()
    eval_n_bins = 15
    cal_method = __calibration_methods__[args.calibration_method]
    #load data
    output_dir = os.path.join(args.ensbl_dir, 'ens_multi_structure_cal') 
    plot_dir = os.path.join(args.ensbl_dir, 'ens_multi_structure_cal', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    test_valid = 'test'
    num_factor_bins = 1
    use_cal_structure = True
    weighted_combine = True
    name_str = 'calibration_method-'+str(args.calibration_method)+'_'+str(num_factor_bins)+'bin_confidence_'+ \
            ('calStruct_' if use_cal_structure else 'uncalStruct_') + ('weightComb' if weighted_combine else 'unweightedComb')
    cal_structure_probs = np.load(output_dir+'/'+name_str+'_structure_test_probs.npy')
    uncal_ens_probs = np.load(output_dir+'/'+name_str+'_uncal_test_probs.npy')
    cal_ens_probs = np.load(output_dir+'/'+name_str+'_test_probs.npy')
    labels = np.load(output_dir +'/'+name_str+'_test_labels.npy')
    '''
    max_num_bins = 10
    for i in range(1,max_num_bins+1,1):
        p = np.load(output_dir+'/'+cal_method+'_'+str(i)+'_'+args.dyn_T_factor+'_'+test_valid+'_probs.npy') 
        probs_list.append(p[:,np.newaxis])
    '''
    #plot reliability curve of various bin_num
    plt.plot([0, 1], [0, 1], linestyle='--')
    #struct_names = ['LEN', 'DSN100', 'DSN121', 'RSN']
    for i in range(cal_structure_probs.shape[1]):
        fop, mpv = calibration_curve((labels == np.argmax(cal_structure_probs[:,i], axis=1)), np.max(cal_structure_probs[:,i], axis=1), n_bins=eval_n_bins)
        if i == 0:
            plt.plot(mpv, fop, 'gx--', label='cal struct')
        else:
            plt.plot(mpv, fop, 'gx--')
    fop, mpv = calibration_curve((labels == np.argmax(uncal_ens_probs, axis=1)), np.max(uncal_ens_probs, axis=1), n_bins=eval_n_bins)
    plt.plot(mpv, fop, 'rv-', lw='2', label='cal struct ens')
    fop, mpv = calibration_curve((labels == np.argmax(cal_ens_probs, axis=1)), np.max(cal_ens_probs, axis=1), n_bins=eval_n_bins)
    plt.plot(mpv, fop, 'bo-', lw='2', label='cal struct cal ens')
    plt.ylabel('Fraction of corrects')
    plt.xlabel('Max Probability')
    plt.title('Top-label Reliability Curve')
    plt.legend()
    plt.savefig(plot_dir+'/reliability_curves_'+args.dyn_T_factor, bbox_inches='tight')
    plt.close()
    
 
    
    
