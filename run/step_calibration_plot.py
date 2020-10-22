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
    output_dir = os.path.join(args.ensbl_dir, 'ens_cal') 
    plot_dir = os.path.join(args.ensbl_dir, 'ens_cal', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    test_valid = 'test'
    uncal_probs = np.load(output_dir +'/uncal_'+test_valid+'_probs.npy')
    labels = np.load(output_dir +'/'+test_valid+'_labels.npy')
    probs_list = []
    max_num_bins = 10
    for i in range(1,max_num_bins+1,1):
        p = np.load(output_dir+'/'+cal_method+'_'+str(i)+'_'+args.dyn_T_factor+'_'+test_valid+'_probs.npy') 
        probs_list.append(p[:,np.newaxis])
    probs = np.concatenate(probs_list, axis=1) #[SampleNum, BinNum, MemberNum, ClassNum]
    #plot reliability curve of various bin_num
    plt.plot([0, 1], [0, 1], linestyle='--')
    fop, mpv = calibration_curve((labels == np.argmax(np.mean(uncal_probs, axis=1), axis=1)), np.max(np.mean(uncal_probs, axis=1), axis=1), n_bins=eval_n_bins)
    plt.plot(mpv, fop, marker='x', label='uncal')
    show_bins = [1, 6]
    for i in range(max_num_bins):
        if (i+1) not in show_bins:
            continue
        probs_ = probs[:,i]
        fop, mpv = calibration_curve((labels==np.argmax(probs_,axis=1)), np.max(probs_, axis=1), n_bins=eval_n_bins)
        if i == 0:
            mk = 'v'
        else:
            mk = 'o'
        plt.plot(mpv, fop, marker=mk, label='r='+str(i+1))
    plt.ylabel('Fraction of corrects')
    plt.xlabel('Max Probability')
    #plt.title('Top-label Reliability Curve')
    plt.legend()
    plt.savefig(plot_dir+'/reliability_curves_'+args.dyn_T_factor, bbox_inches='tight')
    plt.close()
    #plot ECE/TCE to various bin_num
    ECEs_list = []
    TCEs_list = []
    bin_nums = []
    for i in range(max_num_bins):
        probs_ = probs[:,i]
        ece, cce, mce, tce = cal_multiple_ECE(probs_, labels) 
        ECEs_list.append(ece)
        TCEs_list.append(tce)
        bin_nums.append(i+1)
    print(ECEs_list)
    print(TCEs_list)
    #plt.plot(bin_nums, ECEs_list, label='ECE')
    plt.plot(bin_nums, TCEs_list) 
    plt.xlabel('Region Number')
    plt.ylabel('TCE')
    #plt.legend()
    plt.savefig(plot_dir+'/ece_tce_'+args.dyn_T_factor, bbox_inches='tight')
    plt.close()
    
    
 
    
    
