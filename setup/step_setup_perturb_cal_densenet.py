#! /usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
try:
    import cPickle as pickle
except:
    import pickle

commandLineParser = argparse.ArgumentParser(description='Setup Models')
commandLineParser.add_argument('data_name', type=str, choices=['cifar10','cifar100'], default='cifar10',
                               help='data name')
commandLineParser.add_argument('data_path', type=str,
                               help='absolute path to data')
commandLineParser.add_argument('data_indices_path', type=str,
                               help='absolute path to train/valid/test indices to data')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path to directory location where to setup')
commandLineParser.add_argument('--seed', type=int, default=100,
                               help='Specify the global random seed')
# Model Architecture
commandLineParser.add_argument('--model_name', type=str, default='densenet', choices=['densenet', 'torch_densenet'],
                               help='model name')
commandLineParser.add_argument('--growth_rate', type=int, default=12,
                               help='growth rate, k in https://arxiv.org/pdf/1608.06993.pdf')
commandLineParser.add_argument('--depth', type=int, default=40,
                               help='depth')
commandLineParser.add_argument('--num_init_features', type=int, default=24,
                               help='the number of filters to learn in the first convolution layer')
commandLineParser.add_argument('--densenet_block', type=str, default='default', choices=['default', '121'],
                               help='densenet block config')
commandLineParser.add_argument('--num_classes', type=int, default=100,
                               help='num of output classes')
# Training Configure
commandLineParser.add_argument('--n_epochs', type=int, default=300,
                               help='training epochs number')
commandLineParser.add_argument('--batch_size', type=int, default=64,
                               help='batch size')
commandLineParser.add_argument('--valid_size', type=int, default=5000,
                               help='valid set size')
commandLineParser.add_argument('--lr', type=float, default=0.1,
                               help='learning rate')
commandLineParser.add_argument('--lr_scheduler', type=str, default='MultiStepLR', choices=['MultiStepLR', 'ReduceLROnPlateau', 'MultiStepLR_150_225_300'],
                               help='lr schedule')
commandLineParser.add_argument('--warmup', type=int, default=0,
                               help='number of epoch for warmup')
commandLineParser.add_argument('--wd', type=float, default=0.0001,
                               help='weight decay')
commandLineParser.add_argument('--momentum', type=float, default=0.9,
                               help='learning momentum')
commandLineParser.add_argument('--weight_init', type=str, default='torch_default', choices=['torch_default', 'xavier', 'kaiming'],
                               help='weight initialization method')
commandLineParser.add_argument('--loss_fn', type=str, default='CrossEntropy', choices=['CrossEntropy', 'MarginLoss'],
                               help='loss function')
commandLineParser.add_argument('--run_epoch', type=str, default='default', choices=['fast', 'default'],
                               help='fast: skip batch with zero error')
commandLineParser.add_argument('--sample_num', type=int, default=20,
                               help='number of perturbed samples')
commandLineParser.add_argument('--perturb_degree', type=float, default=5.0,
                               help='perturbation degree')
commandLineParser.add_argument('--alpha', type=float, default=0.5,
                               help='loss combining weight')
commandLineParser.add_argument('--sample_start', type=float, default=0.5,
                               help='ratio epoch before start sampling perturb')
commandLineParser.add_argument('--pt_input_ratio', type=float, default=0.25,
                               help='ratio of batch on which sampling is conducted')



def main(argv=None):
    args = commandLineParser.parse_args()

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_setup_perturb_cal_densenet.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if os.path.isdir(args.destination_dir):
        print('destination directory exists. Exiting...')
    else:
        os.makedirs(args.destination_dir)
    if not os.path.isdir(os.path.join(args.destination_dir, 'cfg')):
        os.makedirs(os.path.join(args.destination_dir, 'cfg'))

    network_architecture = dict(model_name=args.model_name,
                                data_name=args.data_name, # 
                                seed=args.seed,
                                depth=args.depth,
                                growth_rate=args.growth_rate, 
                                num_init_features=args.num_init_features,
                                densenet_block=args.densenet_block,
                                num_classes=args.num_classes 
    )
                                
    train_parameters = dict(model_name=args.model_name,
                     data_name=args.data_name,
                     data_path=args.data_path,
                     data_indices_path=args.data_indices_path,
                     seed=args.seed,
                     n_epochs=args.n_epochs,
                     batch_size=args.batch_size,
                     valid_size=args.valid_size,
                     lr=args.lr,
                     lr_scheduler=args.lr_scheduler,
                     warmup=args.warmup,
                     wd=args.wd,
                     momentum=args.momentum,
                     weight_init=args.weight_init,
                     loss_fn=args.loss_fn,
                     run_epoch=args.run_epoch,
                     sample_num=args.sample_num,
                     perturb_degree=args.perturb_degree,
                     alpha=args.alpha,
                     sample_start=args.sample_start,
                     pt_input_ratio=args.pt_input_ratio

    )  
    
    # Pickle network architecture into a file.       
    path = os.path.join(args.destination_dir, 'cfg', 'net_arch.pickle')                     
    with open(path, 'wb') as handle:                 
        pickle.dump(network_architecture, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Pickle training parameters into a file. 
    path = os.path.join(args.destination_dir, 'cfg', 'trn_para.pickle')                     
    with open(path, 'wb') as handle:                 
        pickle.dump(train_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    ## load Pickle
    #path = os.path.join(load_path, 'model/net_arch.pickle')                     
    #       # path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM/combo_finetune/model/net_arch.pickle'
    #        with open(path, 'rb') as handle:
    #            self.network_architecture = pickle.load(handle)

if __name__=='__main__':
    main()
