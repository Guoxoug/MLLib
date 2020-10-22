#! /usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
try:
    import cPickle as pickle
except:
    import pickle

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2', 'wide_resnet28_10', 'wide_resnet28_12', 'wide_resnet40_2']
commandLineParser = argparse.ArgumentParser(description='Setup Models')
commandLineParser.add_argument('data_name', type=str, choices=['cifar10','cifar100','imagenet'], default='cifar10',
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
commandLineParser.add_argument('--model_name', type=str, default='resnet152', choices=__all__,
                               help='model name')
commandLineParser.add_argument('--num_classes', type=int, default=100,
                               help='num of output classes')
commandLineParser.add_argument('--dropout', type=float, default=0,
                               help='dropout rate')
# Training Configure
commandLineParser.add_argument('--n_epochs', type=int, default=300,
                               help='training epochs number')
commandLineParser.add_argument('--batch_size', type=int, default=64,
                               help='batch size')
commandLineParser.add_argument('--valid_size', type=int, default=5000,
                               help='valid set size')
commandLineParser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                               help='weight initialization method')
commandLineParser.add_argument('--lr', type=float, default=0.1,
                               help='learning rate')
commandLineParser.add_argument('--lr_scheduler', type=str, default='MultiStepLR', choices=['MultiStepLR', 'ReduceLROnPlateau', 'MultiStepLR_150_225_300', 'MultiStepLR_60_120_160_200'],
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



def main(argv=None):
    args = commandLineParser.parse_args()

    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/step_setup_resnet.cmd', 'a') as f:
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
                                num_classes=args.num_classes,
                                dropout=args.dropout 
    )
                                
    train_parameters = dict(model_name=args.model_name,
                     data_name=args.data_name,
                     data_path=args.data_path,
                     data_indices_path=args.data_indices_path,
                     seed=args.seed,
                     n_epochs=args.n_epochs,
                     batch_size=args.batch_size,
                     valid_size=args.valid_size,
                     optimizer=args.optimizer,
                     lr=args.lr,
                     lr_scheduler=args.lr_scheduler,
                     warmup=args.warmup,
                     wd=args.wd,
                     momentum=args.momentum,
                     weight_init=args.weight_init,
                     loss_fn=args.loss_fn,
                     run_epoch=args.run_epoch
    )  
    
    # Pickle network architecture into a file.       
    path = os.path.join(args.destination_dir, 'cfg', 'net_arch.pickle')                     
    with open(path, 'wb') as handle:                 
        #pickle.dump(network_architecture, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(network_architecture, handle, protocol=pickle.DEFAULT_PROTOCOL)
    # Pickle training parameters into a file. 
    path = os.path.join(args.destination_dir, 'cfg', 'trn_para.pickle')                     
    with open(path, 'wb') as handle:                 
        #pickle.dump(train_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(train_parameters, handle, protocol=pickle.DEFAULT_PROTOCOL)
        
    ## load Pickle
    #path = os.path.join(load_path, 'model/net_arch.pickle')                     
    #       # path = '/home/alta/relevance/vr311/models_min_data/baseline/ATM/combo_finetune/model/net_arch.pickle'
    #        with open(path, 'rb') as handle:
    #            self.network_architecture = pickle.load(handle)

if __name__=='__main__':
    main()
