
import torch
import torchvision as tv
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models.densenet import *
from torchvision.models.resnet import *
from MLLib.models.densenet import DenseNet
from MLLib.models.densenet_2 import obtain_densenet
from MLLib.models.resnet import obtain_resnet
from MLLib.models.wide_resnet import obtain_wide_resnet
from MLLib.models.lenet import LeNet5
from MLLib.utils.utils import Meter, move_to_device, get_para_num, xavier_init_weights, kaiming_normal_init_weights
from MLLib.utils.temperature_scaling import ModelWithTemperature

densenet_names = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']
resnet_names = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']
wide_resnet_names = [ 'wide_resnet28_10', 'wide_resnet28_12', 'wide_resnet40_2']

def get_model(net_arch):
    # Get densenet configuration
    #block_config = [(net_arch['depth'] - 4) // 6 for _ in range(3)]
    if 'num_classes' in net_arch.keys():
        n_c = net_arch['num_classes']
    else:
        n_c = 100
    if 'dropout' in net_arch.keys():
        dropout_rate=net_arch['dropout']
    else:
        dropout_rate=0
    print('dropout_rate'+str(dropout_rate))
    if net_arch['model_name'] == 'densenet':
        if 'densenet_block' in net_arch.keys() and net_arch['densenet_block'] == '121':
            block_config = [6, 12, 24, 16]
            assert net_arch['num_init_features'] == 64
            assert net_arch['growth_rate'] == 32
        else:
            # in https://arxiv.org/pdf/1608.06993.pdf, designed for cifar10, cifar100, SVHN
            if (net_arch['depth'] - 4) % 3:
                raise Exception('Invalid depth')
            block_config = [(net_arch['depth'] - 4) // 6 for _ in range(3)]
    
        # Make model
        if 'num_init_features' in net_arch.keys():
            n_i_f = net_arch['num_init_features']
        else:
            n_i_f = 24
        model = DenseNet(
            growth_rate=net_arch['growth_rate'],
            block_config=block_config,
            num_classes=n_c, #net_arch['num_classes'],
            num_init_features=n_i_f, #net_arch['num_init_features']
            drop_rate=dropout_rate
        )
    elif net_arch['model_name'] == 'torch_densenet':
        if net_arch['densenet_block'] == '121':
            model = densenet121(pretrained=False,dropout_rate=dropout_rate)
    elif net_arch['model_name'] == 'lenet5':
        model = LeNet5(output_class=n_c)
    elif net_arch['model_name'] in densenet_names:
        m_n = net_arch['model_name']
        model = obtain_densenet(m_n, n_c, dropout_rate) 
    elif net_arch['model_name'] in resnet_names:
        m_n = net_arch['model_name']
        model = obtain_resnet(m_n, n_c, dropout_rate)  
    elif net_arch['model_name'] in wide_resnet_names:
        m_n = net_arch['model_name']
        model = obtain_wide_resnet(m_n, n_c, dropout_rate)
    return model

