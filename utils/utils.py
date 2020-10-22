import numpy as np
import os
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.nn import functional as F


def onehot_convertion_torch(labels, num_classes=100):
    '''
        convert labels to onehot format
    '''
    onehot = torch.zeros([labels.shape[0],num_classes])
    allones = torch.ones([labels.shape[0],num_classes])
    return onehot.scatter_(1,labels,allones)

class PerturbInput(nn.Module):
    def __init__(self, input_size, perturb_degree=5, sample_num=20, pt_input_ratio=0.25, num_classes=100 ):
        super(PerturbInput, self).__init__()
        self.input_size=input_size
        self.perturb_degree=perturb_degree
        self.sample_num=sample_num
        self.pt_input_ratio=pt_input_ratio
        self.num_classes=num_classes
        self.pt = torch.randn([sample_num, input_size[1],input_size[2],input_size[3]])
        self.pt_batch = int(input_size[0]*pt_input_ratio)
        p_n = torch.norm(self.pt, p=2, dim=1, keepdim=True).expand_as(self.pt)
        pt = torch.div(self.pt, p_n)
        self.pt_= pt.unsqueeze_(0).view(1, sample_num, input_size[1], input_size[2], input_size[3]).expand(self.pt_batch, sample_num, input_size[1], input_size[2], input_size[3])
        

    def forward(self, input):
        torch.nn.init.normal_(self.pt)
        pt_input = input[:self.pt_batch].unsqueeze(1).expand_as(self.pt_) + self.pt_
        pt_input =torch.reshape(pt_input, (self.pt_batch*self.sample_num, self.input_size[1], self.input_size[2], self.input_size[3])) 
        return pt_input

def kaiming_normal_init_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.bias, 0)
    return net

def xavier_init_weights(net):
    """the weights of conv layer and fully connected layers 
    are both initilized with Xavier algorithm, In particular,
    we set the parameters to random values uniformly drawn from [-a, a]
    where a = sqrt(6 * (din + dout)), for batch normalization 
    layers, y=1, b=0, all bias initialized to 0.
    """
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    return net


def get_para_num(model, only_trainable=True):
    if only_trainable:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    

def move_to_device(model, device_type, print_model_to_log=False, parallel=False):

    if device_type == 'cuda':
        # Wrap model if multiple gpus
        if parallel and torch.cuda.device_count() > 1:
            model_wrapper = torch.nn.DataParallel(model).cuda()
        else:
            model_wrapper = model.cuda()
    elif device_type == 'cpu':
        model_wrapper=model.to('cpu')
    else:
        model_wrapper = None
    if print_model_to_log:
        print(model_wrapper)
    return model_wrapper 

class Meter():
    """
    A little helper class which keeps track of statistics during an epoch.
    """
    def __init__(self, name, cum=False):
        """
        name (str or iterable): name of values for the meter
            If an iterable of size n, updates require a n-Tensor
        cum (bool): is this meter for a cumulative value (e.g. time)
            or for an averaged value (e.g. loss)? - default False
        """
        self.cum = cum
        if type(name) == str:
            name = (name,)
        self.name = name

        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0

    def update(self, data, n=1):
        """
        Update the meter
        data (Tensor, or float): update value for the meter
            Size of data should match size of ``name'' in the initialized args
        """
        self._count = self._count + n
        if torch.is_tensor(data):
            self._last_value.copy_(data)
        else:
            self._last_value.fill_(data)
        self._total.add_(self._last_value)

    def value(self):
        """
        Returns the value of the meter
        """
        if self.cum:
            return self._total
        else:
            return self._total / self._count

    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])


def prob_to_logit(probs, epsilon=1e-7):
    #convert probability (along the last axis) into logits 
    probs = torch.clamp(probs, epsilon, 1-epsilon)
    logits = torch.log(probs) 
    return logits

def temperature_prob(prob, temp, multiclass=False, min_logit=0, epsilon=1e-7):
    if multiclass:
        return temperature_prob_multiclass(prob, temp, min_logit, epsilon=epsilon)
    else:
        eps = np.ones_like(prob)*epsilon
        one_eps = np.ones_like(prob)*(1-epsilon)
        #logits=np.log(prob)-np.log(1-prob)
        prob = np.minimum(np.maximum(prob, eps), one_eps)
        logits=np.log(prob) - np.log(1-prob)
        logits = logits/temp
        new_prob =1/(1+np.exp(-logits))
        return new_prob

def temperature_prob_multiclass(prob, temp, min_logit=0, epsilon=1e-6):
    #TODO more accurate inversion
    eps = np.ones_like(prob)*epsilon
    one_eps = np.ones_like(prob)*(1-epsilon)
    prob = np.minimum(np.maximum(prob, eps), one_eps)
    logits = np.log(prob) 
    logits = logits/temp
    return softmax(logits, min_logit)
