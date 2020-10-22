#! /usr/bin/env python
from __future__ import print_function
import argparse
import os
import sys
try:
    import cPickle as pickle
except:
    import pickle
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.models.densenet import *
from torchvision.models.resnet import *
from MLLib.models.densenet import DenseNet
from MLLib.models.model_generator import get_model
from MLLib.utils.utils import Meter, move_to_device, get_para_num, xavier_init_weights, kaiming_normal_init_weights
from MLLib.utils.temperature_scaling import ModelWithTemperature
from MLLib.data_importer.ds_cifar100 import DS_cifar100
from MLLib.utils.loss import MarginLoss, SoftCrossEntropyLoss

commandLineParser = argparse.ArgumentParser(description='Train Models')
commandLineParser.add_argument('destination_dir', type=str,
                               help='absolute path to directory location where to setup')
commandLineParser.add_argument('--device_type', type=str, choices=['cpu', 'cuda'], default='cuda',
                               help='choose to run on gpu or cpu')

def run_epoch_fast(loader, model, criterion, optimizer, device_type='cuda', epoch=0, n_epochs=0, train=True, log_every_step=True):
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)
    error_meter = Meter(name='Error', cum=False)

    if train:
        model.train()
        print('Training')
    else:
        model.eval()
        print('Evaluating')

    end = time.time()
    for i, (input, target) in enumerate(loader):
        if train:
            model.zero_grad()
            optimizer.zero_grad()

            # Forward pass
            input = move_to_device(input, device_type, False)
            target = move_to_device(target, device_type, False)
            output = model(input)
            loss = criterion(output, target)

            # Backward pass
            if loss.item()>0:
                loss.backward()
                optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

        else:
            with torch.no_grad():
                # Forward pass
                input = move_to_device(input, device_type, False)
                target = move_to_device(target, device_type, False)
                output = model(input)
                loss = criterion(output, target)

        # Accounting
        _, predictions = torch.topk(output, 1)
        error = 1 - torch.eq(torch.squeeze(predictions), target).float().mean()
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        error_meter.update(error)
        if log_every_step: 
            for param_group in optimizer.param_groups:
                lr_value=param_group['lr']
            print('  '.join([
                '%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
                epoch, n_epochs, i + 1, len(loader)),
                str(time_meter),
                str(loss_meter),
                str(error_meter),
                '%.4f' % lr_value
            ]))

    if not log_every_step:
        print('  '.join([
            #'%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
            #epoch, n_epochs, i + 1, len(loader)),
            '%s: (Epoch %d of %d)' % ('Train' if train else 'Eval',
            epoch, n_epochs),
            str(time_meter),
            str(loss_meter),
            str(error_meter),
        ]))

    return time_meter.value(), loss_meter.value(), error_meter.value()

def smooth_label(label, smooth_percent, class_num=100):
    #[B, 1] -> [B, K]
    residual = (1-smooth_percent).div(class_num-1).repeat(1, class_num)
    target = smooth_percent.repeat(1, class_num)
    smoothed_label = residual.scatter_(1, label.long(), target)
    return smoothed_label

def run_epoch(loader, model, criterion, optimizer, device_type='cuda', epoch=0, n_epochs=0, train=True, log_every_step=True):
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)
    error_meter = Meter(name='Error', cum=False)

    if train:
        model.train()
        print('Training')
    else:
        model.eval()
        print('Evaluating')

    end = time.time()
    for i, (input, target) in enumerate(loader):
        if train:
            model.zero_grad()
            optimizer.zero_grad()

            # Forward pass
            input = move_to_device(input, device_type, False)
            target = move_to_device(target, device_type, False)
            output = model(input)
            loss = criterion(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

        else:
            with torch.no_grad():
                # Forward pass
                input = move_to_device(input, device_type, False)
                target = move_to_device(target, device_type, False)
                output = model(input)
                loss = criterion(output, target)

        # Accounting
        _, predictions = torch.topk(output, 1)
        error = 1 - torch.eq(torch.squeeze(predictions), target).float().mean()
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        error_meter.update(error)
        if log_every_step: 
            for param_group in optimizer.param_groups:
                lr_value=param_group['lr']
            print('  '.join([
                '%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
                epoch, n_epochs, i + 1, len(loader)),
                str(time_meter),
                str(loss_meter),
                str(error_meter),
                '%.4f' % lr_value
            ]))

    if not log_every_step:
        print('  '.join([
            #'%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
            #epoch, n_epochs, i + 1, len(loader)),
            '%s: (Epoch %d of %d)' % ('Train' if train else 'Eval',
            epoch, n_epochs),
            str(time_meter),
            str(loss_meter),
            str(error_meter),
        ]))

    return time_meter.value(), loss_meter.value(), error_meter.value()

def run_epoch_perturb(loader, model, onehot_criterion, soft_criterion, optimizer, device_type='cuda', epoch=0, n_epochs=0, train=True, class_num=100, perturb_degree=5, sample_num=20, alpha=0.5, log_every_step=True):
    time_meter = Meter(name='Time', cum=True)
    loss_meter = Meter(name='Loss', cum=False)
    error_meter = Meter(name='Error', cum=False)

    if train:
        model.train()
        print('Training')
        
    else:
        model.eval()
        print('Evaluating')

    end = time.time()
    for i, (input, target) in enumerate(loader):
        if train:
            model.zero_grad()
            optimizer.zero_grad()

            # Forward pass
            input = move_to_device(input, device_type, False)
            target = move_to_device(target, device_type, False)
            target_ = torch.unsqueeze(target, 1)
            pt_output_sum = torch.zeros_like(target_).float()
            for j in range(sample_num):
                pt = torch.randn_like(input[0])
                pt_flat = torch.flatten(pt)
                p_n = torch.norm(pt_flat, p=2)
                pt = pt.div(p_n)
                pt = pt.unsqueeze(0).expand_as(input) * perturb_degree
                pt = move_to_device(pt, device_type)
                pt_input = input + pt
                #pt_input = torch.clamp(pt_input, 0, 1) # input already normalized, TODO unnormalize first
                p_logits = model.forward(pt_input) 
                p_outputs = torch.argmax(p_logits, dim=1, keepdim=True)
                pt_output_sum = pt_output_sum + torch.eq(p_outputs, target_).float()
            pt_output_mean = torch.div(pt_output_sum, sample_num)
            pt_target = smooth_label(target_, pt_output_mean, class_num)

            output = model(input)
            onehot_loss = onehot_criterion(output, target)
            pt_target =pt_target.detach()
            perturb_loss = soft_criterion(output, pt_target)
            loss = alpha * onehot_loss + (1-alpha) * perturb_loss

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.n_iters = optimizer.n_iters + 1 if hasattr(optimizer, 'n_iters') else 1

        else:
            with torch.no_grad():
                # Forward pass
                input = move_to_device(input, device_type, False)
                target = move_to_device(target, device_type, False)
                output = model(input)
                loss = onehot_criterion(output, target)

        # Accounting
        _, predictions = torch.topk(output, 1)
        error = 1 - torch.eq(torch.squeeze(predictions), target).float().mean()
        batch_time = time.time() - end
        end = time.time()

        # Log errors
        time_meter.update(batch_time)
        loss_meter.update(loss)
        error_meter.update(error)
        if log_every_step: 
            for param_group in optimizer.param_groups:
                lr_value=param_group['lr']
            print('  '.join([
                '%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
                epoch, n_epochs, i + 1, len(loader)),
                str(time_meter),
                str(loss_meter),
                str(error_meter),
                '%.4f' % lr_value
            ]))

    print('pt_output_mean')
    print(pt_output_mean)
    print('onehot_loss:'+str(onehot_loss))
    print('perturb_loss:'+str(perturb_loss))
    if not log_every_step:
        print('  '.join([
            #'%s: (Epoch %d of %d) [%04d/%04d]' % ('Train' if train else 'Eval',
            #epoch, n_epochs, i + 1, len(loader)),
            '%s: (Epoch %d of %d)' % ('Train' if train else 'Eval',
            epoch, n_epochs),
            str(time_meter),
            str(loss_meter),
            str(error_meter),
        ]))

    return time_meter.value(), loss_meter.value(), error_meter.value()


def train(dataset, net_arch, trn_para, save, device_type='cuda', model_filename='model.pth'):
    """
    A function to train a DenseNet-BC on CIFAR-100.

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
    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)

    model = get_model(net_arch)
    if trn_para['weight_init'] == 'xavier':
        model = xavier_init_weights(model)
    elif trn_para['weight_init'] == 'kaiming':
        model = kaiming_normal_init_weights(model)
    model_wrapper = move_to_device(model, device_type, True)
    print('parameter_number: '+str(get_para_num(model_wrapper)))

    n_epochs = trn_para['n_epochs']
    if trn_para['loss_fn'] == 'MarginLoss':
        onehot_criterion = MarginLoss()
    else:
        onehot_criterion = nn.CrossEntropyLoss()
    soft_criterion = SoftCrossEntropyLoss()     

    optimizer = optim.SGD(model_wrapper.parameters(), lr=trn_para['lr'], momentum=trn_para['momentum'], nesterov=True)
    if 'lr_scheduler' in trn_para.keys() and trn_para['lr_scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20)
    elif trn_para['lr_scheduler'] == 'MultiStepLR_150_225_300':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 225, 300], gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
    
    # Make dataloaders
    train_loader = dataset.train_loader
    valid_loader = dataset.valid_loader
    
    # Train model
    best_error = 1
    for epoch in range(1, n_epochs + 1):
        if trn_para['run_epoch'] == 'fast':
            run_epoch_fast(
                loader=train_loader,
                model=model_wrapper,
                criterion=onehot_criterion,
                optimizer=optimizer,
                device_type=device_type,
                epoch=epoch,
                n_epochs=n_epochs,
                train=True,
            )
        else:
            if epoch < trn_para['sample_start'] * n_epochs:
                run_epoch(
                    loader=train_loader,                     
                    model=model_wrapper,
                    criterion=onehot_criterion,
                    optimizer=optimizer,                     
                    device_type=device_type,                 
                    epoch=epoch,
                    n_epochs=n_epochs,                       
                    train=True,
                )
            else:
                run_epoch_perturb(
                    loader=train_loader,
                    model=model_wrapper,
                    onehot_criterion=onehot_criterion,
                    soft_criterion=soft_criterion,
                    optimizer=optimizer,
                    device_type=device_type,
                    epoch=epoch,
                    n_epochs=n_epochs,
                    train=True,
                    class_num=dataset.num_classes,
                    perturb_degree=trn_para['perturb_degree'], 
                    sample_num=trn_para['sample_num'],
                    alpha=trn_para['alpha'],
                )
        valid_results = run_epoch(
            loader=valid_loader,
            model=model_wrapper,
            criterion=onehot_criterion,
            optimizer=optimizer,
            device_type=device_type,
            epoch=epoch,
            n_epochs=n_epochs,
            train=False,
        )

        # Determine if model is the best
        _, _, valid_error = valid_results
        if valid_error[0] < best_error:
            best_error = valid_error[0]
            print('New best error: %.4f' % best_error)
            torch.save(model.state_dict(), os.path.join(save, model_filename))
            with open(os.path.join(save, 'model_ckpt_detail.txt'), 'a') as ckptf:
                ckptf.write('epoch '+str(epoch)+(' reaches new best error %.4f' % best_error)+'\n')

        if 'lr_scheduler' in trn_para.keys() and trn_para['lr_scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(valid_error[0])
        else:
            scheduler.step()

    torch.save(model.state_dict(), os.path.join(save, 'last_'+model_filename))
    print('Train Done!')

def calibrate(dataset, net_arch, trn_para, save, device_type, model_filename='model.pth', calibrated_filename='model_C_ts.pth', batch_size=256):
    """
    Applies temperature scaling to a trained model.

    Takes a pretrained DenseNet-CIFAR100 model, and a validation set
    (parameterized by indices on train set).
    Applies temperature scaling, and saves a temperature scaled version.

    NB: the "save" parameter references a DIRECTORY, not a file.
    In that directory, there should be two files:
    - model.pth (model state dict)
    - valid_indices.pth (a list of indices corresponding to the validation set).

    data (str) - path to directory where data should be loaded from/downloaded
    save (str) - directory with necessary files (see above)
    """
    # Load model state dict
    model_filename = os.path.join(save, model_filename)
    if not os.path.exists(model_filename):
        raise RuntimeError('Cannot find file %s to load' % model_filename)
    state_dict = torch.load(model_filename)

    # Load original model
    orig_model = get_model(net_arch)
    orig_model=move_to_device(orig_model, device_type)
    orig_model.load_state_dict(state_dict)
    
    # data loader
    valid_loader=dataset.valid_loader
    
    # wrap the model with a decorator that adds temperature scaling
    model = ModelWithTemperature(orig_model)

    # Tune the model temperature, and save the results
    model.opt_temperature(valid_loader)
    model_filename = os.path.join(save, calibrated_filename)
    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model sved to %s' % model_filename)

if __name__ == '__main__':
    """
    Train a 40-layer DenseNet-BC on CIFAR-100

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
    with open('CMDs/step_train_perturb_cal_densenet.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    if not os.path.isdir(args.destination_dir):
        print('destination directory not exists. Exiting...')
        raise error('destination directory not exists.')

    path = os.path.join(args.destination_dir, 'cfg', 'net_arch.pickle')                     
    with open(path, 'rb') as handle:
        network_architecture = pickle.load(handle)
    
    path = os.path.join(args.destination_dir, 'cfg', 'trn_para.pickle')                     
    with open(path, 'rb') as handle:                 
        tps = pickle.load(handle) 

    # initialize loaders
    torch.manual_seed(tps['seed'])
    ds_cifar100 = DS_cifar100(tps['data_name'], tps['data_path'], tps['batch_size'], tps['valid_size'], 'cfg/data', tps['data_indices_path']) 
    # Begin train
    train(ds_cifar100, network_architecture, tps, args.destination_dir+'/outputs', args.device_type)
    # Begin calibrate
    calibrate(ds_cifar100, network_architecture, tps, args.destination_dir+'/outputs', args.device_type)
