
import os
import time
import torch
import torchvision as tv
from torch import nn, optim
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


class Data():
    """
    A base class of data importers
    """ 
    def __init__(self, name='data'):
        self.name=name
        self.mean=0
        self.stdv=1
        self.train_indices=[0]
        self.valid_indices=[0]
        self.test_indices=[0]
        self.batch_size=64
        
    
    def data_mean(self):
        return self.mean
    
    def data_stdv(self):
        return self.stdv 
        
     
class DS_cifar100(Data):
    
    def __init__(self, data_name, data_path, batch_size, valid_size=0, save_path='./data', load_path=None, download=False, num_workers=1):
        Data.__init__(self, data_name)
        self.mean=[0.5071, 0.4867, 0.4408]  
        self.stdv=[0.2675, 0.2565, 0.2761]    
        self.data_path=data_path
        self.batch_size=batch_size
        self.valid_size=valid_size
        self.download=download
        self.num_classes=100
        self.num_workers=num_workers
        
        # Data transforms
        mean = self.mean 
        stdv = self.stdv 
        train_transforms = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])
        test_transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        # Split training into train and validation - needed for calibration
        #
        # IMPORTANT! We need to use the same validation set for temperature
        # scaling, so we're going to save the indices for later
        train_set = tv.datasets.CIFAR100(data_path, train=True, transform=train_transforms, download=self.download)
        valid_set = tv.datasets.CIFAR100(data_path, train=True, transform=test_transforms, download=False)
        test_set = tv.datasets.CIFAR100(data_path, train=False, transform=test_transforms, download=False)
        if self.valid_size == 0:
            print('valid size = 0..full train set')
            self.train_indices = torch.randperm(len(train_set))
            
        else:
            if load_path is not None: # use data split already done in load_path, using train_indices.pth, valid_indices.pth, test_indices.pth
                self.train_indices = torch.load(os.path.join(load_path, 'train_indices.pth'))
                self.valid_indices = torch.load(os.path.join(load_path, 'valid_indices.pth'))
                self.test_indices = torch.load(os.path.join(load_path, 'test_indices.pth'))
            else: # Do new data split
                indices = torch.randperm(len(train_set))
                self.train_indices = indices[:len(indices) - self.valid_size]
                self.valid_indices = indices[len(indices) - self.valid_size:] 
                self.test_indices = torch.randperm(len(test_set))
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                torch.save(self.train_indices, os.path.join(save_path, 'train_indices.pth'))
                torch.save(self.valid_indices, os.path.join(save_path, 'valid_indices.pth'))
                torch.save(self.test_indices, os.path.join(save_path, 'test_indices.pth'))
        # Make dataloaders
        self.train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers,
                                               sampler=SubsetRandomSampler(self.train_indices))
        if self.valid_size == 0:
            print('valid size = 0..test_set as valid set')
            self.valid_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=self.batch_size)
        else:
            self.valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers,
                                               sampler=SubsetRandomSampler(self.valid_indices))
        #self.test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=self.batch_size,
        #                                       sampler=SubsetRandomSampler(self.test_indices))
        self.test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=self.batch_size)
        test_train_set = torch.utils.data.Subset(valid_set, self.train_indices)
        self.test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size=self.batch_size)
        test_valid_set = torch.utils.data.Subset(valid_set, self.valid_indices)
        self.test_valid_loader = torch.utils.data.DataLoader(test_valid_set, batch_size=self.batch_size)
    

class DS_cifar10(Data):
    
    def __init__(self, data_name, data_path, batch_size, valid_size=0, save_path='./data', load_path=None, download=False, num_workers=1):
        Data.__init__(self, data_name)
        self.mean=[0.4914, 0.4823, 0.4465]  
        self.stdv=[0.247, 0.243, 0.261]    
        self.data_path=data_path
        self.batch_size=batch_size
        self.valid_size=valid_size
        self.download=download
        self.num_classes=10
        self.num_workers=num_workers
        
        # Data transforms
        mean = self.mean 
        stdv = self.stdv 
        train_transforms = tv.transforms.Compose([
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])
        test_transforms = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        # Split training into train and validation - needed for calibration
        #
        # IMPORTANT! We need to use the same validation set for temperature
        # scaling, so we're going to save the indices for later
        train_set = tv.datasets.CIFAR10(data_path, train=True, transform=train_transforms, download=self.download)
        valid_set = tv.datasets.CIFAR10(data_path, train=True, transform=test_transforms, download=False)
        test_set = tv.datasets.CIFAR10(data_path, train=False, transform=test_transforms, download=False)
        if self.valid_size == 0:
            print('valid size = 0..full train set')
            self.train_indices = torch.randperm(len(train_set))
        else:
            
            if load_path is not None: # use data split already done in load_path, using train_indices.pth, valid_indices.pth, test_indices.pth
                self.train_indices = torch.load(os.path.join(load_path, 'train_indices.pth'))
                self.valid_indices = torch.load(os.path.join(load_path, 'valid_indices.pth'))
                self.test_indices = torch.load(os.path.join(load_path, 'test_indices.pth'))
            else: # Do new data split
                indices = torch.randperm(len(train_set))
                self.train_indices = indices[:len(indices) - self.valid_size]
                self.valid_indices = indices[len(indices) - self.valid_size:] 
                self.test_indices = torch.randperm(len(test_set))
                if not os.path.isdir(save_path):
                    os.mkdir(save_path)
                torch.save(self.train_indices, os.path.join(save_path, 'train_indices.pth'))
                torch.save(self.valid_indices, os.path.join(save_path, 'valid_indices.pth'))
                torch.save(self.test_indices, os.path.join(save_path, 'test_indices.pth'))
         
        # Make dataloaders
        self.train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers,
                                               sampler=SubsetRandomSampler(self.train_indices))
        if self.valid_size == 0:
            print('valid size = 0..test_set as valid set')
            self.valid_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=self.batch_size)
        else:
            self.valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers,
                                               sampler=SubsetRandomSampler(self.valid_indices))
        
        #self.test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=self.batch_size,
        #                                       sampler=SubsetRandomSampler(self.test_indices))
        self.test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=self.batch_size)
        test_train_set = torch.utils.data.Subset(valid_set, self.train_indices)
        self.test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size=self.batch_size)
        test_valid_set = torch.utils.data.Subset(valid_set, self.valid_indices)
        self.test_valid_loader = torch.utils.data.DataLoader(test_valid_set, batch_size=self.batch_size)
    
class DS_imagenet(Data):
    
    def __init__(self, data_name, data_path, batch_size, valid_size=0, save_path='./data', load_path=None, download=False, num_workers=1):
        Data.__init__(self, data_name)
        self.mean=[0.485, 0.456, 0.406]
        self.stdv=[0.229, 0.224, 0.225]    
        self.data_path=data_path
        self.batch_size=batch_size
        self.valid_size=valid_size
        self.download=download
        self.num_classes=10
        self.num_workers=num_workers
        
        # Data transforms
        mean = self.mean 
        stdv = self.stdv 
        train_transforms = tv.transforms.Compose([
            tv.transforms.RandomResizedCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])
        test_transforms = tv.transforms.Compose([
            tv.transforms.Resize(256),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=mean, std=stdv),
        ])

        # Split training into train and validation - needed for calibration
        #
        # IMPORTANT! We need to use the same validation set for temperature
        # scaling, so we're going to save the indices for later
        train_set = tv.datasets.ImageNet(data_path, split='train', transform=train_transforms, download=self.download)
        valid_set = tv.datasets.ImageNet(data_path, split='train', transform=test_transforms, download=False)
        test_set = tv.datasets.ImageNet(data_path, split='val', transform=test_transforms, download=False)
        if load_path is not None: # use data split already done in load_path, using train_indices.pth, valid_indices.pth, test_indices.pth
            self.train_indices = torch.load(os.path.join(load_path, 'train_indices.pth'))
            self.valid_indices = torch.load(os.path.join(load_path, 'valid_indices.pth'))
            self.test_indices = torch.load(os.path.join(load_path, 'test_indices.pth'))
        else: # Do new data split
            indices = torch.randperm(len(train_set))
            print('train_set, test_set')
            print(len(train_set))
            print(len(test_set))
            train_size = 1300000
            self.train_indices = indices[:train_size]
            self.valid_indices = indices[train_size:train_size+self.valid_size:] 
            self.test_indices = torch.randperm(len(test_set))
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            torch.save(self.train_indices, os.path.join(save_path, 'train_indices.pth'))
            torch.save(self.valid_indices, os.path.join(save_path, 'valid_indices.pth'))
            torch.save(self.test_indices, os.path.join(save_path, 'test_indices.pth'))
        # Make dataloaders
        self.train_loader = torch.utils.data.DataLoader(train_set, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers,
                                               sampler=SubsetRandomSampler(self.train_indices))
        self.valid_loader = torch.utils.data.DataLoader(valid_set, pin_memory=True, batch_size=self.batch_size, num_workers=self.num_workers,
                                               sampler=SubsetRandomSampler(self.valid_indices))
        #self.test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=self.batch_size,
        #                                       sampler=SubsetRandomSampler(self.test_indices))
        self.test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=self.batch_size)
        test_train_set = torch.utils.data.Subset(valid_set, self.train_indices)
        self.test_train_loader = torch.utils.data.DataLoader(test_train_set, batch_size=self.batch_size)
        test_valid_set = torch.utils.data.Subset(valid_set, self.valid_indices)
        self.test_valid_loader = torch.utils.data.DataLoader(test_valid_set, batch_size=self.batch_size)
    

def get_dataset(data_name, data_path, batch_size, valid_size=0, save_path='./data', load_path=None):
    if data_name == 'cifar100':
        return DS_cifar100(data_name, data_path, batch_size, valid_size, save_path, load_path)
    elif data_name == 'cifar10':
        return DS_cifar10(data_name, data_path, batch_size, valid_size, save_path, load_path)
    elif data_name == 'imagenet':
        return DS_imagenet(data_name, data_path, batch_size, valid_size, save_path, load_path)
