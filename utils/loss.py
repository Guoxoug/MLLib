
import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np

def Error_topk(logits, labels, k=1): 
    labels_ = torch.squeeze(labels)
    logits_ = torch.squeeze(logits)
    _, topk_index = torch.topk(logits_, k)
    labels_ = torch.unsqueeze(labels_, 1).expand_as(topk_index)
    return 1.0 - torch.sum(torch.eq(labels_, topk_index).float(), dim=1).mean()

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, logits, labels):
        lsm = F.log_softmax(logits, dim=1)
        return (-lsm*labels).mean()

class MarginLoss(nn.Module):
    def __init__(self):
        """
        n_bins (int): number of confidence interval bins
        """
        super(MarginLoss, self).__init__()

    def forward(self, logits, labels, epsilon=1e-8):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        error =  1.0 - predictions.eq(labels).float()
        error.requires_grad=False
        ce = F.cross_entropy(logits, labels, reduction='none')
        masked_ce = error*ce
        return masked_ce.sum()/(error.sum()+epsilon)
def softmax(logits, dim=1):
    return np.exp(logits)/np.sum(np.exp(logits), axis=dim, keepdims=True) 

def get_ll(logits, targets, **args):
    logits = logits.cpu().numpy()
    targets.cpu().numpy()
    preds = softmax(logits, dim=1)
    return np.log(1e-12 + preds[np.arange(len(targets)), targets]).mean()

def get_ece(logits, targets, n_bins=15):
    '''
    from https://github.com/bayesgroup/pytorch-ensembles/blob/master/metrics.py 
    '''
    logits = logits.cpu().numpy()
    targets = targets.cpu().numpy()
    preds = softmax(logits, dim=1)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = (predictions == targets)
    
    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prob_in_bin = np.mean(in_bin)
        if prob_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prob_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece

class TCELoss(nn.Module):
    """
    Calculates the Top Label Calibration Error of a model.

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(TCELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        tce = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prob_in_bin = in_bin.float().mean()
            if prob_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                tce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

        return tce

class ECCELoss(nn.Module):
    """
    Calculates the Expected Class Calibration Error of a model.

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECCELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ecce = torch.zeros(1, device=logits.device)
        num_classes = logits.size()[1]
        for k in range(num_classes):
            labels_k = labels.eq(torch.ones(labels.size(), device=logits.device)*k)
            #print(labels_k)
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                # Calculated |confidence - accuracy| in each bin
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item()) * labels_k
                prob_in_bin = in_bin.float().mean()
                if prob_in_bin.item() > 0:
                    accuracy_in_bin = accuracies[in_bin].float().mean()
                    avg_confidence_in_bin = confidences[in_bin].mean()
                    ecce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    
        return ecce

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        tce = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prob_in_bin = in_bin.float().mean()
            if prob_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                tce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

        return tce


class ACCELoss(nn.Module):
    """
    Calculates the All Class Calibration Error of a model.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ACCELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        #confidences, predictions_ = torch.max(softmaxes, 1)
        #confidences = torch.flatten(softmaxes)  
        confidences = softmaxes
        predictions_ = torch.arange(0,logits.size()[1], device=logits.device).unsqueeze(0).expand(logits.size())
        labels_ = labels.unsqueeze(1).expand(logits.size())
        #accuracies = predictions_.eq(labels_).flatten()
        accuracies = predictions_.eq(labels_)

        ece = torch.zeros(1, device=logits.device)
        num_classes = logits.size()[1]
        for k in range(num_classes): # calculate each class
            conf = confidences[:,k]
            accu = accuracies[:,k]
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                # Calculated |confidence - accuracy| in each bin for class k
                in_bin = conf.gt(bin_lower.item()) * conf.le(bin_upper.item())
                prob_in_bin = in_bin.float().mean()
                if prob_in_bin.item() > 0:
                    accuracy_in_bin = accu[in_bin].float().mean()
                    avg_confidence_in_bin = conf[in_bin].mean()
                    ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin * 1./num_classes
    
        return ece

class ACELoss(nn.Module):
    """
    Calculates the All Calibration Error of a model.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ACELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        #confidences, predictions_ = torch.max(softmaxes, 1)
        confidences = torch.flatten(softmaxes)  
        predictions_ = torch.arange(0,logits.size()[1], device=logits.device).unsqueeze(0).expand(logits.size())
        labels_ = labels.unsqueeze(1).expand(logits.size())
        accuracies = predictions_.eq(labels_).flatten()

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prob_in_bin = in_bin.float().mean()
            if prob_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

        return ece

def get_bin_boundaries(n_bins):
    lower = 0.0
    increment = 1.0 / n_bins
    upper = increment
    boundaries = []
    for i in range(n_bins):
        if i ==0:
            boundaries.append([-0.00001, increment])
        else:
            boundaries.append([increment*i, increment*(i+1)])
    return boundaries

def cal_bined_calibration(probs, corrects, n_bins=15):
    #all_probs = softmax(dsn_cifar100_test_logits,min_logit=50)
    #all_labels = dsn_cifar100_test_labels
    #n_bins = 15
    #probs = all_probs[:,0]
    #targets = all_labels[:,0]
    #preds = np.argmax(probs, axis=1)
    #probs = np.max(probs, axis=1)
    #corrects = (max_preds == labels)
    boundaries = get_bin_boundaries(n_bins)
    total = probs.shape[0]
    ECE = 0.0
    for b_i in boundaries:
        lower, upper = b_i
        ind1 = probs > lower
        ind2 = probs <= upper
        ind = np.where(np.logical_and(ind1, ind2))[0]
        lprobs = probs[ind]       
        lcorrects = corrects[ind]
        if lprobs.shape[0] == 0:#if np.isnan(acc):
            acc = 0.0
            prob = 0.0
        else:
            acc = np.mean(lcorrects)
            prob = np.mean(lprobs)
        ECE += np.abs(acc - prob) * float(lprobs.shape[0])
    ECE /= np.float(total)  
    return ECE



def cal_multiple_ECE(probs_, targets_, n_bins=15):
    probs = np.squeeze(probs_)
    targets = np.squeeze(targets_)
    #ECE
    max_probs = np.max(probs, axis=1)
    max_preds = np.argmax(probs, axis=1)
    max_corrects = np.equal(max_preds, targets)
    ECE = cal_bined_calibration(max_probs, max_corrects, n_bins)
    #ACE
    preds = np.repeat(np.asarray(range(probs.shape[1]))[np.newaxis], probs.shape[0], axis=0)
    labels = np.repeat(targets[:,np.newaxis], probs.shape[1], axis=1) 
    corrects = (preds==labels)
    all_probs = probs.flatten()
    all_corrects = corrects.flatten()
    ACE = cal_bined_calibration(all_probs, all_corrects, n_bins)
    #CCE
    CCEs = [cal_bined_calibration(probs[:,i], corrects[:,i], n_bins) for i in range(probs.shape[1])]
    #MCE
    MCE = np.mean(CCEs)
    return ACE, CCEs, MCE, ECE
