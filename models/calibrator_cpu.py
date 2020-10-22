import torch
from torch import nn, optim
from torch.nn import functional as F
#from MLLib.utils.loss import *
from MLLib.utils.loss_2 import *
from MLLib.utils.utils import prob_to_logit
from sklearn.metrics import roc_auc_score, roc_curve

__calibration_methods__ = ['sig_temp', 'ens_temp_bef', 'ens_temp_aft', 'sig_platt', 'ens_platt_bef', 'ens_platt_aft']

class Calibrator(nn.Module):
    """
    The basic class that do the calibration based on model's predicted logits 
    cal_method: 'temperature_annealing', 'platt_scaling'
    num_members: when calibrating ensemble using before mode, need to specify member numbers
        NB: inputs should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, cal_method=0, num_members = 10, factor_bin_num=1):
        super(Calibrator, self).__init__()
        self.calibration_method = __calibration_methods__[cal_method]
        self.num_members = num_members
        self.factor_bin_num = 1
        self.init_temperature = 1.
        if self.calibration_method == 'sig_temp' or self.calibration_method == 'ens_temp_aft': # one temperature value for all
            self.temperature = nn.Parameter(torch.ones([1, factor_bin_num])*self.init_temperature)
        elif self.calibration_method == 'ens_temp_bef':
            self.temperature = nn.Parameter(torch.ones([self.num_members, factor_bin_num])*self.init_temperature)

    def forward(self, logits):
        pass

    def cal_logits(self, logits, factor_mask):
        if self.calibration_method in ['sig_temp', 'ens_temp_aft']:
            #expand_T = torch.matmul(factor_mask, torch.transpose(self.temperature,0,1)).expand(logits.size())
            #predicted_logits = logits / expand_T
            return self.temperature_scale(logits, factor_mask)
        elif self.calibration_method == 'ens_temp_bef':
            #expand_T = torch.matmul(factor_mask, torch.transpose(self.temperature,0,1)).unsqueeze(2).expand(logits.size())
            predicted_logits = prob_to_logit(torch.mean(F.softmax(self.temperature_scale(logits, factor_mask), dim=2), dim=1))
        return predicted_logits

    def temperature_scale(self, logits, factor_mask): # return member logits
        """
        Perform temperature scaling on logits
        """
        if self.calibration_method in ['sig_temp', 'ens_temp_aft']:
            # Expand temperature to match the size of logits
            #t = self.temperature.unsqueeze(1).expand(logits.size())
            t = torch.matmul(factor_mask, torch.transpose(self.temperature,0,1)).expand(logits.size())
            return logits / t
        elif self.calibration_method == 'ens_temp_bef':
            t = torch.matmul(factor_mask, torch.transpose(self.temperature,0,1)).unsqueeze(2).expand(logits.size())
            #t = self.temperature.unsqueeze(0).unsqueeze(2).expand(logits.size())
            return logits / t

    def eval_(self, logits, labels):
        if self.calibration_method in ['ens_temp_bef'] and len(logits.shape)>2:
            probs = torch.mean(F.softmax(logits, dim=2), dim=1)
            logits_ = prob_to_logit(probs)
        else:
            logits_ = logits

        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = ECELoss()
        tce_criterion = TCELoss()
        ecce_criterion = ECCELoss()
        ace_criterion = ACELoss() 
        acce_criterion = ACCELoss()
        
        nll = nll_criterion(logits_, labels).item()
        ece = ece_criterion(logits_, labels).item()
        tce = tce_criterion(logits_, labels).item()
        ecce = ecce_criterion(logits_, labels).item()
        ace = ace_criterion(logits_, labels).item()
        acce = acce_criterion(logits_, labels).item() 
        error_1 = Error_topk(logits_, labels, 1).item()
        error_5 = Error_topk(logits_, labels, 5).item()
        return ('ACC_1:%.4f, ERR_1:%.4f, ERR_5:%.4f, NLL:%.4f, ACCE(e-4):%.4f, ACE(e-4):%.4f, ECCE(e-2):%.4f, ECE(e-2):%.4f, TCE(e-2):%.4f\n %.2f & %.4f & %.2f & %.2f & %.2f & %.2f' % (1-error_1, error_1, error_5, nll, acce*1e4, ace*1e4, ecce*1e2, ece*1e2, tce*1e2, 100*(1-error_1), nll, acce*1e4, ace*1e4, ecce*1e2, ece*1e2))
        #return ('ACC_1:%.4f, ERR_1:%.4f, ERR_5:%.4f, NLL:%.4f, ACCE(e-4):%.4f, ACE(e-4):%.4f, ECCE(e-2):%.4f, ECE(e-2):%.4f, TCE(e-2):%.4f\n %.2f & %.4f & %.2f & %.2f' % (1-error_1, error_1, error_5, nll, acce*1e4, ace*1e4, ecce*1e2, ece*1e2, tce*1e2, 100*(1-error_1), nll, ace*1e4, ece*1e2))
        
    def nll_prob_label(prob, label):#[B, C], [B, ]
        ind = torch.cat((torch.arange(0,prob.shape[1]).unsqueeze(1), label.unsqueeze(1)),dim=1)
        selected_prob = torch.gather(prob, 1, ind) 
        
    def opt_calibration(self, valid_logits, valid_labels, valid_masks, opt_target, opt_method, num_iters, learning_rate, verbose = False):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """

        # Calculate NLL and ECE before temperature scaling
        
        if verbose:
            result = self.eval_(valid_logits, valid_labels)
            print('Before Calibration on Valid Set - '+result)

        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = ECELoss()
        tce_criterion = TCELoss()
        acce_criterion = ACCELoss()
        ecce_criterion = ECCELoss()
        NLL_criterion = nn.NLLLoss()
        def opt_loss(opt_target, logits, labels):
            if opt_target == 'NLL':
                return nll_criterion(logits, labels)
            elif opt_target == 'TCE':
                return tce_criterion(logits, labels)
            elif opt_target == 'ECE':
                return ece_criterion(logits, labels)
            elif opt_target == 'ACCE':
                return acce_criterion(logits, labels)
            elif opt_target == 'ECCE':
                return ecce_criterion(logits, labels)
            else:
                raise Exception('Not implemented loss: '+opt_target)
        # Next: optimize the temperature w.r.t. NLL
        if opt_method == 'LBFGS': 
            optimizer = optim.LBFGS([self.temperature], lr=learning_rate, max_iter=num_iters)
            def eval():
                loss = opt_loss(opt_target, self.cal_logits(valid_logits, valid_masks), valid_labels)
                loss.backward()
                return loss
            optimizer.step(eval)
        elif opt_method == 'SGD':
            optimizer = optim.SGD([self.temperature], lr=learning_rate)
            for tit in range(num_iters):
                predicted_logits = self.cal_logits(valid_logits, valid_masks)
                optimizer.zero_grad()
                loss = opt_loss(opt_target, predicted_logits, valid_labels)
                loss.backward()
                optimizer.step()
                #print(loss)
                #print(self.temperature)
        else:
            raise Exception('Not implemented optimizer: '+opt_method)
        if verbose:
            # Calculate NLL and ECE after calibration
            if self.calibration_method in ['sig_temp','ens_temp_bef','ens_temp_aft']:
                print('Optimal temperature: ' +str(self.temperature))
            predicted_logits = self.cal_logits(valid_logits, valid_masks)
            result = self.eval_(predicted_logits, valid_labels)
            print('After Calibration on Valid Set - '+result)



