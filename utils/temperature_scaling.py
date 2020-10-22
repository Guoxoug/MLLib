import torch
from torch import nn, optim
from torch.nn import functional as F
from MLLib.utils.loss import TCELoss, ECELoss, Error_topk


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def set_temperature(self, t):
        self.tempearature = t
    # This function probably should live outside of this class, but whatever
    def opt_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()
        tce_criterion = TCELoss().cuda()
        self.model.eval()
        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        before_temperature_tce = tce_criterion(logits, labels).item()
        before_error_1 = Error_topk(logits, labels, 1).item()
        before_error_5 = Error_topk(logits, labels, 5).item()
        print('Before temperature - ERR_1: %.4f, ERR_5: %.4f, NLL: %.4f, ECE: %.4f, TCE: %.4f' % (before_error_1, before_error_5, before_temperature_nll, before_temperature_ece, before_temperature_tce))

        # Next: optimize the temperature w.r.t. NLL
        ## Fix network weights
        for param in self.model.parameters(): 
            param.requires_grad = False
        #optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        #def eval():
        #    loss = nll_criterion(self.temperature_scale(logits), labels)
        #    loss.backward()
        #    return loss
        #optimizer.step(eval)
        optimizer = optim.SGD([self.temperature], lr=0.01, momentum=0.9)
        for tit in range(50):
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            optimizer.step()
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_tce = tce_criterion(self.temperature_scale(logits), labels).item()
        after_error_1 = Error_topk(self.temperature_scale(logits), labels, 1).item()
        after_error_5 = Error_topk(self.temperature_scale(logits), labels, 5).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - ERR_1: %.4f, ERR_5: %.4f, NLL: %.4f, ECE: %.4f, TCE: %.4f' % (after_error_1, after_error_5, after_temperature_nll, after_temperature_ece, after_temperature_tce))

        return self


