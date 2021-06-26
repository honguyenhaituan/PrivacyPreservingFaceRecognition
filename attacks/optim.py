import operator
import torch
from torch.optim import Adadelta, Adagrad, Adam, AdamW, Adamax, ASGD, RMSprop, Rprop, SGD
from torch.optim.optimizer import Optimizer

class I_FGSM: 
    def __init__(self, params, epsilon=20): 
        self.params = params
        self.epsilon = epsilon / 255
        self.alpha = 1 / 255
        self.updated_params = []
        for param in self.params:
            self.updated_params.append(torch.zeros_like(param))

    @torch.no_grad()
    def _cal_update(self, idx):
        return -self.alpha * torch.sign(self.params[idx].grad)

    @torch.no_grad()
    def step(self):
        for idx, (param, updated_param) in enumerate(zip(self.params, self.updated_params)):
            if param is None: 
                continue
    
            n_update = torch.clamp(updated_param + self._cal_update(idx), -self.epsilon, self.epsilon)
            update = n_update - updated_param
            n_param = torch.clamp(param + update, 0, 1)
            update = n_param - param

            param += update
            updated_param += update

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

class MI_FGSM(I_FGSM):
    def __init__(self, params, epsilon=20, momemtum=0):
        super(MI_FGSM, self).__init__(params, epsilon)
        self.momentum = momemtum
        self.o_grad = []
        for param in self.params:
            self.o_grad.append(torch.zeros_like(param))

    @torch.no_grad()
    def _cal_update(self, idx):
        grad = self.o_grad[idx] * self.momentum + self.params[idx].grad / torch.sum(torch.abs(self.params[idx].grad))
        return -self.alpha * torch.sign(grad)

    def zero_grad(self):
        for o_grad, param in zip(self.o_grad, self.params):
            if param.grad is not None:
                o_grad = o_grad * self.momentum + param.grad / torch.sum(torch.abs(param.grad))
        super().zero_grad()

class WrapOptim: 
    @torch.no_grad()
    def __init__(self, params, epsilon, optimizer:Optimizer):
        self.optim = optimizer
        self.params = params
        self.epsilon = epsilon / 255
        self.params_init = []
        for param in params:
            self.params_init.append(param.clone())

    @torch.no_grad()
    def step(self):
        self.optim.step()
        for param, param_init in zip(self.params, self.params_init):
            total_update = param - param_init
            update = torch.clamp(total_update, -self.epsilon, self.epsilon)

            param += update - total_update
            param.clamp_(0, 1)
    
    def zero_grad(self):
        self.optim.zero_grad()

def get_optim(opt, params) -> I_FGSM:
    if opt.name_attack == 'I-FGSM': 
        return I_FGSM(params, opt.epsilon)
    if opt.name_attack == 'MI-FGSM':
        return MI_FGSM(params, opt.epsilon, opt.momentum)

    optimizer = None
    if opt.name_attack == 'Adadelta':
        optimizer = Adadelta(params)
    if opt.name_attack == 'Adagrad':
        optimizer = Adagrad(params)
    if opt.name_attack == 'Adam':
        optimizer = Adam(params)
    if opt.name_attack == 'AdamW':
        optimizer = AdamW(params)
    if opt.name_attack == 'Adamax':
        optimizer = Adamax(params)
    if opt.name_attack == 'ASGD':
        optimizer = ASGD(params)
    if opt.name_attack == 'RMSprop':
        optimizer = RMSprop(params, lr=0.004)
    if opt.name_attack == 'Rprop':
        optimizer = Rprop(params)
    if opt.name_attack == 'SGD':
        optimizer = SGD(params, lr=0.1, momentum=opt.momentum)
    
    if optimizer:
        return WrapOptim(params, opt.epsilon, optimizer)

    return None