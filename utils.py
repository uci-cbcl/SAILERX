import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

RBF = False

def dice_loss(pred, target, with_logit):
    if with_logit:
        pred = torch.sigmoid(pred)
    smooth = 1e-13
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        input = torch.sigmoid(input)
        input_ = 1 - input
        logpt = (torch.stack([input_, input], 1) + 1e-13).log()
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def loss_zinb(x, mu, theta, pi, eps=1e-7):
    '''Log likelihood according to a zinb model.
    Adapted from scVI https://github.com/YosefLab/scVI
    '''
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting
    
    # mu = torch.exp(mu.clamp_max(16))
    theta = torch.exp(theta.clamp(-50, 16))

    softplus_pi = F.softplus(-pi)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)
    bool_nzero = (x > eps).type(torch.float32)
    bool_zero = (x < eps).type(torch.float32)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul(bool_zero, case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul(bool_nzero, case_non_zero)

    res = mul_case_zero.sum() + mul_case_non_zero.sum() #mul_case_zero.sum() / bool_zero.sum() + mul_case_non_zero.sum() / bool_nzero.sum()
    return -res

def apprx_kl(mu, sigma):
    '''Adapted from https://github.com/dcmoyer/invariance-tutorial/
    Function to calculate approximation for KL(q(z|x)|q(z))
        Args:
            mu: Tensor, (B, z_dim)
            sigma: Tensor, (B, z_dim)
    '''
    var = sigma.pow(2)
    var_inv = var.reciprocal()

    first = torch.matmul(var, var_inv.T)

    r = torch.matmul(mu * mu, var_inv.T)
    r2 = (mu * mu * var_inv).sum(axis=1)

    second = 2 * torch.matmul(mu, (mu * var_inv).T)
    second = r - second + (r2 * torch.ones_like(r)).T

    r3 = var.log().sum(axis=1)
    third = (r3 * torch.ones_like(r)).T - r3

    return 0.5 * (first + second + third)

def get_cos(x1, mu_2):
    x1 = F.normalize(x1, p=2)
    z2 = F.normalize(mu_2, p=2)
    c1 = torch.matmul(x1, x1.T)
    c2 = torch.matmul(z2, z2.T)
    return c1, c2
