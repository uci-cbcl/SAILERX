import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence as kl
import torch.nn.functional as F


Downsample = 1000
Downsample2 = 500

LAYER_1 = 1600
LAYER_2 = 320

SUB_1 = 64 #100 #64
SUB_2 = 32 #50 #32

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-13
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()
    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


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

    res = mul_case_zero.sum() / bool_zero.sum() + mul_case_non_zero.sum() / bool_nzero.sum()
    return -res


class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        pad = int(kernel_size/2)
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=True),
            nn.BatchNorm1d(ch_out),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class TPConv(nn.Module):
    def __init__(self, ch_in, ch_out, up_size, kernel_size):
        super(TPConv,self).__init__()
        pad = int(kernel_size/2)
        self.up = nn.Sequential(
            nn.ConvTranspose1d(ch_in, ch_out, up_size, stride=up_size),
            nn.BatchNorm1d(ch_out),
			nn.LeakyReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, kernel_size=kernel_size, stride=1, padding=pad, bias=True),
		    nn.BatchNorm1d(ch_out),
			nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x


class VAEconv(nn.Module):
    '''Standard VAE class
    '''
    def __init__(self, input_dim, z_dim):
        super(VAEconv, self).__init__()
        self.ConvEncoder = nn.Sequential(
            ConvBlock(1, 16, 101),
            nn.MaxPool1d(20, 20),
            ConvBlock(16, 32, 31),
            nn.MaxPool1d(10, 10),
            ConvBlock(32, 64, 7),
            nn.MaxPool1d(5, 5),
            nn.Conv1d(64, 1, kernel_size=1, bias=True),
            nn.BatchNorm1d(1),
            nn.LeakyReLU(inplace=True)
        )
        self.mean_enc = nn.Sequential(
            nn.Linear(int(input_dim/Downsample), z_dim)
        )
        self.var_enc = nn.Sequential(
            nn.Linear(int(input_dim/Downsample), z_dim)
        )
        self.LinearDec = nn.Sequential(
            nn.Linear(z_dim, int(input_dim/Downsample)),
            nn.BatchNorm1d(num_features=int(input_dim/Downsample)),
            nn.LeakyReLU(0.2, True)
        )
        self.ConvDecoder = nn.Sequential(
            TPConv(1, 64, 5, 7),
            TPConv(64, 32, 10, 31),
            TPConv(32, 16, 20, 101),
            nn.Conv1d(16, 1, kernel_size=101, padding=50, bias=True)
        )
    
    def forward(self, x, no_rec=False):
        out = x.unsqueeze(1)
        out = self.ConvEncoder(out)
        out = out.squeeze(1)
        # out = self.LinearEncoder(out)
        mean = self.mean_enc(out)
        log_var = self.var_enc(out)
        if no_rec:
            return mean, log_var
        else:
            z = Normal(mean, torch.exp(log_var).sqrt()).rsample()
            rec = self.LinearDec(z)
            rec = rec.unsqueeze(1)
            rec = self.ConvDecoder(rec)
            rec = rec.squeeze(1)
            return mean, log_var, z, rec


class DesnseEncoder(nn.Module):
    def __init__(self, input_dim, z_dim):
        super(DesnseEncoder, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, LAYER_1),
            nn.BatchNorm1d(num_features=LAYER_1),
            # nn.LeakyReLU(0.2, False),
            nn.PReLU(),
            nn.Linear(LAYER_1, LAYER_2),
            nn.BatchNorm1d(num_features=LAYER_2),
            # nn.LeakyReLU(0.2, False),
            nn.PReLU()
        )
        self.mean_enc = nn.Linear(LAYER_2, z_dim)
        self.var_enc = nn.Linear(LAYER_2, z_dim)

    def forward(self, x):
        out = self.Encoder(x)
        mean = self.mean_enc(out)
        log_var = self.var_enc(out)
        return mean, log_var


class DesnseEncoder3(nn.Module):
    def __init__(self, input_dim, z_dim, l_1=800, l_2=160):
        super(DesnseEncoder3, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, l_1),
            nn.BatchNorm1d(num_features=l_1),
            # nn.LeakyReLU(0.2, False),
            nn.PReLU(),
            nn.Linear(l_1, l_2),
            nn.BatchNorm1d(num_features=l_2),
            # nn.LeakyReLU(0.2, False),
            nn.PReLU()
        )
        self.mean_enc = nn.Linear(l_2, z_dim)
        self.var_enc = nn.Linear(l_2, z_dim)

    def forward(self, x):
        out = self.Encoder(x)
        mean = self.mean_enc(out)
        log_var = self.var_enc(out)
        return mean, log_var


class VAE(nn.Module):
    '''Standard VAE class
    '''
    def __init__(self, input_dim, z_dim):
        super(VAE, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.BatchNorm1d(num_features=1000),
            nn.PReLU(),
            nn.Linear(1000, 100),
            nn.BatchNorm1d(num_features=100),
            nn.PReLU()
        )
        self.mean_enc = nn.Linear(100, z_dim)
        self.var_enc = nn.Linear(100, z_dim)
        self.Decoder = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.BatchNorm1d(num_features=100),
            # nn.LeakyReLU(0.2, True),
            nn.PReLU(),
            nn.Linear(100, 1000),
            nn.BatchNorm1d(num_features=1000),
            nn.PReLU(),
            nn.Linear(1000, input_dim),
        )
    
    def forward(self, x, no_rec=False):
        if no_rec:
            out = self.Encoder(x)
            mean = self.mean_enc(out)
            log_var = self.var_enc(out)
            return mean, log_var
        else:
            out = self.Encoder(x)
            mean = self.mean_enc(out)
            log_var = self.var_enc(out)
            z = Normal(mean, torch.exp(log_var)).rsample()
            rec = self.Decoder(z)
            return mean, log_var, z, rec


class VAE2(nn.Module):
    '''Conditional VAE
    '''
    def __init__(self, input_dim, z_dim, batch=False):
        super(VAE2, self).__init__()
        if batch:
            c = 2
        else:
            c = 1
        self.Encoder = DesnseEncoder(input_dim, z_dim)
        self.Decoder = nn.Sequential(
            nn.Linear(z_dim+c, LAYER_2),
            nn.BatchNorm1d(num_features=LAYER_2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(LAYER_2, LAYER_1),
            nn.BatchNorm1d(num_features=LAYER_1),
            nn.LeakyReLU(0.2, True),
            nn.Linear(LAYER_1, input_dim),
        )
    
    def forward(self, x, l, b=None, no_rec=False):
        if no_rec:
            mean, log_var = self.Encoder(x)
            return mean, log_var
        else:
            if b is not None:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l, b), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec
            else:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec


class VAE3(nn.Module):
    '''Conditional VAE for RNA
    '''
    def __init__(self, input_dim, z_dim, batch=False, l_1=1600, l_2=320):
        super(VAE3, self).__init__()
        if batch:
            c = 2
        else:
            c = 1
        self.Encoder = DesnseEncoder3(input_dim, z_dim)
        self.Decoder = nn.Sequential(
            nn.Linear(z_dim+c, l_2),
            nn.BatchNorm1d(num_features=l_2),
            # nn.LeakyReLU(0.2, True),
            nn.PReLU(),
            nn.Linear(l_2, l_1),
            nn.BatchNorm1d(num_features=l_1),
            # nn.LeakyReLU(0.2, True),
            nn.PReLU(),
            nn.Linear(l_1, input_dim),
        )
    
    def forward(self, x, l, b=None, no_rec=False):
        if no_rec:
            mean, log_var = self.Encoder(x)
            return mean, log_var
        else:
            if b is not None:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l, b), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec
            else:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec


class SplitEnc(nn.Module):
    ''' Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''
    def __init__(self, input_dim, z_dim):
        super(SplitEnc, self).__init__()
        self.input_dim = input_dim
        self.split_layer = nn.ModuleList()
        for n in self.input_dim:
            # assert isinstance(n, int)
            layer1 = nn.Linear(n, SUB_1)
            nn.init.xavier_uniform_(layer1.weight)
            bn1 = nn.BatchNorm1d(SUB_1)
            act1 = nn.PReLU()
            layer2 = nn.Linear(SUB_1, SUB_2)
            nn.init.xavier_uniform_(layer2.weight)
            bn2 = nn.BatchNorm1d(SUB_2)
            act2 = nn.PReLU()
            self.split_layer.append(
                nn.ModuleList([layer1, bn1, act1, layer2, bn2, act2])
            )
        self.mean_enc = nn.Linear(SUB_2*len(input_dim), z_dim)
        self.var_enc = nn.Linear(SUB_2*len(input_dim), z_dim)
        nn.init.xavier_uniform_(self.mean_enc.weight)
        nn.init.xavier_uniform_(self.var_enc.weight)

    def forward(self, x):
        xs = torch.split(x, self.input_dim, dim=1)
        assert len(xs) == len(self.input_dim)
        enc_chroms = []
        for init_mod, chrom_input in zip(self.split_layer, xs):
            for f in init_mod:
                chrom_input = f(chrom_input)
            enc_chroms.append(chrom_input)
        enc1 = torch.cat(enc_chroms, dim=1)
        mean = self.mean_enc(enc1)
        log_var = self.var_enc(enc1)
        return mean, log_var


class SplitDec(nn.Module):
    ''' Addapted from https://github.com/wukevin/babel/blob/main/babel/models/autoencoders.py
    '''
    def __init__(self, input_dim, z_dim):
        super(SplitDec, self).__init__()
        self.input_dim = input_dim
        self.dec1 = nn.Linear(z_dim, len(self.input_dim) * SUB_2)
        self.bn1 = nn.BatchNorm1d(len(self.input_dim) * SUB_2)
        self.act1 = nn.PReLU()
        self.split_layer = nn.ModuleList()
        for n in self.input_dim:
            # assert isinstance(n, int)
            layer1 = nn.Linear(SUB_2, SUB_1)
            nn.init.xavier_uniform_(layer1.weight)
            bn1 = nn.BatchNorm1d(SUB_1)
            act1 = nn.PReLU()
            layer2 = nn.Linear(SUB_1, n)
            nn.init.xavier_uniform_(layer2.weight)
            self.split_layer.append(
                nn.ModuleList([layer1, bn1, act1, layer2])
            )

    def forward(self, x):
        x = self.act1(self.bn1(self.dec1(x)))
        xs = torch.chunk(x, chunks=len(self.input_dim), dim=1)
        rec_chroms = []
        for init_mod, chrom_input in zip(self.split_layer, xs):
            for f in init_mod:
                chrom_input = f(chrom_input)
            rec_chroms.append(chrom_input)
        rec = torch.cat(rec_chroms, dim=1)
        return rec


class VAESplit(nn.Module):
    '''Conditional VAE with split chrom
    '''
    def __init__(self, input_dim, z_dim, batch=False):
        super(VAESplit, self).__init__()
        if batch:
            c = 2
        else:
            c = 1
        self.input_dim = input_dim
        self.Encoder = SplitEnc(input_dim, z_dim)
        self.Decoder = SplitDec(input_dim, z_dim+c)
    
    def forward(self, x, l, b=None, no_rec=False):
        if no_rec:
            mean, log_var = self.Encoder(x)
            return mean, log_var
        else:
            if b is not None:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l, b), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec
            else:
                mean, log_var = self.Encoder(x)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, l), 1)
                rec = self.Decoder(z_c)
                return mean, log_var, z, rec


class VAEInv(nn.Module):
    def __init__(self, vae):
        super(VAEInv, self).__init__()
        self.vae = vae

    def forward(self, x, l, b=None, no_rec=False):
        if no_rec:
            mean, log_var = self.vae.forward(x, l, no_rec=True)
            return mean, log_var
        else:
            if b is not None:
                mean, log_var, z, rec = self.vae(x, l, b)
            else:
                mean, log_var, z, rec = self.vae(x, l)
            return mean, log_var, z, rec


class VAECount(nn.Module):
    def __init__(self, input_dim, z_dim, batch=False, L_1=1000, L_2=100):
        super(VAECount, self).__init__()
        if batch:
            c = 2
        else:
            c = 1
        self.Encoder = nn.Sequential(
            nn.Linear(input_dim, L_1),
            nn.BatchNorm1d(num_features=L_1),
            nn.PReLU(),
            nn.Linear(L_1, L_2),
            nn.BatchNorm1d(num_features=L_2),
            nn.PReLU()
        )
        self.mean_enc = nn.Linear(L_2, z_dim)
        self.var_enc = nn.Linear(L_2, z_dim)
        self.Decoder = nn.Sequential(
            nn.Linear(z_dim+c, L_2),
            nn.BatchNorm1d(num_features=L_2),
            nn.PReLU(),
            nn.Linear(L_2, L_1),
            nn.BatchNorm1d(num_features=L_1),
            nn.PReLU()
        )
        self.rho_dec = nn.Sequential(nn.Linear(L_1, input_dim),
                                nn.Softmax(dim=-1))
        self.theta_dec = nn.Linear(L_1, input_dim)
        self.dropout_dec = nn.Linear(L_1, input_dim)

    def forward(self, x, dx, b=None, no_rec=False):
        if no_rec:
            out = self.Encoder(x)
            mean = self.mean_enc(out)
            log_var = self.var_enc(out)
            return mean, log_var
        else:
            if b is not None:
                out = self.Encoder(x)
                mean = self.mean_enc(out)
                log_var = self.var_enc(out)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, dx, b), 1)
                out2 = self.Decoder(z_c)
                rate = self.rho_dec(out2)
                dropout = self.dropout_dec(out2)
                theta = self.theta_dec(out2)
            else:
                out = self.Encoder(x)
                mean = self.mean_enc(out)
                log_var = self.var_enc(out)
                z = Normal(mean, torch.exp(log_var)).rsample()
                z_c = torch.cat((z, dx), 1)
                out2 = self.Decoder(z_c)
                rate = self.rho_dec(out2)
                dropout = self.dropout_dec(out2)
                theta = self.theta_dec(out2)
            return mean, log_var, z, theta, rate, dropout


class AlignedVAE(nn.Module):
    '''
    '''
    def __init__(self, vae_x, vae_y):
        super(AlignedVAE, self).__init__()
        self.vae_x = vae_x # VAECount class
        self.vae_y = vae_y # VAE2

    def forward(self, x, y, dx, dy, b=None, no_rec=False):
        if no_rec:
            x_mean, _ = self.vae_x(x, dx, no_rec=True)
            y_mean, _ = self.vae_y(y, dy, no_rec=True)
            return x_mean, y_mean
        else:
            if b is not None:
                x_mean, x_lvar, z_x, theta, rate, dropout = self.vae_x(x, dx, b)
                y_mean, y_lvar, z_y, y_rec = self.vae_y(y, dy, b)
                return (x_mean, x_lvar, z_x, theta, rate, dropout), (y_mean, y_lvar, z_y, y_rec)
            else:
                x_mean, x_lvar, z_x, theta, rate, dropout = self.vae_x(x, dx)
                y_mean, y_lvar, z_y, y_rec = self.vae_y(y, dy)
                return (x_mean, x_lvar, z_x, theta, rate, dropout), (y_mean, y_lvar, z_y, y_rec)


class NormAlignedVAE(nn.Module):
    '''
    '''
    def __init__(self, vae_x, vae_y):
        super(NormAlignedVAE, self).__init__()
        self.vae_x = vae_x # VAE2
        self.vae_y = vae_y # VAE2

    def forward(self, x, y, dx, dy, b=None, no_rec=False):
        if no_rec:
            x_mean, _ = self.vae_x(x, dx, no_rec=True)
            y_mean, _ = self.vae_y(y, dy, no_rec=True)
            return x_mean, y_mean
        else:
            if b is not None:
                x_mean, x_lvar, z_x, x_rec = self.vae_x(x, dx, b)
                y_mean, y_lvar, z_y, y_rec = self.vae_y(y, dy, b)
                return (x_mean, x_lvar, z_x, x_rec), (y_mean, y_lvar, z_y, y_rec)
            else:
                x_mean, x_lvar, z_x, x_rec = self.vae_x(x, dx)
                y_mean, y_lvar, z_y, y_rec = self.vae_y(y, dy)
                return (x_mean, x_lvar, z_x, x_rec), (y_mean, y_lvar, z_y, y_rec)
