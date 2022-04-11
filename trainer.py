from pickle import FALSE
import numpy as np
import pandas as pd 

import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence as kl
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR

from dataset import ChSplitDS
from model import VAE2, VAECount, AlignedVAE, NormAlignedVAE, VAEInv, VAESplit

from utils import apprx_kl, get_cos


WARM_UP = 10
CYCLE = 100
STOP_CYCLE = 400 
LAMBDA = 1
SAILER = False
LR_LMD = 0.995
MSE_WEIGHT = 100

        
class PlTrainer(object):
    def __init__(self, args):
        self.name = args.name
        self.max_epoch = args.max_epoch
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.log = args.log
        self.out_every = args.out_every
        self.pos_w = args.pos_w
        if args.cuda_dev:
            torch.cuda.set_device(args.cuda_dev[0])
            self.cuda_dev = f'cuda:{args.cuda_dev[0]}'
            self.device = 'cuda'
        else:
            self.cuda_dev = None
            self.device = 'cpu'
        print(f'Using {self.device}')
        self.z_dim = args.z_dim
        self.batch_size = args.batch_size
        self.start_save = args.start_save
        self.start_epoch = args.start_epoch
        self.ckpt_dir = os.path.join(args.ckpt_dir, self.name)
        self.dataset = ChSplitDS(args.data_type)

        self.dataloader =  DataLoader(self.dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=0, #2*len(args.cuda_dev),
                                        pin_memory=True,
                                        drop_last=True)
        input_dim1 = self.dataset.padto1
        input_dim2 = self.dataset.padto2
        if args.sample_batch:
            self.de_batch = True
            self.vae = VAESplit(input_dim2, args.z_dim, batch=True)
        else:
            self.de_batch = False
            self.vae = VAESplit(input_dim2, args.z_dim)
            self.vaeI = VAEInv(self.vae)
            self.model = nn.DataParallel(self.vaeI, device_ids=args.cuda_dev)
        if args.load_ckpt:
            if os.path.isfile(args.load_ckpt):
                print('Loading ' + args.load_ckpt)
                if self.cuda_dev:
                    self.model.module.load_state_dict(torch.load(args.load_ckpt, map_location=self.cuda_dev))
                else:
                    self.model.module.load_state_dict(torch.load(args.load_ckpt, map_location='cpu'))
                print('Finished Loading ckpt...')
            else:
                raise Exception(args.load_ckpt + "\nckpt does not exist!")
        self.model.to(self.device)
        if args.optimizer == 'adam':
            self.optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.cycle = CYCLE * self.dataset.__len__() // self.batch_size // len(args.cuda_dev)
        lr_lmd = lambda epoch: LR_LMD**epoch
        self.le_scdlr = LambdaLR(self.optim, lr_lambda=lr_lmd)
        self.le_scdlr.last_epoch = self.start_epoch-1

    def transfer_depth(self, d, mean, std):
        d = d.log()
        d = (d - mean) / std
        d = d.unsqueeze(1).float().to(self.device)
        return d

    def encode(self, batch_size=2000):
        dataloader =  DataLoader(self.dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=0,
                                pin_memory=True,
                                drop_last=False)
        labels = []
        latent_y = torch.zeros(self.dataset.__len__(), self.z_dim)
        self.model.eval()
        for i, dp in tqdm(enumerate(dataloader)):
            x1, x2, l, d1, d2 = dp
            x2 = x2.float().to(self.device)
            labels = labels + l
            with torch.no_grad():
                y_mean, y_logvar = self.model(x2, d2, no_rec=True)
                # z_mean, _ = self.model.forward(x, no_rec=True)
                # z_mean, _, _, _ = self.model(x)
                latent_y[i*batch_size: (i+1)*batch_size] = y_mean.cpu()
        return latent_y, labels

    def warm_up(self):
        if not os.path.exists(self.ckpt_dir):
            print(f'Making dir {self.ckpt_dir}')
            os.makedirs(self.ckpt_dir)
        self.model.train()
        self.pbar = tqdm(total=WARM_UP)
        total_iter = 0
        for step in range(WARM_UP):
            for x1, x2, l, d1, d2 in self.dataloader:
                x1 = x1.float().to(self.device)
                x2 = x2.float().to(self.device)
                x2_in = x2
                d2 = d2.log()
                d2 = (d2 - self.dataset.atac_mean) / self.dataset.atac_std
                d2 = d2.unsqueeze(1).float().to(self.device)
                mu_2, logvar_2, z2, rec = self.model(x2_in, d2)
                c1, c2 = get_cos(x1,mu_2)
                kld_algn = F.mse_loss(c1, c2, reduction='mean')
                total_iter += 1
                pos_weight = torch.Tensor([self.pos_w]).to(self.device)
                bce = nn.BCEWithLogitsLoss(pos_weight = pos_weight)
                if SAILER:
                    rec_loss = bce(rec, x2)
                else:
                    rec_loss = bce(rec, x2) + kld_algn*MSE_WEIGHT
                self.optim.zero_grad()
                rec_loss.backward()
                self.optim.step()
                if total_iter%50 == 0:
                    self.pbar.write(f'[{total_iter}] vae_recon_loss:{rec_loss.item()}')
            self.pbar.update(1)
        torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, 'warmup.pt'))
        self.pbar.write("[Warmup Finished]")
        self.pbar.close()

    def train(self):
        if not os.path.exists(self.ckpt_dir):
            print(f'Making dir {self.ckpt_dir}')
            os.makedirs(self.ckpt_dir)
        self.model.train()
        kl_list, rec_list, align_list, mkl_list = [], [], [], []
        print('Contrasive Training started')
        self.pbar = tqdm(total=self.max_epoch)
        total_iter = (self.start_epoch-1) * self.dataset.__len__() // self.batch_size + 1
        for epoch in range(self.start_epoch, self.start_epoch + self.max_epoch):
            epoch_kl, epoch_rec, epoch_align, epoch_mkl = [], [], [], []
            for x1, x2, l, mse_w, d2 in self.dataloader:
                if epoch > STOP_CYCLE:
                    kl_w = 1
                else:
                    kl_w = np.round(np.min([2 * (total_iter -(total_iter//self.cycle) * self.cycle) / self.cycle, 1]), 3)
                x1 = x1.float().to(self.device)
                x2 = x2.float().to(self.device)
                x2_in = x2
                d2 = d2.log()
                dd = d2.clone().float().to(self.device)
                d2 = (d2 - self.dataset.atac_mean) / self.dataset.atac_std
                d2 = d2.unsqueeze(1).float().to(self.device)
                mu_2, logvar_2, z2, rec = self.model(x2_in, d2)
                mean2 = torch.zeros_like(mu_2)
                var2 = torch.ones_like(logvar_2)
                kld_z2 = kl(Normal(mu_2, torch.exp(logvar_2).sqrt()), Normal(mean2, var2)).sum()
                c1, c2 = get_cos(x1,mu_2)
                kld_algn = F.mse_loss(c1, c2, reduction='none')
                dd = mse_w
                W = torch.matmul(dd, dd.T)
                kld_algn = torch.multiply(kld_algn, W).sum()
                kld_z = kld_z2
                pos_weight = torch.Tensor([self.pos_w]).to(self.device)
                bce = F.binary_cross_entropy_with_logits(rec, x2, weight=pos_weight, reduction='sum')
                rec_loss = bce
                m_kld2 = apprx_kl(mu_2, torch.exp(logvar_2).sqrt()).mean() - 0.5 * self.z_dim
                m_kld = m_kld2
                if SAILER:
                    loss = kld_z*kl_w + rec_loss + m_kld*kl_w
                else:
                    loss = kld_z*kl_w + rec_loss + m_kld*kl_w + kld_algn*MSE_WEIGHT
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                epoch_kl.append(kld_z2.item())
                epoch_rec.append(rec_loss.item())
                epoch_align.append(kld_algn.item())
                epoch_mkl.append(m_kld.item())
                total_iter += 1
            kl_list.append(np.mean(epoch_kl))
            rec_list.append(np.mean(epoch_rec))
            align_list.append(np.mean(epoch_align))
            mkl_list.append(np.mean(epoch_mkl))
            self.le_scdlr.step()
            self.pbar.update(1)
            self.pbar.write(f'[{epoch}], iter {total_iter}, klw {np.round(kl_w, 4)}, vae_recon_loss:{np.mean(epoch_rec)} vae_kld:{np.mean(epoch_kl)} m_kld:{np.mean(epoch_mkl)}')
            if epoch % self.out_every == 0:
                logdata = {
                    'iter': list(range(self.start_epoch, epoch+1)),
                    'kl': kl_list,
                    'bce': rec_list,
                    'align': align_list,
                    'mkl': mkl_list,
                }
                df = pd.DataFrame(logdata)
                df.to_csv(os.path.join(self.ckpt_dir, 'inv' + self.log), index=False)
                if epoch > self.start_save:
                    torch.save(self.model.module.state_dict(), os.path.join(self.ckpt_dir, f'{epoch}.pt'))
        self.pbar.write("[Contrasive Training Finished]")
        self.pbar.close()
