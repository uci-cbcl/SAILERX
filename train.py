import argparse
from ezdict import EZDict
from trainer import PlTrainer
import torch

args = EZDict({
    "name": 'pbmc10k_pca_d50_w20',
    'log': 'train_log.csv',
    'load_ckpt': False, #'/home/yingxinc4/DeepATAC/models/newsub_bce25_s0.35f3000var/warmup.pt',
    'cuda_dev': [0], #False
    'sample_batch': False,
    "max_epoch": 450,
    'start_epoch': 1,
    'batch_size': 200,
    'start_save': 290,
    'data_type': 'pbmc10k', #'share_seq', 'pbmc'
    'lr': 1e-4,
    'pos_w': 20, #babel 42, pbmc_callpeak 20
    'weight_decay': 5e-4,
    'optimizer': 'adam',
    'z_dim': 50,
    'out_every': 5,
    'ckpt_dir': './models/'
})


solver = PlTrainer(args)
solver.warm_up()
solver.train()
