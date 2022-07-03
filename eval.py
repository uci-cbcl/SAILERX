import os
import sys
import torch
import numpy as np
import argparse

from torch.utils.data import Dataset, DataLoader
from trainer import PlTrainer
import pandas as pd
import matplotlib.pyplot as plt
import umap
from random import shuffle
import matplotlib
import seaborn as sns
from sknetwork.clustering import Louvain
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description='SAILERX')
parser.add_argument('-t', '--train_type', default='multi', type=str, help='training type: multi, hybrid') #
parser.add_argument('--name', default='main', type=str, help='name of the experiment')
parser.add_argument('--log', default='train_log.csv', type=str, help='name of log file')
parser.add_argument('-l', '--load_ckpt', default=False, type=str, help='path to ckpt loaded')
parser.add_argument('-cuda', '--cuda_dev', default=None, type=int, help='GPU want to use')
parser.add_argument('-batch', '--sample_batch', default=False, type=bool, help='Add batch effect correction')
parser.add_argument('--max_epoch', default=400, type=int, help='maximum training epoch')
parser.add_argument('--start_epoch', default=0, type=int, help='starting epoch')
parser.add_argument('-b', '--batch_size', default=200, type=int, help='batch size')
parser.add_argument('--start_save', default=350, type=int, help='epoch starting to save models')
parser.add_argument('-d', '--data_type', type=str, help='name of dataset')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--pos_w', default=20, type=float, help='BCE positive weight')
parser.add_argument('--weight_decay', default=5e-4, type=str, help='weight decay for adam')
parser.add_argument('--z_dim', default=50, type=int, help='latent dim')
parser.add_argument('--out_every', default=2, type=int, help='save ckpt every x epoch')
parser.add_argument('--ckpt_dir', default='./models/', type=str, help='output directory')
parser.add_argument('--LAMBDA', default=1, type=float, help='lambda value') #
parser.add_argument('--GAMMA', default=6000, type=float, help='gamma value') #
args = parser.parse_args()

solver = PlTrainer(args)
z = solver.encode_latent()
z = z.cpu().numpy()
out_pth = os.path.join(args.ckpt_dir, args.name)
print(f'Exporting to {out_pth}')
np.save(out_pth + '/embedding', z)