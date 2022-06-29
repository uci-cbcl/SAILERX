import argparse
# from ezdict import EZDict
from trainer import PlTrainer
import torch 

# args = EZDict({
#     "name": 'pbmc10k_pca_d50_w20',
#     'log': 'train_log.csv',
#     'load_ckpt': False, 
#     'cuda_dev': [0], #False
#     'sample_batch': False,
#     "max_epoch": 450,
#     'start_epoch': 1,
#     'batch_size': 200,
#     'start_save': 290, # starting to save at epoch
#     'data_type': 'pbmc10k', # 'snare'
#     'lr': 1e-4,
#     'pos_w': 20, 
#     'weight_decay': 5e-4,
#     'optimizer': 'adam',
#     'z_dim': 50,
#     'out_every': 5,
#     'ckpt_dir': './models/'
# })

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
parser.add_argument('--model_type', default='inv', type=str, help='model type')
parser.add_argument('-d', '--data_type', type=str, help='path to dataset')
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
if args.train_type== 'multi':
    solver.warm_up()
    solver.train()
else:
    solver.hybrid_warmup()
    solver.hybrid_train()
