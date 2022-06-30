import os
import numpy as np
import pandas as pd
import h5py
import time
from scipy.sparse import csr_matrix

import torch
from torch.utils.data import Dataset

class ChSplitDS(Dataset):
    def __init__(self, name, ref='pca', batch=False):
        super(ChSplitDS, self).__init__()
        if not os.path.isfile(f'./data/{name}.h5'):
            raise Exception(f'Dataset {name} does not exist!')
        tic = time.time()
        f = h5py.File(f'./data/{name}.h5','r')
        self.rna_data = f[f'{ref}']['data'][:]
        g_atac = f['atac']
        self.atac_data = csr_matrix((g_atac['data'][:], g_atac['indices'][:], g_atac['indptr'][:]), g_atac.attrs['shape'])
        self.padto1 = self.atac_data.shape[-1]
        self.padto2 = tuple(g_atac['chrom_size'][:])
        self.cell_label = np.array(f['cells']['labels'][:]).astype(str)
        self.rna_depth = f['cells']['rna_depth'][:]
        if batch:
            self.batch = f['cells']['batch'][:]
        else:
            self.batch = None
        f.close()
        tok = time.time()
        print(f"Finish loading in {tok-tic}, data size {self.rna_data.shape} and {self.atac_data.shape}")
        self.rna_depth = self.rna_depth / self.rna_depth.max()
        self.atac_depth = np.asarray(self.atac_data.sum(axis=1)).flatten()
        self.atac_mean = np.log(self.atac_depth).mean()
        self.atac_std = np.log(self.atac_depth).std()
        self.atac_depth = self.atac_depth / self.atac_depth.max()
        self.weight = np.log(np.divide(self.rna_depth, self.atac_depth))
        self.weight = self.weight + np.abs(self.weight.min())
        self.weight = self.weight / self.weight.max()
        self.pcs = self.rna_data
        self.norm = False

    def __getitem__(self, index):
        rna = np.array(self.pcs[index]).flatten().astype(np.float)
        atac = np.array(self.atac_data[index].todense()).flatten().astype(np.bool).astype(np.float)
        if self.batch is not None:
            return rna, atac, self.batch[index], self.weight[index], np.sum(atac)
        else:
            return rna, atac, self.cell_label[index], self.weight[index], np.sum(atac)

    def __len__(self):
        return len(self.cell_label)


class HybridDS(Dataset):
    def __init__(self, name, ref='pca'):
        super(HybridDS, self).__init__()
        if not os.path.isfile(f'data/{name}.h5'):
            raise Exception(f'Dataset {name} does not exist!')
        tic = time.time()
        f = h5py.File(f'Data/{name}.h5','r')
        self.rna_data = f[f'{ref}']['data'][:]
        g_atac = f['atac']
        self.atac_data = csr_matrix((g_atac['data'][:], g_atac['indices'][:], g_atac['indptr'][:]), g_atac.attrs['shape'])
        self.padto1 = self.atac_data.shape[-1]
        self.padto2 = tuple(g_atac['chrom_size'][:])
        self.cell_label = np.array(f['cells']['labels'][:]).astype(str)
        self.rna_depth = f['cells']['rna_depth'][:]
        g_single = f['single']
        self.single_data = csr_matrix((g_single['single_data'][:], g_single['single_indices'][:], g_single['single_indptr'][:]), g_single.attrs['shape'])
        f.close()
        tok = time.time()
        print(f"Finish loading in {tok-tic}, data size {self.rna_data.shape} and {self.atac_data.shape}")
        self.batch = True
        self.rna_depth = self.rna_depth / self.rna_depth.max()
        self.atac_depth = np.asarray(self.atac_data.sum(axis=1)).flatten()
        self.atac_mean = np.log(self.atac_depth).mean()
        self.atac_std = np.log(self.atac_depth).std()
        self.atac_depth = self.atac_depth / self.atac_depth.max()
        self.single_mean = np.log(self.single_data.sum(axis=1)).mean()
        self.single_std = np.log(self.single_data.sum(axis=1)).std()
        self.weight = np.log(np.divide(self.rna_depth, self.atac_depth))
        self.weight = self.weight + np.abs(self.weight.min())
        self.weight = self.weight / self.weight.max()
        self.pcs = self.rna_data
        self.max = int(self.single_data.shape[0])
        self.norm = False

    def __getitem__(self, index):
        rna = np.array(self.pcs[index]).flatten().astype(np.float)
        atac = np.array(self.atac_data[index].todense()).flatten().astype(np.bool).astype(np.float)
        if index > self.max - 1:
            s_index = np.random.randint(self.max - 1)
        else:
            s_index = index
        sig_data = np.array(self.single_data[s_index].todense()).flatten().astype(np.bool).astype(np.float)
        return rna, atac, self.cell_label[index], self.weight[index], np.sum(atac), sig_data, np.sum(sig_data)

    def __len__(self):
        return len(self.cell_label)
