# SAILERX
Pytorch implementation for SAILERX

## Setup

Clone the repository.

```
git clone https://github.com/uci-cbcl/SAILER.git
```

Navigate to the root of this repo and setup the conda environment.

```
conda env create -f deepatac.yml
```

Activate conda environment.

```
conda activate deepatac
```

## Data

Please download data [here](https://drive.google.com/drive/folders/1yQeF3Ch_yZg2hXRcTe9X30ilQB_qaRq1?usp=sharing) and setup your data folder as the following structure:

```
SAILERX
|___data  
    |___...
```

## Standard training
To train with one multimodal sc-deq data (scRNA-seq + scATAC-seq). Using PBMC 10k as an example.
```
python train.py -d pbmc10k -cuda 0 --pos_w 20
```

## Training with data from two batches
To train with multiple multimodal sc-deq data (scRNA-seq + scATAC-seq). Using PBMC 10k + 3k as an example.
```
python train.py -d pbmc_batch -cuda 0 --pos_w 20 -batch True
```

## Hybrid training
To train with multiple multimodal sc-deq data (scRNA-seq + scATAC-seq). Using PBMC 10k + 3k as an example.
```
python train.py -d pbmc_hybrid -cuda 0 --pos_w 20 -batch True -t hybrid
```

For more info, please use
```
python train.py -h
```
or see examples [here](https://github.com/uci-cbcl/SAILERX/tree/main/notebooks).