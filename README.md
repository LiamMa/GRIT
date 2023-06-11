# Read Me of GRIT
> This is a demo version of code for `Graph Inductive Bias Transformer (GRIT)` used for reproduction in the submission to ICML-2023 (Please don't distribute)


> The code base is built upon GraphGPS's code (https://github.com/rampasek/GraphGPS) with several modifications (not all functions from GraphGPS outside our model have been verified)

### Python environment setup with Conda
```bash
conda create -n graphgps python=3.9
conda activate graphgps 

# please change the cuda/device version as you need

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 --trusted-host download.pytorch.org


pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric==2.2.0 -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --trusted-host data.pyg.org
# the installation of pyg-lib is optional (but not fully support for torch=1.10.1), more details please see https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html


# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
## conda install openbabel fsspec rdkit -c conda-forge
# alternative if conda doesn't work: 
pip install rdkit

pip install torchmetrics==0.9.1
# pip install performer-pytorch # (optional) not used in our model
pip install ogb
pip install tensorboardX
pip install yacs
pip install opt_einsum
pip install graphgym 
pip install pytorch-lightning # required by graphgym 
pip install setuptools==59.5.0

# distuitls has conflicts with pytorch with latest version of setuptools

# pip install wandb # the wandb is used in GraphGPS but not used in GRIT (ours); please verify the usability before using.

# pip install mlflow 
## ---> to use mlflow
## mlflow server --backend-store-uri mlruns --port 5000

```

### Running New Transformer
```bash
# current deg_scaler design is slightly different from the old one; to verify the performance

# Run
python main.py --cfg configs/GRIT/zinc-GRIT.yaml  wandb.use False accelerator "cuda:0" optim.max_epoch 2000 seed 41 dataset.dir 'xx/xx/data'

python main.py --cfg configs/GRIT/zinc-GRIT.yaml  wandb.use False accelerator "cpu" optim.max_epoch 2 seed 41

# replace 'cuda:0' with the device to use
# replace 'xx/xx/data' with your data-dir (by default './datasets"
```



