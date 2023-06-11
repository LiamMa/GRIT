
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml  wandb.use False accelerator "cuda:0" seed 0  &
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml  wandb.use False accelerator "cuda:1" seed 1  &
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml  wandb.use False accelerator "cuda:2" seed 2  &
python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml  wandb.use False accelerator "cuda:3" seed 3  &
wait

python main.py --cfg configs/GRIT/zinc-GRIT-RRWP.yaml accelerator "cpu" seed 0
