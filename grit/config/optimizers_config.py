from torch_geometric.graphgym.register import register_config


@register_config('extended_optim')
def extended_optim_cfg(cfg):
    """Extend optimizer config group that is first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg
    """

    # Number of batches to accumulate gradients over before updating parameters
    # Requires `custom` training loop, set `train.mode: custom`
    cfg.optim.batch_accumulation = 1

    # ReduceLROnPlateau: Factor by which the learning rate will be reduced
    cfg.optim.reduce_factor = 0.5

    # ReduceLROnPlateau: #epochs without improvement after which LR gets reduced
    cfg.optim.schedule_patience = 10

    # ReduceLROnPlateau: Lower bound on the learning rate
    cfg.optim.min_lr = 0.0

    # For schedulers with warm-up phase, set the warm-up number of epochs
    cfg.optim.num_warmup_epochs = 50

    # Clip gradient norms while training
    cfg.optim.clip_grad_norm = False


    # ---- Early stopping setting ---------
    # early stop by lr
    cfg.optim.early_stop_by_lr = False

    # early stop by performance
    cfg.optim.early_stop_by_perf = False

    cfg.optim.stop_patience = 100

    cfg.optim.num_cycles = 0.5

    cfg.optim.min_lr_mode = "threshold"


