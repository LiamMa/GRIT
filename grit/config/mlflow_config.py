from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_mlflow')
def set_cfg_mlflow(cfg):
    """
    MLflow tracker configuration.
    """

    # MLflow group
    cfg.mlflow = CN()

    # Use MLflow or not
    cfg.mlflow.use = False
    # Optional run name
    cfg.mlflow.project = " "
    cfg.mlflow.name = " "
