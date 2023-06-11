import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import GNNPreMP
from torch_geometric.graphgym.models.layer import (new_layer_config,
                                                   BatchNorm1dNode)
from torch_geometric.graphgym.register import register_network



class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension fo the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


@register_network('GritTransformer')
class GritTransformer(torch.nn.Module):
    '''
        The proposed GritTransformer (Graph Inductive Bias Transformer)
    '''

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.ablation = True
        self.ablation = False

        if cfg.posenc_RRWP.enable:
            self.rrwp_abs_encoder = register.node_encoder_dict["rrwp_linear"]\
                (cfg.posenc_RRWP.ksteps, cfg.gnn.dim_inner)
            rel_pe_dim = cfg.posenc_RRWP.ksteps
            self.rrwp_rel_encoder = register.edge_encoder_dict["rrwp_linear"] \
                (rel_pe_dim, cfg.gnn.dim_edge,
                 pad_to_full_graph=cfg.gt.attn.full_attn,
                 add_node_attr_as_self_loop=False,
                 fill_value=0.
                 )


        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        global_model_type = cfg.gt.get('layer_type', "GritTransformer")
        # global_model_type = "GritTransformer"

        TransformerLayer = register.layer_dict.get(global_model_type)

        layers = []
        for l in range(cfg.gt.layers):
            layers.append(TransformerLayer(
                in_dim=cfg.gt.dim_hidden,
                out_dim=cfg.gt.dim_hidden,
                num_heads=cfg.gt.n_heads,
                dropout=cfg.gt.dropout,
                act=cfg.gnn.act,
                attn_dropout=cfg.gt.attn_dropout,
                layer_norm=cfg.gt.layer_norm,
                batch_norm=cfg.gt.batch_norm,
                residual=True,
                norm_e=cfg.gt.attn.norm_e,
                O_e=cfg.gt.attn.O_e,
                cfg=cfg.gt,
            ))
        # layers = []

        self.layers = torch.nn.Sequential(*layers)
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)

        return batch




