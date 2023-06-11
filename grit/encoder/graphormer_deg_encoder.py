import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch import nn
import torch_geometric as pyg


@register_node_encoder('GraphormerDeg')
class GraphormerDegEncoder(torch.nn.Module):
    def __init__(self, emb_dim, max_deg=150):
        super().__init__()

        self.deg_emb = nn.Embedding(max_deg, emb_dim)

    def forward(self, batch):
        if "deg" in batch:
            deg = batch.deg
        else:
            deg = pyg.utils.degree(batch.edge_index[1],
                                   num_nodes=batch.num_nodes,
                                   dtype=torch.float
                                   )

        deg_emb = self.deg_emb(deg.type(torch.long))
        
        batch.x = batch.x + deg_emb
        return batch
