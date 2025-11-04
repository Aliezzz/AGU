import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, h_dim, heads, bias, self_loop, out_channels, dropout, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, h_dim, heads=heads, dropout=self.dropout, bias=bias, add_self_loops=self_loop))
        self.convs.append(GATConv(h_dim * heads, out_channels, heads=1, concat=False, dropout=self.dropout, bias=bias, add_self_loops=self_loop))

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i in range(self.num_layers - 1):
            x1 = F.relu(self.convs[i](x, edge_index))
            x1 = F.dropout(x1, p=self.dropout, training=self.training)

        x2 = self.convs[-1](x1, edge_index)

        return x1, x2

    def gnndelete_forward(self, x, edge_index, return_all_emb=False):
        x1 = self.convs[0](x, edge_index)
        x = F.relu(x1)
        x2 = self.convs[1](x, edge_index)

        if return_all_emb:
            return x1, x2

        return x2
    
    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
    