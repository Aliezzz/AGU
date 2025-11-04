import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, h_dim, bias, self_loop, out_channels, dropout, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, h_dim, bias=bias, add_self_loops=self_loop))
        self.convs.append(GCNConv(h_dim, out_channels, bias=bias, add_self_loops=self_loop))

    def forward(self, x, edge_index):
        for i in range(self.num_layers - 1):
            x1 = F.relu(self.convs[i](x, edge_index))
            x1 = F.dropout(x1, training=self.training, p=self.dropout)

        x2 = self.convs[-1](x1, edge_index)

        return x1, x2

    def gnndelete_forward(self, x, edge_index, return_all_emb=False):
        x1 = self.convs[0](x, edge_index)
        x = F.relu(x1)
        x = F.dropout(x, p=0.2, training=self.training)
        x2 = self.convs[1](x, edge_index)

        if return_all_emb:
            return x1, x2

        return x2

    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i < len(self.convs) - 1:
                    x = F.relu(x)
                xs.append(x[:batch.batch_size].cpu())

            x_all = torch.cat(xs, dim=0)

        return x_all

    def reset_parameters(self):
        for i in range(self.num_layers):
            self.convs[i].reset_parameters()
