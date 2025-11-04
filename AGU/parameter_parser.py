import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parameter_parser():
    parser = argparse.ArgumentParser()

    ######################### general parameters ################################
    parser.add_argument('--is_vary', type=bool, default=False, help='control whether to use multiprocess')
    parser.add_argument('--cuda', type=int, default=0, help='specify gpu')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--exp', type=str, default='Unlearn', choices=["Unlearn", "Attack"])
    parser.add_argument('--method', type=str, default='AGU')
    parser.add_argument('--target_model', type=str, default='GCN', choices=["GCN", "GAT", 'GIN', "SGC", "SAGE"])
    parser.add_argument('--n_layers', type=int, default=2)
    
    ########################## unlearning task parameters ######################
    parser.add_argument('--dataset_name', type=str, default='cora',
                        choices=["cora", "citeseer", "pubmed", "CS", "Physics", "flickr", "Photo", "Computers"])
    parser.add_argument('--unlearn_task', type=str, default='node', choices=['feature', "node", "edge"])
    parser.add_argument('--unlearn_ratio', type=float, default=0.05)
    
    ########################## training parameters ###########################
    parser.add_argument('--is_split', type=str2bool, default=True, help='splitting train/test data')
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_runs', type=int, default=2)

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--bias', type=bool, default=False)
    parser.add_argument('--self_loop', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--gat_dim', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--decay', type=float, default=0.0001)

    ######################### agu parameters ###########################
    parser.add_argument('--agu_epochs', type=int, default=30)
    parser.add_argument('--agu_unlearn_lr', type=float, default=0.01)
    parser.add_argument('--edge_weight', type=float, default=0.1)
    parser.add_argument('--n_nei_select', type=int, default=2)
    parser.add_argument('--nei_range', type=int, default=2)
    parser.add_argument('--affected_ratio', type=float, default=0.4)  
    parser.add_argument('--margin_threshold', type=float, default=1e-4)  

    args = vars(parser.parse_args())
    return args
