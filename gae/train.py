from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import warnings
import os

from model import GCNModelVAE
from optimizer import loss_function
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score
from torch.utils.tensorboard import SummaryWriter

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return 'cuda:'+str(np.argmax(memory_available))

device = torch.cuda.is_available()
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=4, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.0005, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset-str', type=str, default='citeseer', help='type of dataset.')
parser.add_argument('--device', type=str, default=get_freer_gpu() if device else 'cpu')
parser.add_argument('--noise_dim', type=int, default=32)
parser.add_argument('--K', type=int, default=2)
parser.add_argument('--J', type=int, default=1)
args = parser.parse_args()



warnings.filterwarnings('ignore')

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features = load_data(args.dataset_str)
    features = features.to(args.device)
    n_nodes, feat_dim = features.shape

    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()

    adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
    adj = adj_train

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_norm = adj_norm.to(args.device)

    adj_label = adj_train + sp.eye(adj_train.shape[0])
    # adj_label = sparse_to_tuple(adj_label)
    adj_label = torch.FloatTensor(adj_label.toarray())
    adj_orig_tile = adj_label.unsqueeze(2).repeat(1, 1, args.K)
    adj_orig_tile = adj_orig_tile.to(args.device)

    pos_weight = torch.tensor(float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()).float().to(device=args.device)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

    psi_input_dim = args.noise_dim + feat_dim
    logv_input_dim = feat_dim

    model = GCNModelVAE(psi_input_dim, logv_input_dim, args.hidden1, args.hidden2, args.dropout, args.K, args.J, args.noise_dim, args.device).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        warm_up = torch.min(torch.FloatTensor([epoch/300, 1])).to(args.device)

        t = time.time()
        model.train()
        optimizer.zero_grad()

        reconstruct_iw, log_prior_iw, log_H_iw, psi_iw_vec = model(features, adj_norm)
        hidden_emb = psi_iw_vec.data.contiguous().cpu().numpy()

        loss = loss_function(reconstructed_iw=reconstruct_iw, log_prior_iw=log_prior_iw, log_H_iw=log_H_iw,
                             adj_orig_tile=adj_orig_tile, nodes=n_nodes, K=args.K, pos_weight=pos_weight, norm=norm,
                             warm_up=warm_up, device=args.device)

        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        roc_score_val, ap_score_val = get_roc_score(hidden_emb, adj_orig, val_edges, val_edges_false)
        roc_score_test, ap_score_test = get_roc_score(hidden_emb, adj_orig, test_edges, test_edges_false)

        print('Epoch:', '%04d --->   ' %(epoch+1), 'training_loss = {:.5f}   '.format(cur_loss), 'val_AP = {:.5f}   '.format(ap_score_val),
              'val_ROC = {:.5f}   '.format(roc_score_val), 'test_AP = {:.5f}   '.format(ap_score_test), 'test_ROC = {:.5f}   '.format(roc_score_test),
              'time = {:.5f}   '.format(time.time() - t))

        writer.add_scalar('Loss/train_loss', cur_loss, epoch)

        writer.add_scalar('Average Precision/test', ap_score_test, epoch)
        writer.add_scalar('Average Precision/val', ap_score_val, epoch)

        writer.add_scalar('Area under Roc(AUC)/test', roc_score_test, epoch)
        writer.add_scalar('Area under Roc(AUC)/val', roc_score_val, epoch)


    print("Optimization Finished!")


if __name__ == '__main__':
    writer = SummaryWriter(log_dir='./logs')
    gae_for(args)
