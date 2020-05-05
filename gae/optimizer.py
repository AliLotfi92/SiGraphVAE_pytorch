import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(reconstructed_iw, log_prior_iw, log_H_iw, adj_orig_tile, nodes, K, pos_weight, norm, warm_up, device):

    binary_cross_binary = F.binary_cross_entropy_with_logits(reconstructed_iw, adj_orig_tile, pos_weight=pos_weight, size_average=False, reduce=False)
    log_lik_iw = -1 * norm * torch.mean(binary_cross_binary, dim=[0, 1])
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    #print(KLD, cost)

    loss_iw0 = -torch.logsumexp(log_lik_iw + (log_prior_iw - log_H_iw) * warm_up/ nodes, dim=0) + torch.log(torch.tensor(K, dtype=float).to(device))
    return loss_iw0
