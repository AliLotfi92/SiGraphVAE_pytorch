import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Bernoulli
from distributions import RiemannianNormal, WrappedNormal
from utils import Constants

from layers import GraphConvolution, GraphConvolutionK
import manifolds


class GCNModelVAE(nn.Module):
    def __init__(self, psi_input_dim, logv_input_dim, hidden_dim1, hidden_dim2, dropout, K, J, noise_dim=32, device='cpu'):
        c = nn.Parameter(1. * torch.ones(1), requires_grad=False)
        self.latent_dim = hidden_dim2
        self.device = device
        super(GCNModelVAE, self).__init__()

        self.gc1_logv = GraphConvolution(logv_input_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2_logv = GraphConvolution(hidden_dim1, self.latent_dim, dropout, act=lambda x: x)

        self.gc1_psi = GraphConvolutionK(psi_input_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2_psi = GraphConvolutionK(hidden_dim1, self.latent_dim, dropout, act=lambda x: x)
        self.K = K
        self.J = J
        self.dc = InnerProductDecoder(rep=self.K, dropout=dropout, act=lambda x: x)
        self.noise_dim = noise_dim

    def sample_logv(self, x, adj):
        h_logv = self.gc1_logv(x, adj)
        logv = self.gc2_logv(h_logv, adj)
        return logv

    def sample_psi(self, rep, x, adj):
        input = x.unsqueeze(1)
        input = input.repeat(1, rep, 1)
        B = Bernoulli(0.5)
        e = B.sample(sample_shape=[input.shape[0], input.shape[1], self.noise_dim]).to(self.device)
        input_= torch.cat((input, e), dim=2)
        h_mu = self.gc1_psi(input_, adj)
        mu = self.gc2_psi(h_mu, adj)
        return mu

    def sample_n(self, mu, std):
        if self.training:
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        z_logvar = self.sample_logv(x, adj)
        #### this adding constant .eta makes a lot difference in cost,but not in accuracy and those metrics
        #z_log_iw = (F.softplus(z_logvar) + Constants.eta).unsqueeze(1).repeat(1, self.K, 1)
        z_log_iw = z_logvar.unsqueeze(1).repeat(1, self.K, 1)
        sigma_iw1 = torch.exp(z_log_iw / 2)
        sigma_iw2 = sigma_iw1.unsqueeze(2).repeat(1, 1, self.J+1, 1)


        psi_iw = self.sample_psi(self.K, x, adj)
        psi_iw_vec = psi_iw.mean(1)

        zs_sample_iw = self.sample_n(psi_iw, sigma_iw1)
        zs_sample_iw1 = zs_sample_iw.unsqueeze(2)
        zs_sample_iw2 = zs_sample_iw1.repeat(1, 1, self.J+1, 1)

        psi_iw_star = self.sample_psi(self.J, x, adj)
        psi_iw_star0 = psi_iw_star.unsqueeze(1)
        psi_iw_star1 = psi_iw_star0.repeat(1, self.K, 1, 1)
        psi_iw_star2 = torch.cat((psi_iw_star1, psi_iw.unsqueeze(2)), dim=2)


        ker = torch.exp(-0.5 * ((zs_sample_iw2 - psi_iw_star2).pow(2)/(sigma_iw2 + 1e-10).pow(2)).sum(3))

        log_H_iw_vec = torch.log(ker.mean(2) + 1e-10) - 0.5 * z_log_iw.sum(2)
        log_H_iw = log_H_iw_vec.mean(0)

        log_prior_iw_vec = -0.5 * zs_sample_iw.pow(2).sum(2)
        log_prior_iw = log_prior_iw_vec.mean(0)

        z_sample_iw = zs_sample_iw
        logits_x_iw = self.dc(z_sample_iw)

        reconstruct_iw = logits_x_iw

        return reconstruct_iw, log_prior_iw, log_H_iw, psi_iw_vec


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, rep, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.K = rep

    def forward(self,  z):

        for i in range(self.K):
            input_ = z[:, i, :].squeeze()
            input_ = F.dropout(input_, self.dropout, training=self.training)
            adj = self.act(torch.mm(input_, input_.t())).unsqueeze(2)
            if i == 0:
                output = adj
            else:
                output = torch.cat((output, adj), dim=2)
        return output
