import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import distributions
import manifolds
import math


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.init = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.weight = Parameter(torch.nn.init.uniform_(torch.FloatTensor(in_features, out_features), a=-self.init, b=self.init), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolutionK(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolutionK, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init = math.sqrt(6.0 / (self.in_features + self.out_features))
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.nn.init.uniform_(torch.FloatTensor(in_features, out_features), a=-self.init, b=self.init), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        K = input.shape[1]
        for i in range(K):
            x = input[:, i, :].squeeze()
            x = F.dropout(x, self.dropout, self.training)
            support = torch.mm(x, self.weight)
            output_ = torch.spmm(adj, support)
            output_ = self.act(output_)
            if i == 0:
                output = output_.unsqueeze(1)
            else:
                output = torch.cat((output, output_.unsqueeze(1)), dim=1)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
