from re import L
import numpy as np
import math

from torch.nn.modules.linear import Linear
from utils import normalize, get_spectral_embedding, cat_mtrx

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch

import scipy.io as scio


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class Encoder(nn.Module):
    def __init__(self, h_size, d_size):
        super(Encoder, self).__init__()

        self.gcn = GraphConvolution(d_size, h_size)
        self.gcn2 = GraphConvolution(h_size, h_size)
        self.d_size = d_size

    def forward(self, feature, adj):
        x = get_spectral_embedding(feature, self.d_size)
        #x = adj
        adj = normalize(adj)
        output = self.gcn(x, adj)
        # TODO: softsign
        #output = F.softsign(output)
        #output = F.elu(output)
        output = F.leaky_relu(output, negative_slope=0.2)
        output = F.layer_norm(output,output.size())
        #output = F.dropout(output, 0.2)
        output = self.gcn2(output, adj)
        #output = F.leaky_relu(output, negative_slope=0.2)
        #output = F.dropout(output, 0.6)
        #output  = F.softmax(output)
        return output


class Decoder(nn.Module):
    def __init__(self, h_size):
        super(Decoder, self).__init__()

        self.weight_up = Parameter(torch.randn((h_size, h_size)))
        #self.weight_down = Parameter(torch.randn((h_size, h_size)))

    def forward(self, x):
       # output_up = (x @ self.weight_up) @ x.T
       # output_down = (x @ self.weight_down) @ x.T
       output = x @ x.T
       output = F.sigmoid(output)
       return output


class Decoder_cat(nn.Module):
    def __init__(self, h_size):
        super(Decoder_cat, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(h_size * 2, h_size * 2),
            # nn.Dropout(),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size * 2]),#
            nn.Linear(h_size * 2, h_size),
            # nn.Dropout(),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size]),
            nn.Linear(h_size, h_size // 2),
            # nn.Dropout(),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size // 2]),
            nn.Linear(h_size // 2, 1)
        )
        # if agg_way == 'cat':
        #     self.agg = torch.cat
        # elif agg_way == 'mean':
        #     self.agg = torch.mean

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def cat_(self, tensor_a, tensor_b):
        return torch.cat([tensor_a.expand(*tensor_b.shape), tensor_b], dim=1) #uncertain len(para)
    
    def forward(self, x):
        output = torch.cat([self.cat_(x[i:i+1], x[i+1:]) for i in range(x.shape[0] - 1)], dim=0)
        output = F.sigmoid(self.mlp(output))

        index, count, padding = 0, 1, []
        for i in range(x.shape[0] - 1, -1, -1):
            padding.append(
                torch.cat([output.new_full([1, count], 0), output[index: index+i].T], dim=1)
            )
            index += i; count += 1
        output = torch.cat(padding, dim=0)
        output = output + output.T
        return output

class Decoder_mean(nn.Module):
    def __init__(self, h_size):
        super(Decoder_mean, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size]),
            nn.Linear(h_size, h_size//2),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size//2]),
            nn.Linear(h_size//2, h_size//2),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size//2]),
            nn.Linear(h_size//2, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def cat_(self, tensor_a, tensor_b):
        return (tensor_a+tensor_b)/2

    def forward(self, x):
        output = torch.cat([self.cat_(x[i:i+1], x[i+1:]) for i in range(x.shape[0] - 1)], dim = 0)
        output = F.sigmoid(self.mlp(output))

        index, count, padding = 0, 1, []
        for i in range(x.shape[0] - 1, -1, -1):
            padding.append(
                torch.cat([output.new_full([1, count], 0), output[index: index+i].T], dim=1)
            )
            index += i; count += 1
        output = torch.cat(padding, dim=0)
        output = output + output.T
        return output
   

class Decoder_gru(nn.Module):
    def __init__(self, h_size):
        super(Decoder_gru, self).__init__()

        self.gru = nn.GRUCell(64, h_size)
        self.linear = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size]),
            nn.Linear(h_size, h_size//2),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size//2]),
            nn.Linear(h_size//2, h_size//2),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size//2]),
            nn.Linear(h_size//2, 1)
        )
        self.h_size = h_size

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def cat_(self, tensor_a, tensor_b):
        return torch.cat([tensor_a.expand(*tensor_b.shape), tensor_b], dim=1) #uncertain len(para)
    
    def forward(self, x):
        x_ = torch.cat([self.cat_(x[i:i+1], x[i+1:]) for i in range(x.shape[0] - 1)], dim=0)
        hid = torch.randn(x_.shape[0], self.h_size).cuda()
        x1, x2 = torch.chunk(x_, 2, dim=1)
        output = self.gru(x1, hid)
        output = self.gru(x2, output)
        output = F.sigmoid(self.linear(output))

        index, count, padding = 0, 1, []
        for i in range(x.shape[0] - 1, -1, -1):
            padding.append(
                torch.cat([output.new_full([1, count], 0), output[index: index+i].T], dim=1)
            )
            index += i; count += 1
        output = torch.cat(padding, dim=0)
        output = output + output.T
        return output

class Decoder_gru_2(nn.Module):
    def __init__(self, h_size):
        super(Decoder_gru_2, self).__init__()

        self.gru = nn.GRUCell(64, h_size)
        self.linear = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size]),
            nn.Linear(h_size, h_size//2),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size//2]),
            nn.Linear(h_size//2, h_size//2),
            nn.ReLU(),
            nn.LayerNorm([3486, h_size//2]),
            nn.Linear(h_size//2, 1)
        )
        self.h_size = h_size

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def cat_(self, tensor_a, tensor_b):
        return torch.cat([tensor_a.expand(*tensor_b.shape), tensor_b], dim=1) #uncertain len(para)
    
    def forward(self, x):
        x_ = torch.cat([self.cat_(x[i:i+1], x[i+1:]) for i in range(x.shape[0] - 1)], dim=0)
        #hid = torch.randn(x_.shape[0], self.h_size).cuda()
        x1, x2 = torch.chunk(x_, 2, dim=1)
        output = self.gru(x1, x2)
        #output = self.gru(x2, output)
        output = F.sigmoid(self.linear(output))

        index, count, padding = 0, 1, []
        for i in range(x.shape[0] - 1, -1, -1):
            padding.append(
                torch.cat([output.new_full([1, count], 0), output[index: index+i].T], dim=1)
            )
            index += i; count += 1
        output = torch.cat(padding, dim=0)
        output = output + output.T
        return output

class Translator(nn.Module):
    def __init__(self, h_size, is_softsign: bool):
        super(Translator, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(h_size, 64),
            #nn.BatchNorm1d(64),
            #nn.ELU(),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout(0.3),
            nn.LayerNorm([84,64]),
            #nn.Dropout(0.5),
            # nn.Linear(64, h_size),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(64, 32),
            #nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.LayerNorm([84, 32]),
            #nn.Dropout(0.5),

            nn.Linear(32, 32),
            #nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(32, h_size),
            #nn.LeakyReLU(0.2, inplace=True)
            #nn.ELU(),
            # nn.Dropout(0.2),
            #nn.LayerNorm([82,h_size])
        )
        self.is_softsign = is_softsign

    def forward(self, x):
        output = self.mlp(x)
        # if self.is_softsign:
        #     output = F.softsign(output)
        # else:
        #     output = F.leaky_relu(output, negative_slope=0.2)
        # output  = F.softmax(output)
        return output


class Generator(nn.Module):
    def __init__(self, d_size, h_size, is_softsign: bool, decoder='cat'):
        super(Generator, self).__init__()

        self.encoder = Encoder(h_size, d_size)
        if decoder == 'origin':
            self.decoder = Decoder(h_size)
        else:
            self.decoder = eval(f'Decoder_{decoder}')(h_size)
        self.translator = Translator(h_size, is_softsign=is_softsign)
        self.prelu = nn.PReLU()
        self.is_softsign = is_softsign

    def forward(self, feature, adj):
        output = self.encoder(feature, adj)
        output1= self.translator(output)
        output = self.decoder(output1)
        # if self.is_softsign:
        #     output = F.softsign(output)
        # else:
        #      output = F.leaky_relu(output, 0.2)
            #output = self.prelu(output)
            #output = F.elu(output)
        return output

class Generator1(nn.Module):
    def __init__(self, d_size, h_size, is_softsign: bool):
        super(Generator1, self).__init__()

        self.encoder = Encoder(h_size, d_size)
        self.decoder = Decoder(h_size)
        self.translator = Translator(h_size, is_softsign=is_softsign)
        self.prelu = nn.PReLU()
        self.is_softsign = is_softsign

    def forward(self, feature, adj):
        output = self.encoder(feature, adj)
        output1= self.translator(output)
        output = self.decoder(output1)
        if self.is_softsign:
            output = F.softsign(output)
        else:
             output = F.leaky_relu(output, 0.2)
            #output = self.prelu(output)
            #output = F.elu(output)
        return output

class PretrainED(nn.Module):
    def __init__(self, d_size, h_size, decoder='cat'):
        super(PretrainED, self).__init__()
        self.encoder = Encoder(h_size, d_size)
        if decoder == 'origin':
            self.decoder = Decoder(h_size)
        else:
            self.decoder = eval(f'Decoder_{decoder}')(h_size)

    def forward(self, feature, adj):
        output = self.encoder(feature, adj)
        output = self.decoder(output)
        return output


class Discriminator(nn.Module):
    def __init__(self, d_size, h_size, rep_size, dropout):
        super(Discriminator, self).__init__()
        self.d_size = d_size
        self.dropout = dropout
        self.gc = GraphConvolution(d_size, h_size)
        self.gc2 = GraphConvolution(h_size, h_size)

        self.main = nn.Sequential(
            nn.Linear(h_size, int(rep_size / 2)),
            #nn.LeakyReLU(0.2),
            nn.ELU(),
            # nn.LayerNorm([82,int(rep_size / 2)]),
            nn.Linear(int(rep_size / 2), 16),
            #nn.LeakyReLU(0.2),
            # nn.ELU(),
            # nn.LayerNorm([82, 16])
        )

        self.sigmoid_output = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.normalize = nn.InstanceNorm2d(1)

    def forward(self, feature, adj):
        x = get_spectral_embedding(feature, d=self.d_size)
        adj = normalize(adj)
        x = self.gc(x, adj)
        #x = F.leaky_relu(x)
        # x = F.elu(x)
        # tmp_shape = x.shape
        # x = x.reshape(1, 1, tmp_shape[0], tmp_shape[1])
        # self.normalize(x)
        # x = x.reshape(tmp_shape[0], tmp_shape[1])
        #x = F.dropout(x, self.dropout, training=True)
        #x = F.layer_norm(x, x.size())
        # x = self.gc2(x, adj)
        # x = F.dropout(x, self.dropout, training=True)
        x = F.relu(x)
        x = self.main(x)
        x = x.mean(0)
        # x = F.log_softmax(x, dim=1)
        x = self.sigmoid_output(x)

        return x

'''
class Discriminator1(nn.Module):
    def __init__(self, d_size, h_size, rep_size, dropout):
        super(Discriminator1, self).__init__()
        self.d_size = d_size
        self.dropout = dropout
        self.gc = GraphConvolution(d_size, h_size)
        self.gc2 = GraphConvolution(h_size, h_size)

        self.main = nn.Sequential(
            nn.Linear(h_size, int(rep_size / 2)),
            nn.LeakyReLU(0.2),
            nn.Linear(int(rep_size / 2), 16),
            nn.LeakyReLU(0.2)
        )

        self.sigmoid_output = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.normalize = nn.InstanceNorm2d(1)

    def forward(self, feature, adj):
        # get spectral embedding from adj, D = D_X
        # x=None
        # adj_c = None
        # for i in range(batch_size):
        #     temp1 = adj[i]
        #     temp=get_spectral_embedding(adj[i], d=self.d_size)
        #     if x is None:
        #         x = temp
        #     else:
        #         x=x.concatenate(temp)
        #     if adj_c is None:
        #         adj_c = temp1
        #     else:
        #         adj_c = adj_c.concatenate(temp1)
        # x = x.reshape(-1, 82, self.d_size)
        # adj = adj_c.reshape(-1, 82, 82)
        #x = get_spectral_embedding(feature, d=self.d_size)
        x = feature
        #print(np.sum(adj.cpu().numpy().flatten() == 0))
        adj = normalize(feature)
        # print(adj)
        # GCN layer N*D -> N*D'
        x = self.gc(x, adj)
        # tmp_shape = x.shape
        # x = x.reshape(1, 1, tmp_shape[0], tmp_shape[1])
        # self.normalize(x)
        # x = x.reshape(tmp_shape[0], tmp_shape[1])
        x = F.dropout(x, self.dropout, training=True)
        x = F.relu(x)
        x = F.layer_norm(x, x.size())
        # x = self.gc2(x, adj)
        # x = F.dropout(x, self.dropout, training=True)
        # x = F.relu(x)
        x = self.main(x)
        x = x.mean(0)
        # x = F.log_softmax(x, dim=1)
        x = self.sigmoid_output(x)

        return x
'''
