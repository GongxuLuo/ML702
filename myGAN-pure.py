#from lcy.viewModel.utils import adj
from inspect import stack
import sys
sys.path.append('/home/LAB/luogx/lcy/viewModel/')

import argparse
#from lcy.CondGen.models import Discriminator
import os
import numpy as np
import math
import pickle
import warnings
import statistics as s
import random

from model import *
from dataset import BpDataset, HivDataset, PpmiDataset, Ppmi622Dataset
from utils import *

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch

from evaluate import print_stats, evaluate, clean_lists

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--to1_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--gamma", type=int, default=15, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr1", type=float, default=0.00006 , help="adam: learning rate")
parser.add_argument("--lr2", type=float, default=0.00006, help="adam: learning rate")
parser.add_argument("--dlr1", type=float, default=0.6, help="adam: learning rate discriminatro")
parser.add_argument("--dlr2", type=float, default=0.6, help="adam: learning rate dicriminator2")
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--k_fold", type=int, default=5, help="size of each image dimension")
parser.add_argument("--dropout", type=float, default=0.3, help="...")
parser.add_argument("--dataset", type=str, default="bp", help="dataset: choose dataset by name")
parser.add_argument("--generator", type=str, default="cat", help="generator: choose generator by name")

parser.add_argument('--d_size', type=int, default=32, help="generator input size")
parser.add_argument('--h_size', type=int, default=64, help="generator hidden size")
parser.add_argument('--o_size', type=int, default=64, help="generator output size")
parser.add_argument('--rep_size', type=int, default=32, help="generator output size")

parser.add_argument('--is_neg', type=bool, default=False, help="is negative")
parser.add_argument("--agg_way", type=str, default="gru_2", help="aggregation way")

opt = parser.parse_args()
print(opt)
random.seed(100)
np.random.seed(100)
# torch.manual_seed(100)#cpu 
# torch.cuda.manual_seed(345)
# torch.cuda.manual_seed_all(345)
cuda = True if torch.cuda.is_available() else False

data_path = './data/bp/BP.mat'

load_ED = False

# Loss function
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator

generator_1to2 = Generator(opt.d_size, 64, False, opt.generator)
generator_2to1 = Generator(32, opt.h_size, False, opt.generator)
Discriminator1 = Discriminator(opt.d_size, opt.h_size, opt.rep_size, opt.dropout)
Discriminator2 = Discriminator(opt.d_size, opt.h_size, opt.rep_size, opt.dropout)

# if load_ED:
#     generator_1to2.encoder = pickle.load(open('./checkpoint/view1_encoder', 'rb'))
#     #generator_1to2.decoder = pickle.load(open('./checkpoint/view2_decoder', 'rb'))
#     generator_2to1.encoder = pickle.load(open('./checkpoint/view2_encoder', 'rb'))
#     #generator_2to1.decoder = pickle.load(open('./checkpoint/view1_decoder', 'rb'))
#     Discriminator1 = pickle.load(open('./checkpoint/discriminator1', 'rb'))
#     Discriminator2 = pickle.load(open('./checkpoint/discriminator2', 'rb'))

if cuda:
    generator_1to2.cuda()
    generator_2to1.cuda()
    Discriminator1.cuda()
    Discriminator2.cuda()


# Optimizers
optimizer_g1to2 = torch.optim.Adam(filter(lambda p: p.requires_grad, generator_1to2.parameters()), lr=opt.lr1, betas=(opt.b1, opt.b2))
optimizer_g2to1 = torch.optim.Adam(filter(lambda p: p.requires_grad, generator_2to1.parameters()), lr=opt.lr2, betas=(opt.b1, opt.b2))
optimizer_d1 = torch.optim.Adam(Discriminator1.parameters(), lr=opt.dlr1)
optimizer_d2 = torch.optim.Adam(Discriminator2.parameters(), lr=opt.dlr2)


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
mse_loss = torch.nn.MSELoss()

# ----------
#  Training
# ----------
if opt.dataset == 'bp':
    print('load bp dataset')
    dataset = BpDataset(train='train', K=opt.k_fold)
elif opt.dataset == 'hiv':
    print('load hiv dataset')
    dataset = HivDataset(train='train')
elif opt.dataset == 'ppmi':
    print('load ppmi dataset')
    dataset = PpmiDataset(train='train')
elif opt.dataset == 'ppmi_622':
    print('load ppmi_622 dataset')
    dataset = Ppmi622Dataset(train='train')
else:
    print('valid dataset')

dataTrain = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)
generator_1to2.train(); generator_2to1.train()
Discriminator1.train(); Discriminator2.train()
# for param in generator_1to2.parameters():
#     print(type(param.data), param.size(), param)
for epoch in range(opt.n_epochs):
    rec1_list, rec2_list, disc1_real_list, disc1_false_list, disc2_false_list, disc2_real_list, gen1_disc_list, gen2_disc_list = [], [], [], [], [], [], [], []
    gen_list = []
    for i, (views_A, views_B, labels) in enumerate(dataTrain):
        views_A = views_A[0]
        views_A_edge_num = torch.sum(views_A != 0).item()
        Variable(views_A.type(Tensor)).cuda()
        #views_A_adj = adj(views_A,'view1')
        views_B = views_B[0]
        views_B_edge_num = torch.sum(views_B != 0).item()

        Variable(views_B.type(Tensor)).cuda()
        #views_B_adj = adj(views_B, 'view2')
        labels = labels[0]

        # Adversarial ground truths
        valid = Variable(torch.ones(1)).cuda()
        fake = Variable(torch.zeros(1)).cuda()

        # Configure input
        real_views_A = Variable(views_A.type(Tensor)).cuda()
        real_views_B = Variable(views_B.type(Tensor)).cuda()

        # -----------------
        #  Train Generator_1to2
        # -----------------

        gen_views_B = generator_1to2(real_views_A, real_views_A)
        # gen_views_B = adj(gen_views_B, 'view2')
        gen_views_B = topk_adj(gen_views_B, views_B_edge_num, opt.is_neg)

        d_out = Discriminator2(real_views_B, real_views_B)
        real_loss = adversarial_loss(d_out, valid)
        disc2_real_list.append(d_out.data)
        #print(Discriminator2(real_views_B, real_views_B))
        d_out = Discriminator2(gen_views_B.detach(), gen_views_B.detach())
        fake_loss = adversarial_loss(d_out, fake)
        disc2_false_list.append(d_out.data)
        d_loss = real_loss + fake_loss
        optimizer_d1.zero_grad()
        d_loss.backward()
        optimizer_d1.step()
        
        optimizer_g1to2.zero_grad()
        d_out_g = Discriminator2(gen_views_B, gen_views_B)
        gan_loss = adversarial_loss(d_out_g, valid)
        # disc2_list.append(d_out.data)
        gen2_disc_list.append(d_out_g.data)

        rec_loss = F.l1_loss(gen_views_B, real_views_B)
        # rec_loss = mse_loss(gen_views_B, real_views_B)
        rec1_list.append(rec_loss.data)
        gen_loss = gan_loss + opt.gamma * rec_loss

        gen_loss.backward()
        optimizer_g1to2.step()

        # ---------------------
        #  Train Generator_2to1
        # ---------------------
        if epoch < opt.to1_epochs:

            gen_views_A = generator_2to1(real_views_B, real_views_B)
            # gen_views_A = adj(gen_views_A, 'view1')
            gen_views_A = topk_adj(gen_views_A, views_A_edge_num, False)

            optimizer_d2.zero_grad()
            d_out = Discriminator1(real_views_A, real_views_A)
            real_loss = adversarial_loss(d_out, valid)
            disc1_real_list.append(d_out.data)
            d_out = Discriminator1(gen_views_A.detach(), gen_views_A.detach())
            fake_loss = adversarial_loss(d_out, fake)
            disc1_false_list.append(d_out.data)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d2.step()

            optimizer_g2to1.zero_grad()
            d_out = Discriminator1(gen_views_A, gen_views_A)
            gan_loss = adversarial_loss(d_out, valid)
            # disc1_list.append(d_out.data)
            gen1_disc_list.append(d_out.data)
            rec_loss = F.l1_loss(gen_views_A, real_views_A)
            rec2_list.append(rec_loss)
            gen_loss = gan_loss + opt.gamma * rec_loss
            gen_loss.backward()
            optimizer_g2to1.step()

            
    # print(generator_1to2.decoder.weight_up)

    if epoch < opt.to1_epochs:    
        print("[Epoch %d/%d] [rec1 loss: %f] [rec2 loss: %f] [dis1_real: %f] [dis1_false: %f] [dis2_real: %f] [dis2_false: %f] [gen1: %f] [gen2: %f]" 
        % (epoch, opt.n_epochs, torch.mean(torch.stack(rec1_list)), torch.mean(torch.stack(rec2_list)),
        torch.mean(torch.stack(disc1_real_list)), torch.mean(torch.stack(disc1_false_list)), torch.mean(torch.stack(disc2_real_list)),torch.mean(torch.stack(disc2_false_list)),
        torch.mean(torch.stack(gen1_disc_list)), torch.mean(torch.stack(gen2_disc_list))))
    else:
        print("[Epoch %d/%d] [rec1 loss: %f] "
        % (epoch, opt.n_epochs, torch.mean(torch.stack(rec1_list)))
        )
    # for p in generator_1to2.parameters():
        # print(p.grad)
    #print(np.mean(gen_list))

generator_1to2.eval(); generator_2to1.eval()
Discriminator2.eval(); Discriminator2.eval()
test(opt, generator_1to2, generator_2to1)


# pickle.dump(generator_1to2, open('./checkpoint/generator_1to2', 'wb'))
# pickle.dump(generator_2to1, open('./checkpoint/generator_2to1', 'wb'))
