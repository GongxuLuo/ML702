import sys

import argparse
import os
import numpy as np
import math
import pickle
import warnings
import random
import itertools

from model import *
from dataset import BpDataset, HivDataset, PpmiDataset, Ppmi622Dataset
from utils import *

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from evaluate import print_stats, evaluate, clean_lists

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--to1_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--gamma", type=int, default=15, help="number of epochs of training")
parser.add_argument("--gamma_cyc", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr1", type=float, default=0.00006 , help="adam: learning rate")
parser.add_argument("--lr2", type=float, default=0.00006, help="adam: learning rate")
parser.add_argument("--dlr1", type=float, default=0.001, help="adam: learning rate discriminatro")
parser.add_argument("--dlr2", type=float, default=0.001, help="adam: learning rate dicriminator2")
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--k_fold", type=int, default=5, help="size of each image dimension")
parser.add_argument("--dropout", type=float, default=0.3, help="...")
parser.add_argument("--dataset", type=str, default="bp", help="dataset: choose dataset by name")
parser.add_argument("--generator", type=str, default="cat", help="generator: choose generator by name")

parser.add_argument('--d_size', type=int, default=32, help="generator input size")
parser.add_argument('--h_size', type=int, default=64, help="generator hidden size")
parser.add_argument('--o_size', type=int, default=64, help="generator output size")
parser.add_argument('--rep_size', type=int, default=32, help="generator output size")

parser.add_argument('--is_neg', type=bool, default=False, help="is negative")
parser.add_argument('--load', type=bool, default=False, help="load saved model")
parser.add_argument("--agg_way", type=str, default="cat", help="aggregation way")

opt = parser.parse_args()
print(opt)
random.seed(100)
np.random.seed(100)
torch.manual_seed(100)
#torch.cuda.manual_seed(345)
#torch.cuda.manual_seed_all(345)
cuda = True if torch.cuda.is_available() else False


# Loss function
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Initialize generator and discriminator
G_AB = Generator(opt.d_size, 64, False, opt.generator)
G_BA = Generator(32, opt.h_size, False, opt.generator)
D_A = Discriminator(opt.d_size, opt.h_size, opt.rep_size, opt.dropout)
D_B = Discriminator(opt.d_size, opt.h_size, opt.rep_size, opt.dropout)


if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()


# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()),
    lr=opt.lr1,
    betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.dlr1, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.dlr2, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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

for epoch in range(opt.n_epochs):
    G_AB.train(); G_BA.train()
    D_A.train(); D_B.train()
    rec1_list, rec2_list, disc1_list, disc2_list, gen1_disc_list, gen2_disc_list = [], [], [], [], [], []
    cycle1_list, cycle2_list = [], []
    gen_list = []
    for i, (views_A, views_B, labels) in enumerate(dataTrain):
        views_A, views_B, labels = views_A[0], views_B[0], labels[0]
        views_A_edge_num = torch.sum(views_A != 0).item()
        views_B_edge_num = torch.sum(views_B != 0).item()

        # Adversarial ground truths
        valid = Variable(torch.ones(1), requires_grad=False).cuda()
        fake = Variable(torch.zeros(1), requires_grad=False).cuda()

        # Configure input
        real_A = Variable(views_A.type(Tensor)).cuda()
        real_B = Variable(views_B.type(Tensor)).cuda()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A, real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B, real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A, real_A)
        fake_B = topk_adj(fake_B, views_B_edge_num)
        loss_GAN_AB = criterion_GAN(D_B(fake_B, fake_B), valid)
        fake_A = G_BA(real_B, real_B)
        fake_A = topk_adj(fake_A, views_A_edge_num)
        loss_GAN_BA = criterion_GAN(D_A(fake_A, fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B, fake_B)
        recov_A = topk_adj(recov_A, views_A_edge_num)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A, fake_A)
        recov_B = topk_adj(recov_B, views_B_edge_num)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + opt.gamma_cyc * loss_cycle + opt.gamma * loss_identity
        rec1_list.append(loss_id_A); rec2_list.append(loss_id_B)
        gen1_disc_list.append(loss_GAN_AB); gen2_disc_list.append(loss_GAN_BA)
        cycle1_list.append(loss_cycle_A); cycle2_list.append(loss_cycle_B)

        loss_G.backward()
        optimizer_G.step()

        # ----------------------
        #  Train Discriminator_1
        # ----------------------
        
        optimizer_D_A.zero_grad()
    
        real_loss = criterion_GAN(D_A(real_A, real_A), valid)
        disc1_list.append(real_loss.data)
        fake_loss = criterion_GAN(D_A(fake_A.detach(), fake_A.detach()), fake)
        disc1_list.append(fake_loss.data)
    
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D_A.step()
        

        # ---------------------
        #  Train Discriminator_2
        # ---------------------

        optimizer_D_B.zero_grad()
    
        real_loss = criterion_GAN(D_B(real_B, real_B), valid)
        disc2_list.append(real_loss.data)
        fake_loss = criterion_GAN(D_B(fake_B.detach(), fake_B.detach()), fake)
        disc2_list.append(fake_loss.data)
    
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D_B.step()


    print("[Epoch %d/%d] [rec1 loss: %f] [rec2 loss: %f] [cyc1: %f] [cyc2: %f] [dis1: %f] [dis2: %f] [gen1: %f] [gen2: %f]" 
        % (
            epoch, opt.n_epochs,
            torch.mean(torch.stack(rec1_list)),
            torch.mean(torch.stack(rec2_list)),
            torch.mean(torch.stack(cycle1_list)),
            torch.mean(torch.stack(cycle2_list)),
            torch.mean(torch.stack(disc1_list)),
            torch.mean(torch.stack(disc2_list)),
            torch.mean(torch.stack(gen1_disc_list)),
            torch.mean(torch.stack(gen2_disc_list)),
        )
    )
test(opt, G_AB, G_BA)


pickle.dump(G_AB, open('./checkpoint/cycle_gan/generator_1to2', 'wb'))
pickle.dump(G_BA, open('./checkpoint/cycle_gan/generator_2to1', 'wb'))
