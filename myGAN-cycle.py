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
parser.add_argument('--seed', type=int, default=100, help="random seed")
parser.add_argument('--cuda_seed', type=int, default=345, help="random cuda seed")

opt = parser.parse_args()
print(opt)
random.seed(opt.seed)
np.random.seed(opt.seed)
# torch.manual_seed(opt.seed)
# torch.cuda.manual_seed(opt.cuda_seed)
#torch.cuda.manual_seed_all(opt.cuda_seed)
cuda = True if torch.cuda.is_available() else False


# Loss function
adversarial_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()
criterion_recon = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

# Initialize generator and discriminator
generator_1to2 = Generator(opt.d_size, 64, False, opt.generator)
generator_2to1 = Generator(32, opt.h_size, False, opt.generator)
discriminator1 = Discriminator(opt.d_size, opt.h_size, opt.rep_size, opt.dropout)
discriminator2 = Discriminator(opt.d_size, opt.h_size, opt.rep_size, opt.dropout)

if opt.load:
    model_path = f'./checkpoint/cycle_{opt.generator}/'
    generator_1to2.encoder = pickle.load(open(model_path + 'g1to2_encoder', 'rb'))
    generator_1to2.decoder = pickle.load(open(model_path + 'g2to1_decoder', 'rb'))
    generator_2to1.encoder = pickle.load(open(model_path + 'g2to1_encoder', 'rb'))
    generator_2to1.decoder = pickle.load(open(model_path + 'g1to2_decoder', 'rb'))
    discriminator1 = pickle.load(open(model_path + 'discriminator1', 'rb'))
    discriminator2 = pickle.load(open(model_path + 'discriminator2', 'rb'))
    print('Load pre-train model successfully.')

if cuda:
    generator_1to2.cuda()
    generator_2to1.cuda()
    discriminator1.cuda()
    discriminator2.cuda()


# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(generator_1to2.parameters(), generator_2to1.parameters()),
    lr=opt.lr1,
    betas=(opt.b1, opt.b2)
)
optimizer_d1 = torch.optim.Adam(discriminator1.parameters(), lr=opt.dlr1, betas=(opt.b1, opt.b2))
optimizer_d2 = torch.optim.Adam(discriminator2.parameters(), lr=opt.dlr2, betas=(opt.b1, opt.b2))

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
    generator_1to2.train(); generator_2to1.train()
    discriminator1.train(); discriminator2.train()
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
        real_views_A = Variable(views_A.type(Tensor)).cuda()
        real_views_B = Variable(views_B.type(Tensor)).cuda()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        gen_views_B = generator_1to2(real_views_A, real_views_A)
        gen_views_B = topk_adj(gen_views_B, views_B_edge_num)
        gen_views_A = generator_2to1(real_views_B, real_views_B)
        gen_views_A = topk_adj(gen_views_A, views_A_edge_num)
        
        # GAN loss
        disc_AB = discriminator2(gen_views_B, gen_views_B)
        disc_BA = discriminator1(gen_views_A, gen_views_A)
        gan_loss_AB = adversarial_loss(disc_AB, valid)
        gan_loss_BA = adversarial_loss(disc_BA, valid)
        gan_loss = (gan_loss_AB + gan_loss_BA) / 2
        gen2_disc_list.append(disc_AB.data)
        gen1_disc_list.append(disc_BA.data)

        # reconstruct loss
        rec_loss_AB = criterion_recon(gen_views_B, real_views_B)
        rec_loss_BA = criterion_recon(gen_views_A, real_views_A)
        rec_loss = (rec_loss_AB + rec_loss_BA) / 2
        rec1_list.append(rec_loss_AB.data)
        rec2_list.append(rec_loss_BA.data)

        cyc_views_A = generator_2to1(gen_views_B, gen_views_B)
        cyc_views_A = topk_adj(cyc_views_A, views_A_edge_num)
        cyc_views_B = generator_1to2(gen_views_A, gen_views_A)
        cyc_views_B = topk_adj(cyc_views_B, views_B_edge_num)

        # cycle loss
        cycle_loss_A = criterion_cycle(cyc_views_A, real_views_A)
        cycle_loss_B = criterion_cycle(cyc_views_B, real_views_B)
        cycle_loss = (cycle_loss_A + cycle_loss_B) / 2
        cycle1_list.append(cycle_loss_A)
        cycle2_list.append(cycle_loss_B)

        gen_loss = gan_loss + opt.gamma * rec_loss + opt.gamma_cyc * cycle_loss

        if epoch % 2 == 0:
            gen_loss.backward()
            optimizer_G.step()
        
        # ---------------------
        #  Train Discriminator_1
        # ---------------------
        
        optimizer_d1.zero_grad()
    
        d_out = discriminator1(real_views_A, real_views_A)
        real_loss = adversarial_loss(d_out, valid)
        # print("real {}".format(d_out))
        disc1_list.append(real_loss.data)
        d_out = discriminator1(gen_views_A.detach(), gen_views_A.detach())
        fake_loss = adversarial_loss(d_out, fake)
        # print("fake {}".format(d_out))
        disc1_list.append(fake_loss.data)
    
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d1.step()
        

        # ---------------------
        #  Train Discriminator_2
        # ---------------------

        optimizer_d2.zero_grad()
    
        d_out = discriminator2(real_views_B, real_views_B)
        real_loss = adversarial_loss(d_out, valid)
        disc2_list.append(real_loss.data)
        d_out = discriminator2(gen_views_B.detach(), gen_views_B.detach())
        fake_loss = adversarial_loss(d_out, fake)
        disc2_list.append(fake_loss.data)
    
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d2.step()


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
    #val(opt, generator_1to2, generator_2to1)
# test(opt, generator_1to2, generator_2to1)

pretrain = 'pretrain_' if opt.load else ''
if not os.path.exists(f'./checkpoint/Experiments/Mygan_cycle_{pretrain}{opt.generator}/'):
    os.mkdir(f'./checkpoint/Experiments/Mygan_cycle_{pretrain}{opt.generator}/')
pickle.dump(generator_1to2, open(f'./checkpoint/Experiments/Mygan_cycle_{pretrain}{opt.generator}/generator_1to2', 'wb'))
pickle.dump(generator_2to1, open(f'./checkpoint/Experiments/Mygan_cycle_{pretrain}{opt.generator}/generator_2to1', 'wb'))
