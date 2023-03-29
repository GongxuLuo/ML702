import argparse
import os
import numpy as np
import random
import pickle
import scipy.io as scio

from models import *
from evaluate import print_stats, evaluate, clean_lists
from utils import to_adj

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch

from sklearn.metrics import accuracy_score
from pprint import pprint

random.seed(333)
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=20,
                    help="number of epochs of training")
parser.add_argument("--to1_epochs", type=int, default=20,
                    help="number of epochs of training")
parser.add_argument("--gamma", type=int, default=1,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1,
                    help="size of the batches")
parser.add_argument("--lr1", type=float, default=0.002,
                    help="adam: learning rate")
parser.add_argument("--lr2", type=float, default=0.005,
                    help="adam: learning rate")
parser.add_argument("--dropout1", type=float, default=0.4,
                    help="adam: learning rate")
parser.add_argument("--dropout2", type=float, default=0.3,
                    help="adam: learning rate")
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28,
                    help="size of each image dimension")
parser.add_argument("--k_fold", type=int, default=10,
                    help="size of each image dimension")

parser.add_argument('--d_size', type=int, default=32,
                    help="generator input size")
parser.add_argument('--h_size', type=int, default=128,
                    help="generator hidden size")
parser.add_argument('--o_size', type=int, default=64,
                    help="generator output size")
parser.add_argument('--rep_size', type=int, default=64,
                    help="generator output size")

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = nn.BCELoss()

generator_1to2 = pickle.load(open('./checkpoint/generator_1to2', 'rb'))
generator_2to1 = pickle.load(open('./checkpoint/generator_2to1', 'rb'))

def model_init():
    # Initialize generator and discriminator
    global discriminator1
    global discriminator2
    discriminator1 = Discriminator1(opt.d_size, opt.h_size, opt.rep_size, opt.dropout1)
    discriminator2 = Discriminator2(82, opt.h_size, opt.rep_size, opt.dropout2)

    if cuda:
        discriminator1.cuda()
        discriminator2.cuda()

    # Optimizers
    global optimizer_d1
    global optimizer_d2
    optimizer_d1 = torch.optim.Adam(
        discriminator1.parameters(), lr=opt.lr1, betas=(opt.b1, opt.b2))
    optimizer_d2 = torch.optim.Adam(
        discriminator2.parameters(), lr=opt.lr2, betas=(opt.b1, opt.b2))


Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Preparing
# ----------
data = scio.loadmat('./dataset/BP/BP.mat').copy()
view1 = Tensor(np.transpose(data['fmri'], [2, 0, 1]))
view2 = Tensor(np.transpose(data['dti'], [2, 0, 1]))
labels = Tensor((data['label'] + 1) / 2)
#print(labels)


if cuda:
    view1 = view1.cuda()
    view2 = view2.cuda()
    labels = labels.cuda()

index = list(range(len(view1)))
random.shuffle(index)
# train_index = index[:int(len(index) * 0.8)]
# test_index = index[int(len(index) * 0.8):]

view1_data, view2_data = [], []
for idx in index:
    with torch.no_grad():
        real_view_A, real_view_B, label = view1[idx], view2[idx], labels[idx]
        _, _, gen_view_B = generator_1to2(real_view_A)
        _, _, gen_view_A = generator_2to1(real_view_B)
        gen_view_B = F.softsign(gen_view_B)
        gen_view_A = F.leaky_relu(gen_view_A)
        view1_data.append([real_view_A, label])
        #view1_data.append([gen_view_A, to_adj(gen_view_A), label])
        view2_data.append([real_view_B, label])
        #view2_data.append([gen_view_B, to_adj(gen_view_B), label])

view1_data.extend(view2_data)



# ----------
#  Training
# ----------


def val():
    dis1_loss_list, dis2_loss_list = [], []
    predict1_list, real1_labels = [], []
    predict2_list, real2_labels = [], []
    for view_A, view_A_adj, label in view1_data[ki * slice_len: (ki+1) * slice_len]:
        with torch.no_grad():
            cls_loss = adversarial_loss(discriminator1(view_A, view_A_adj), label)
            predict1_list.append(discriminator1(view_A, view_A_adj).data.cpu().numpy())
            real1_labels.append(label.data.cpu().numpy())
            dis1_loss_list.append(cls_loss.data)
    for view_B, view_B_adj, label in view2_data[ki * slice_len: (ki+1) * slice_len]:
        with torch.no_grad():
            cls_loss = adversarial_loss(discriminator2(view_B, view_B_adj), label)
            predict2_list.append(discriminator2(view_B, view_B_adj).data.cpu().numpy())
            real2_labels.append(label.data.cpu().numpy())
            dis2_loss_list.append(cls_loss.data)
    #pprint(predict1_list)
    #pprint(real1_labels)
    print('VAL:', end=' ')
    print(
        "[cls1 loss: %f] [cls2 loss: %f] [acc1: %f] [acc2: %f]"
        % (torch.mean(torch.stack(dis1_loss_list)),
           torch.mean(torch.stack(dis2_loss_list)),
           accuracy_score(real1_labels, [1.0 if i > 0.5 else 0.0 for i in predict1_list]),
           accuracy_score(real2_labels, [1.0 if i > 0.5 else 0.0 for i in predict2_list]))
    )


def test():
    cls1_list, cls2_list = [], []
    predict1_list, real1_labels = [], []
    predict2_list, real2_labels = [], []
    for idx in test_index:
        view_A, view_B, label = view1[idx], view2[idx], labels[idx]

        with torch.no_grad():
            cls_loss = adversarial_loss(discriminator1(view_A, to_adj(view_A)), label)
            predict1_list.append(discriminator1(view_A, to_adj(view_A)).data.cpu().numpy())
            real1_labels.append(label.data.cpu().numpy())
            cls1_list.append(cls_loss.data)
            cls_loss = adversarial_loss(discriminator2(view_B, to_adj(view_B)), label)
            predict2_list.append(discriminator2(view_B, to_adj(view_B)).data.cpu().numpy())
            real2_labels.append(label.data.cpu().numpy())
            cls2_list.append(cls_loss.data)

    #print(predict1_list)
    print('TEST:', end=' ')
    print(
        "[cls1 loss: %f] [cls2 loss: %f] [acc1: %f] [acc2: %f]"
        % (torch.mean(torch.stack(cls1_list)),
           torch.mean(torch.stack(cls2_list)),
           accuracy_score(real1_labels, [1.0 if i > 0.5 else 0.0 for i in predict1_list]),
           accuracy_score(real2_labels, [1.0 if i > 0.5 else 0.0 for i in predict2_list]))
    )


slice_len = len(train_index) // opt.k_fold
for ki in range(opt.k_fold):
    model_init()
    print(f"K FOLDER {ki}:")
    for epoch in range(opt.n_epochs):
        cls1_list, cls2_list = [], []
        predict1_list, real1_labels = [], []
        predict2_list, real2_labels = [], []
        for view_A, view_A_adj, label in view1_data[: ki * slice_len] + view1_data[(ki+1) * slice_len:]:

            optimizer_d1.zero_grad()
            cls_loss = adversarial_loss(discriminator1(view_A, view_A_adj), label)
            predict1_list.append(discriminator1(view_A, view_A_adj).data.cpu().numpy())
            real1_labels.append(label.data.cpu().numpy())
            cls1_list.append(cls_loss.data)

            cls_loss.backward()
            optimizer_d1.step()

        for view_B, view_B_adj, label in view2_data[: ki * slice_len] + view2_data[(ki+1) * slice_len:]:

            optimizer_d2.zero_grad()

            cls_loss = adversarial_loss(discriminator2(view_B, view_B_adj), label)
            predict2_list.append(discriminator2(view_B, view_B_adj).data.cpu().numpy())
            real2_labels.append(label.data.cpu().numpy())
            cls2_list.append(cls_loss.data)

            cls_loss.backward()
            optimizer_d2.step()

        print("[Epoch %d/%d] [cls1 loss: %f] [cls2 loss: %f] [acc1: %f] [acc2: %f]"
              % (epoch, opt.n_epochs, torch.mean(torch.stack(cls1_list)), torch.mean(torch.stack(cls2_list)),
              accuracy_score(real1_labels, [1.0 if i > 0.5 else 0.0 for i in predict1_list]),
              accuracy_score(real2_labels, [1.0 if i > 0.5 else 0.0 for i in predict2_list])))

        val()
        test()


test()
