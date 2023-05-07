#from lcy.viewModel.utils import print_graph_stat
from numpy.lib.type_check import real
#from lcy.CondGen.models import Discriminator
import pickle
import argparse
import os
import numpy as np
import math
import scipy.io as scio
import warnings
from torch.utils import data

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch
from utils import *
from evaluate import *

from model import PretrainED, Encoder, Decoder, Discriminator

warnings.filterwarnings("ignore")
torch.manual_seed(100)
torch.cuda.manual_seed(100)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--v1_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr1", type=float, default=0.00006, help="adam: learning rate")
parser.add_argument("--lr2", type=float, default=0.00006, help="adam: learning rate")
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--dropout", type=float, default=0.3, help="...")
parser.add_argument("--gamma", type=int, default=15)
parser.add_argument("--dataset", type=str, default="ppmi_622", help="dataset: choose dataset by name")
parser.add_argument("--generator", type=str, default="cat", help="generator: choose generator by name")

parser.add_argument('--d_size', type=int, default=32, help="generator input size")
parser.add_argument('--h_size', type=int, default=64, help="generator hidden size")
parser.add_argument('--o_size', type=int, default=64, help="generator output size")
parser.add_argument('--rep_size', type=int, default=32, help="generator output size")

opt = parser.parse_args()
print(opt)

torch.manual_seed(100)#cpu 
torch.cuda.manual_seed(100)
data_path = './data/{}/{}.npy'

class BpDataset1(Dataset):
    def __init__(self, dataset: str, datatype: str):
        if datatype == 'train_1to1':
            self.x_data_1 = np.concatenate([np.load(data_path.format(dataset, 'view1_train')),
                                            np.load(data_path.format(dataset, 'view1_test1'))])
        elif datatype == 'train_2to2':
            self.x_data_1 = np.concatenate([np.load(data_path.format(dataset, 'view2_train')),
                                            np.load(data_path.format(dataset, 'view2_test2'))])
        elif datatype == 'test_1to1':
            self.x_data_1 = np.load(data_path.format(dataset, 'view1_test2'))
        elif datatype == 'test_2to2':
            self.x_data_1 = np.load(data_path.format(dataset, 'view2_test1'))
        else:
            assert False, 'Need correct data type name.'
        self.len = self.x_data_1.shape[0]

    def __getitem__(self, item):
        return self.x_data_1[item]

    def __len__(self):
        return self.len

# class BpDataset1(Dataset):
#     def __init__(self, dataset: str, datatype: str):
#         super(BpDataset1).__init__()
#         if datatype == 'train_1to1':
#             self.x_data = np.load('./data/{}/{}.npy'.format(dataset, 'view1_train'))
#         elif datatype == 'train_2to2':
#             self.x_data = np.load('./data/{}/{}.npy'.format(dataset, 'view2_train'))
#         elif datatype == 'test_1to1':
#             self.x_data = np.load('./data/{}/{}.npy'.format(dataset, 'view1_test'))
#         elif datatype == 'test_2to2':
#             self.x_data = np.load('./data/{}/{}.npy'.format(dataset, 'view2_test'))
#         else:
#             assert False, 'Need correct data type name.'
#         self.len = self.x_data.shape[0]
#     def __getitem__(self, index):
#         return self.x_data[index]
#     def __len__(self):
#         return self.len



def test(view: str):
    rec1_list, rec2_list = [], []
    real_A_list, gen_A_list = [], []
    real_B_list, gen_B_list = [], []
    if view == 'view1':
        for i, (view_A) in enumerate(dataTest):
            view_A = view_A[0]
            real_views_A = Variable(view_A.type(Tensor))
            real_A_list.append(real_views_A)
            views_A_elge_number = torch.sum(real_views_A != 0).item()
            with torch.no_grad():
                gen_views_A = F.relu(view1(real_views_A, real_views_A))
                gen_views_A = topk_adj(gen_views_A, views_A_elge_number, False)
                gen_A_list.append(gen_views_A)
                rec1_loss = F.l1_loss(gen_views_A, real_views_A).mean()
                rec1_list.append(rec1_loss.data)
        print('VIEW A EVAL:')
        clean_lists()
        evaluate([view.cpu().numpy() for view in real_A_list] ,[view.cpu().numpy() for view in gen_A_list])
        print_stats()
        print("Test [rec1 loss: %f]" %(torch.mean(torch.stack(rec1_list))))
        # print_graph_stat(gen_A_list, '[gen view A]')
    elif view=='view2':
        for i, (view_B) in enumerate(dataTest):
            view_B = view_B[0]
            real_views_B = Variable(view_B.type(Tensor))
            real_B_list.append(real_views_B)
            views_B_elge_number = torch.sum(real_views_B != 0).item()
            with torch.no_grad():
                gen_views_B = F.softsign(view2(real_views_B, real_views_B))
                gen_views_B = topk_adj(gen_views_B, views_B_elge_number, True)
                gen_B_list.append(gen_views_B)
                rec2_loss = F.l1_loss(gen_views_B, real_views_B).mean()
                rec2_list.append(rec2_loss)
        print('VIEW B EVAL:')
        clean_lists()
        evaluate([view.cpu().numpy() for view in real_B_list] ,[view.cpu().numpy() for view in gen_B_list])
        print_stats()
        print("Test [rec2 loss: %f]" %(torch.mean(torch.stack(rec2_list))))
        # print_graph_stat(gen_B_list, '[gen view A]')
    else:
        print("input the view")


    # for i, (views_{}.format(view)) in enumerate(dataTest):
    #     views_A = views_A[0]
    #     views_B = views_B[0]

    #     real_views_A = Variable(views_A.type(Tensor))
    #     real_views_B = Variable(views_B.type(Tensor))

    #     with torch.no_grad():
    #         gen_views_A = F.relu(view1(real_views_A))
    #         rec1_loss = F.l1_loss(gen_views_A, real_views_A).mean()
    #         rec1_list.append(rec1_loss.data)

    #         gen_views_B = F.softsign(view2(real_views_B))
    #         rec2_loss = F.l1_loss(gen_views_B, real_views_B).mean()
    #         rec2_list.append(rec2_loss.data)

    # print(
    #     "TEST [rec1 loss: %f] [rec2 loss: %f]"
    #     % (torch.mean(torch.stack(rec1_list)), torch.mean(torch.stack(rec2_list)))
    # )


cuda = True if torch.cuda.is_available() else False

view1 = PretrainED(opt.d_size, opt.h_size, opt.generator)
view2 = PretrainED(opt.d_size, opt.h_size, opt.generator)
discriminator1 = Discriminator(opt.d_size, opt.h_size, opt.rep_size, opt.dropout)
discriminator2 = Discriminator(opt.d_size, opt.h_size, opt.rep_size, opt.dropout)

if cuda:
    view1.cuda()
    view2.cuda()
    discriminator1.cuda()
    discriminator2.cuda()

adversarial_loss = torch.nn.BCELoss()
#adversarial_loss.cuda()

# Optimizers
# optimizer_view1 = torch.optim.Adam([{'params':view1.parameters()},{'params':discriminator1.parameters()}], lr=opt.lr1, betas=(opt.b1, opt.b2))
# optimizer_view2 = torch.optim.Adam([{'params':view2.parameters()}, {'params':discriminator2.parameters()}], lr=opt.lr2, betas=(opt.b1, opt.b2))
optimizer_view1 = torch.optim.Adam(view1.parameters(), lr=opt.lr1, betas=(opt.b1, opt.b2))
optimizer_view2 = torch.optim.Adam(view2.parameters(), lr=opt.lr2, betas=(opt.b1, opt.b2))
optimizer_d1 = torch.optim.Adam(discriminator1.parameters(), lr=opt.lr1, betas=(opt.b1, opt.b2))
optimizer_d2 = torch.optim.Adam(discriminator2.parameters(), lr=opt.lr1, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


dataTrain = torch.utils.data.DataLoader(
    dataset=BpDataset1(opt.dataset, datatype='train_1to1'),
    batch_size=opt.batch_size,
    shuffle=True,
)


dataTest = torch.utils.data.DataLoader(
    dataset=BpDataset1(opt.dataset, datatype='test_1to1'),
    batch_size=opt.batch_size,
)
# -----------------
#  Train view1ED
# -----------------
view1.train(); view2.train()
discriminator1.train(); discriminator2.train()
for epoch in range(opt.v1_epochs):
    rec1_list, rec2_list = [], []
    D_list = []
    # d_real_list, d_gen_list = [], []
    for i, (views_A) in enumerate(dataTrain):
        views_A = views_A[0]
        # views_B = views_B[0]
        ones_label = Variable(torch.ones(1)).cuda()
        zeros_label = Variable(torch.zeros(1)).cuda()

        real_views_A = Variable(views_A.type(Tensor))
        views_A_elge_number = torch.sum(real_views_A != 0).item()
        #real_views_B = Variable(views_B.type(Tensor))

        gen_views_A = F.leaky_relu(view1(real_views_A, real_views_A), negative_slope=0.2)
        gen_views_A = topk_adj(gen_views_A, views_A_elge_number, False)
        # output =discriminator1(real_views_A)
        # errD_real = nn.BCELoss(output, ones_label)
        # output = discriminator1(gen_views_A)
        # errD_gen = nn.BCELoss(output, zeros_label)
        # D_loss = (errD_gen + errD_real) / 2
        # D_list.append(D_loss.data.mean())
        
        # D_loss.backword()
        # optimizer_d1.step()


        optimizer_view1.zero_grad()
        gan_loss = adversarial_loss(discriminator1(gen_views_A, gen_views_A), ones_label) 
        #ctrastive_loss = discriminator1(gen_views_A) @ discriminator1(real_views_A) - discriminator1(gen_views_A) @ discriminator1()
        rec1_loss = F.l1_loss(gen_views_A, real_views_A)
        gen_loss = gan_loss + opt.gamma * rec1_loss
        rec1_list.append(rec1_loss.data)
        #rec1_loss.backward()
        gen_loss.backward()
        optimizer_view1.step()


        optimizer_d1.zero_grad()
        real_loss = adversarial_loss(discriminator1(real_views_A, real_views_A), ones_label)
        fake_loss = adversarial_loss(discriminator1(gen_views_A.detach(), gen_views_A.detach()), zeros_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d1.step()    
    #print(rec1_list)
    print(
        "[Epoch %d/%d] [rec1 loss: %f]"
        % (epoch, opt.v1_epochs, torch.mean(torch.stack(rec1_list)))
    )

test('view1')

dataTrain = torch.utils.data.DataLoader(
    dataset=BpDataset1(opt.dataset, datatype='train_2to2'),
    batch_size=opt.batch_size,
    shuffle=True,
)

dataTest = torch.utils.data.DataLoader(
    dataset=BpDataset1(opt.dataset, datatype='test_2to2'),
    batch_size=opt.batch_size,
)
# ---------------------
#  Train view2ED
# ---------------------
for epoch in range(opt.n_epochs):
    rec1_list, rec2_list = [], []
    for i, (views_B) in enumerate(dataTrain):
        #views_A = views_A[0]
        views_B = views_B[0]

        ones_label = Variable(torch.ones(1)).cuda()
        zeros_label = Variable(torch.zeros(1)).cuda()

        #real_views_A = Variable(views_A.type(Tensor))
        real_views_B = Variable(views_B.type(Tensor))
        views_B_elge_number = torch.sum(real_views_B != 0).item()
        optimizer_view2.zero_grad()

        gen_views_B = F.softsign(view2(real_views_B, real_views_B))
        gen_views_B = topk_adj(gen_views_B, views_B_elge_number, True)
        gan_loss = adversarial_loss(discriminator2(gen_views_B, gen_views_B), ones_label)     
        rec2_loss = F.l1_loss(gen_views_B, real_views_B)
        rec2_list.append(rec2_loss.data)
        gen_loss = gan_loss + opt.gamma * rec2_loss
        #rec2_loss.backward()
        gen_loss.backward()
        optimizer_view2.step()


        optimizer_d2.zero_grad()
        real_loss = adversarial_loss(discriminator2(real_views_B, real_views_B), ones_label)
        fake_loss = adversarial_loss(discriminator2(gen_views_B.detach(), gen_views_B.detach()), zeros_label)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_d2.step() 

    print(
        "[Epoch %d/%d] [rec2 loss: %f]"
        % (epoch, opt.n_epochs, torch.mean(torch.stack(rec2_list)))
    )



test('view2')

model_save_path = f'./checkpoint/pure_{opt.generator}/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

pickle.dump(view1.encoder, open(f'./checkpoint/pure_{opt.generator}/view1_encoder', 'wb'))
pickle.dump(view1.decoder, open(f'./checkpoint/pure_{opt.generator}/view1_decoder', 'wb'))
pickle.dump(view2.encoder, open(f'./checkpoint/pure_{opt.generator}/view2_encoder', 'wb'))
pickle.dump(view2.decoder, open(f'./checkpoint/pure_{opt.generator}/view2_decoder', 'wb'))
pickle.dump(discriminator1, open(f'./checkpoint/pure_{opt.generator}/discriminator1', 'wb'))
pickle.dump(discriminator2, open(f'./checkpoint/pure_{opt.generator}/discriminator2', 'wb'))
