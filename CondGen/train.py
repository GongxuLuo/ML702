#from lcy.CondGen.utils import Tensor, transform_adj
#from lcy.CondGen.utils import transform_adj
import os
import pickle
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict
import scipy.io as scio

from graph_stat import *
from options import Options
from GVGAN import *
from utils import *
import pprint
from evaluate import *

warnings.filterwarnings("ignore")

random.seed(321)
# torch.manual_seed(100)
# torch.cuda.manual_seed(100)


def load_data(DATA_DIR):
    dataset = 'ppmi_622'
    attr_vecs = []  # essential data, type: list(np.ndarray)

    train_view1_mats = np.load(f'./dataset/{dataset}/view1_train.npy')
    train_attr_vecs = attr_vecs[:int(len(attr_vecs) * .8)]
    train_view2_mats = np.load(f'./dataset/{dataset}/view2_train.npy')
    test1_view1_mats = np.load(f'./dataset/{dataset}/view1_test1.npy')
    test1_view2_mats = np.load(f'./dataset/{dataset}/view2_test1.npy')
    test1_attr_vecs = attr_vecs[int(len(attr_vecs) * .8):]
    test2_view1_mats = np.load(f'./dataset/{dataset}/view1_test2.npy')
    test2_view2_mats = np.load(f'./dataset/{dataset}/view2_test2.npy')
    test2_attr_vecs = attr_vecs[int(len(attr_vecs) * .8):]
    return train_view1_mats, test1_view1_mats, train_attr_vecs, test2_attr_vecs, \
        train_view2_mats, test1_view2_mats


def train(train_view1_mats, train_attr_vecs, train_view2_mats, opt=None):
    training_index = list(range(0, len(train_view1_mats)))
    torch.autograd.set_detect_anomaly(True)

    # G.train()
    # D.train()
    max_epochs = opt.max_epochs
    for epoch in range(max_epochs):
        D_real_list, D_rec_enc_list, D_rec_noise_list, D_list, Encoder_list = [], [], [], [], []
        # g_loss_list, rec_loss_list, prior_loss_list = [], [], []
        g_loss_list, rec_loss_list, prior_loss_list, aa_loss_list = [], [], [], []
        random.shuffle(training_index)
        for i in training_index:
            ones_label = Variable(torch.ones(1)).cuda()
            zeros_label = Variable(torch.zeros(1)).cuda()
            adj = Variable(torch.from_numpy(
                train_view1_mats[i]).float()).cuda()
            adj_2 = Variable(torch.from_numpy(
                train_view2_mats[i]).float()).cuda()
            adj_2_edge_number = torch.sum(adj_2 != 0).item()
            if adj.shape[0] <= opt.d_size + 2:
                continue
            if opt.av_size == 0:
                attr_vec = None
            else:
                # attr_vec = Variable(train_attr_vecs[i, :]).cuda()
                attr_vec = Variable(torch.from_numpy(
                    train_attr_vecs[i]).float()).cuda()

            G.set_attr_vec(attr_vec)
            D.set_attr_vec(attr_vec)

            norm = adj.shape[0] * adj.shape[0] / \
                float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
            pos_weight = float(
                adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            # print('pos_weight', pos_weight)

            mean, logvar, rec_adj = G(adj)

            noisev = torch.randn(mean.shape, requires_grad=True).cuda()
            #noisev = torch.where(torch.abs(noisev.data) > torch.mean(torch.abs(noisev.data)), 1 * torch.sign(noisev.data), Tensor(0))
            #noise_edge_number = torch.sum(noisev != 0).item()
            noisev = cat_attr(noisev, attr_vec)
            rec_noise = G.decoder(noisev)
            e = int(np.sum(train_view1_mats[i])) // 2

            # c_adj = topk_adj(F.sigmoid(rec_adj), e * 2)
            c_adj = F.sigmoid(rec_adj)
            c_adj = transform_adj(c_adj, adj_2_edge_number, True)
            # c_noise = topk_adj(F.sigmoid(rec_noise), e * 2)
            c_noise = F.leaky_relu(rec_noise)

            c_noise = transform_adj(c_noise, adj_2_edge_number, True)

            # train discriminator
            output = D(adj_2)
            errD_real = criterion_bce(output, ones_label)
            D_real_list.append(output.data.mean())
            # output = D(rec_adj)
            output = D(c_adj)
            errD_rec_enc = criterion_bce(output, zeros_label)
            D_rec_enc_list.append(output.data.mean())
            # output = D(rec_noise)
            output = D(c_noise)

            errD_rec_noise = criterion_bce(output, zeros_label)
            D_rec_noise_list.append(output.data.mean())

            dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
            # print ("print (dis_img_loss)", dis_img_loss)
            D_list.append(dis_img_loss.data.mean())
            # if epoch % 3 == 0:
            opt_dis.zero_grad()
            dis_img_loss.backward(retain_graph=True)
            opt_dis.step()

            # AA_loss b/w rec_adj and adj
            # aa_loss = loss_MSE(rec_adj, adj)

            loss_BCE_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_BCE_logits.cuda()

            aa_loss = loss_BCE_logits(rec_adj, adj)

            # print(c_adj,c_adj)
            # aa_loss = loss_BCE(c_adj, adj)

            # train decoder
            output = D(adj)
            errD_real = criterion_bce(output, ones_label)
            # output = D(rec_adj)
            output = D(c_adj)

            errD_rec_enc = criterion_bce(output, zeros_label)
            errG_rec_enc = criterion_bce(output, ones_label)
            # output = D(rec_noise)
            output = D(c_noise)

            errD_rec_noise = criterion_bce(output, zeros_label)
            errG_rec_noise = criterion_bce(output, ones_label)

            similarity_rec_enc = D.similarity(c_adj)
            # TODO: 结构损失
            similarity_data = D.similarity(adj_2)

            dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
            # print (dis_img_loss)
            # gen_img_loss = norm*(aa_loss + errG_rec_enc  + errG_rec_noise)- dis_img_loss #- dis_img_loss #aa_loss #+ errG_rec_enc  + errG_rec_noise # - dis_img_loss
            gen_img_loss = - errD_rec_enc  # norm*(aa_loss) #

            g_loss_list.append(gen_img_loss.data.mean())
            # print("similarity_data=",similarity_data.data)
            rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
            #rec_loss = torch.sqrt(((c_adj - adj_2) ** 2).mean())
            # rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
            rec_loss_list.append(rec_loss.data.mean())
            # err_dec =  gamma * rec_loss + gen_img_loss

            err_dec = opt.gamma * rec_loss + gen_img_loss
            opt_dec.zero_grad()

            pl = []
            for j in range(mean.size()[0]):
                prior_loss = 1 + logvar[j, :] - \
                    mean[j, :].pow(2) - logvar[j, :].exp()
                prior_loss = (-0.5 * torch.sum(prior_loss)) / \
                    torch.numel(mean[j, :].data)
                pl.append(prior_loss)
            prior_loss_list.append(sum(pl))
            err_enc = sum(pl) + gen_img_loss + opt.beta * (rec_loss)
            opt_enc.zero_grad()

            err_dec.backward(retain_graph=True)
            err_enc.backward()
            opt_dec.step()

            # train encoder
            # fix me: sum version of prior loss

            opt_enc.step()
            Encoder_list.append(err_enc.data.mean())

        print('[%d/%d]: D_real:%.4f, D_enc:%.4f, D_noise:%.4f, Loss_D:%.4f, Loss_G:%.4f, rec_loss:%.4f, prior_loss:%.4f'
              % (epoch,
                 max_epochs,
                 torch.mean(torch.stack(D_real_list)),
                 torch.mean(torch.stack(D_rec_enc_list)),
                 torch.mean(torch.stack(D_rec_noise_list)),
                 torch.mean(torch.stack(D_list)),
                 # Loss_D = -Loss_G
                 torch.mean(torch.stack(g_loss_list)),
                 torch.mean(torch.stack(rec_loss_list)),
                 torch.mean(torch.stack(prior_loss_list)))
              )


def test(test_view1_mats, test_view2_mats, opt):
    testing_index = list(range(0, len(test_view1_mats)))
    graph_ref, graph_gen = [], []
    for i in testing_index:
        with torch.no_grad():
            adj = Variable(torch.from_numpy(test_view1_mats[i]).float()).cuda()
            adj_2 = Variable(torch.from_numpy(
                test_view2_mats[i]).float()).cuda()
            adj_2_edge_number = torch.sum(adj_2 != 0).item()
            graph_ref.append(adj_2)
            rec_loss_list_for_test = []
            _, _, rec_adj = G(adj)
            c_adj = F.softsign(rec_adj)
            c_adj = transform_adj(c_adj, adj_2_edge_number, opt.is_neg)
            graph_gen.append(c_adj)
            rec_loss = F.l1_loss(c_adj, adj_2).mean()
            #rec_loss = ((c_adj - adj_2) ** 2).mean()
            rec_loss_list_for_test.append(rec_loss)
    clean_lists()
    evaluate([i.cpu().numpy() for i in graph_ref],
             [i.cpu().numpy() for i in graph_gen])
    print_stats()
    print("test: %.4f" % torch.mean(torch.stack(rec_loss_list_for_test)))
    print_graph_stat(graph_ref, '[real adj]')
    print_graph_stat(graph_gen, '[gen adj]')


if __name__ == '__main__':

    print('=========== OPTIONS ===========')
    pprint.pprint(vars(opt))
    print(' ======== END OPTIONS ========\n\n')

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    train_view1_mats, test_view1_mats, train_attr_vecs, test_attr_vecs, train_view2_mats, test_view2_mats = load_data(
        DATA_DIR=opt.DATA_DIR)

    # output_dir = opt.output_dir
    train(train_view1_mats, train_attr_vecs, train_view2_mats, opt=opt)
    test(test_view1_mats, test_view2_mats, opt)
    pickle.dump(G, open('./checkpoint/generator_1to2', 'wb'))
    pickle.dump(D, open('./checkpoint/discriminate_1to2', 'wb'))
