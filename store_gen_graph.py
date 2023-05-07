import numpy as np
import random
import pickle
import torch

from model import *
from dataset import BpDataset

generator_1to2 = pickle.load(open('./checkpoint/generator_1to2', 'rb'))
generator_2to1 = pickle.load(open('./checkpoint/generator_2to1', 'rb'))

generator_1to2.eval()
generator_2to1.eval()

dataTrain = torch.utils.data.DataLoader(
    dataset=BpDataset(train='train', K=1),
    batch_size=1,
)
dataTest = torch.utils.data.DataLoader(
    dataset=BpDataset(train='test', K=1),
    batch_size=1,
)
weight = generator_1to2.decoder.weight_up

gen_view1_list, gen_view2_list = [], []
for i, (views_A, views_B, labels) in enumerate(dataTrain):
    view_A = views_A[0]
    view_B = views_B[0]
    label = labels[0]

    with torch.no_grad():
        gen_view_B = generator_1to2(view_A)
        gen_view_A = generator_2to1(view_B)
        tmp = generator_1to2.translator(generator_1to2.encoder(view_A))
        tmp = tmp @ tmp.T
        # tmp = (tmp @ weight) @ tmp.T
        if i == 0:
            np.savetxt('trans_tmp.txt', tmp.data.cpu().numpy(), fmt='%.5f')
    gen_view1_list.append(gen_view_A.data.cpu().numpy())
    gen_view2_list.append(gen_view_B.data.cpu().numpy())

with open('./data/gen_graph/train_graph_view1.npy', 'wb') as f:
    np.save(f, np.array(gen_view1_list))
with open('./data/gen_graph/train_graph_view2.npy', 'wb') as f:
    np.save(f, np.array(gen_view2_list))

gen_view1_list, gen_view2_list = [], []
for i, (views_A, views_B, labels) in enumerate(dataTest):
    view_A = views_A[0]
    view_B = views_B[0]
    label = labels[0]

    with torch.no_grad():
        gen_view_B = generator_1to2(view_A)
        gen_view_A = generator_2to1(view_B)
    gen_view1_list.append(gen_view_A.data.cpu().numpy())
    gen_view2_list.append(gen_view_B.data.cpu().numpy())

with open('./data/gen_graph/test_graph_view1.npy', 'wb') as f:
    np.save(f, np.array(gen_view1_list))
with open('./data/gen_graph/test_graph_view2.npy', 'wb') as f:
    np.save(f, np.array(gen_view2_list))
