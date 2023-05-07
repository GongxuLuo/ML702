from operator import ge
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.manifold import SpectralEmbedding
import numpy as np
from pprint import pprint

from evaluate import print_stats, evaluate, clean_lists
from dataset import BpDataset, HivDataset, PpmiDataset, Ppmi622Dataset
from graph_stat import compute_graph_statistics


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def get_spectral_embedding(adj, d):
    adj_ = adj.data.cpu().numpy()
    emb = SpectralEmbedding(n_components=d)
    res = emb.fit_transform(adj_)
    x = torch.from_numpy(res).float().cuda()
    return x


def normalize(adj):
    adj = adj.data.cpu().numpy()
    operation = np.where(adj>=0, 1, -1)
    adj = np.abs(adj)
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    np.set_printoptions(threshold=np.inf)
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    degree_mat_sqrt = np.diag(np.power(rowsum, 0.5).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).dot(degree_mat_sqrt)
    adj_normalized = np.multiply(adj_normalized, operation)
    return torch.from_numpy(adj_normalized).float().cuda()


def cat_mtrx(xt, xd):
	return torch.triu(xt, diagonal=0) + torch.triu(xd, diagonal=1)


def print_graph_stat(gen_list, info):
    statics = None
    for graph in gen_list:
        if statics is None:
            statics = compute_graph_statistics(graph.cpu().numpy())
        else:
            tmp = compute_graph_statistics(graph.cpu().numpy())
            for k in tmp:
                statics[k] += tmp[k]
    for k in statics:
        statics[k] /= len(gen_list)
    print(info)
    pprint(statics)


def test(opt, generator_1to2, generator_2to1):
    generator_1to2.eval(); generator_2to1.eval()
    clean_lists()
    rec1_list, rec2_list = [], []
    real_A_list, real_B_list, gen_A_list, gen_B_list = [], [], [], []

    if opt.dataset == 'bp':
        dataset = BpDataset(train='test', K=opt.k_fold)
    elif opt.dataset == 'hiv':
        dataset = HivDataset(train='test')
    elif opt.dataset == 'ppmi':
        dataset = PpmiDataset(train='test')
    elif opt.dataset == 'ppmi_622':
        test_622(opt, generator_1to2, generator_2to1)
        return
    dataTest = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
    )
    for i, (views_A, views_B, _) in enumerate(dataTest):
        views_A = views_A[0]
        views_B = views_B[0]
        views_A_edge_num = torch.sum(views_A != 0).item()
        views_B_edge_num = torch.sum(views_B != 0).item()

        real_views_A = Variable(views_A.type(Tensor))
        real_views_B = Variable(views_B.type(Tensor))
        real_B_list.append(real_views_B)
        real_A_list.append(real_views_A)

        with torch.no_grad():
            gen_views_B = generator_1to2(real_views_A, real_views_A)
            gen_views_B = topk_adj(gen_views_B, views_B_edge_num, opt.is_neg)
            gen_B_list.append(gen_views_B)
            rec_loss = F.l1_loss(gen_views_B, real_views_B)
            rec1_list.append(rec_loss.data)
            # for item in gen_B_list:
            #     for i in range(len(item)):
            #         for j in range(len(item)):
            #             if torch.abs(item[i,j]) >= 0.5:
            #                 item[i,j] = torch.sign(item[i,j]) * 1
            #             else:
            #                 item[i,j] = 0

            #gen_B_list[torch.abs(gen_B_list) >= 0.5] = torch.sign(gen_B_list) * 1 
            #gen_B_list[torch.abs(gen_B_list) < 0.5] = 0
            gen_views_A = generator_2to1(real_views_B, real_views_B)
            gen_views_A = topk_adj(gen_views_A, views_A_edge_num, False)
            gen_A_list.append(gen_views_A)
            rec_loss = F.l1_loss(gen_views_A, real_views_A)
            rec2_list.append(rec_loss.data)
            '''for item in gen_A_list:
                for i in range(len(item)):
                    for j in range(len(item)):
                        if item[i,j] >= 0.5:
                            item[i,j] = 1
                        else:
                            item[i,j] = 0'''
            # gen_A_list[gen_A_list >= 0.5] = 1
            # gen_A_list[gen_A_list < 0.5] = 0
    
    # dataTest = torch.utils.data.DataLoader(
    #     dataset=BpDataset(train='test2'),
    #     batch_size=opt.batch_size,
    # )
    # for i, (views_A, views_B, _) in enumerate(dataTest):
    #     views_A = views_A[0]
    #     views_B = views_B[0]

    #     real_views_A = Variable(views_A.type(Tensor))
    #     real_views_B = Variable(views_B.type(Tensor))
    #     real_B_list.append(real_views_B)
    #     real_A_list.append(real_views_A)

    #     with torch.no_grad():
    #         gen_views_A = generator_2to1(real_views_B)
    #         gen_A_list.append(gen_views_A)
    #         rec_loss = opt.gamma * F.l1_loss(gen_views_A, real_views_A)
    #         rec2_list.append(rec_loss.data)

    print('VIEW A EVAL:')
    clean_lists()
    evaluate([view.cpu().numpy() for view in real_A_list] ,[view.cpu().numpy() for view in gen_A_list])
    print_stats()
    print('VIEW B EVAL:')
    clean_lists()
    evaluate([view.cpu().numpy() for view in real_B_list] ,[view.cpu().numpy() for view in gen_B_list])
    print_stats()
    print('TEST:')
    print(
        "[rec1 loss: %f] [rec2 loss: %f]"
        % (torch.mean(torch.stack(rec1_list)), torch.mean(torch.stack(rec2_list)))
    )
    # print_graph_stat(real_A_list, '[real view A]')
    # print_graph_stat(gen_A_list, '[gen view A]')
    # print_graph_stat(real_B_list, '[real view B]')
    # print_graph_stat(gen_B_list, '[gen view B]')


def test_copy(opt, generator_1to2, generator_2to1):
    clean_lists()
    rec1_list, rec2_list = [], []
    real_A_list, real_B_list, gen_A_list, gen_B_list = [], [], [], []

    if opt.dataset == 'bp':
        dataset = BpDataset(train='test', K=opt.k_fold)
    elif opt.dataset == 'hiv':
        dataset = HivDataset(train='test')
    elif opt.dataset == 'ppmi':
        dataset = PpmiDataset(train='test')
    dataTest = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
    )
    for i, (views_A, views_B, _) in enumerate(dataTest):
        views_A = views_A[0]
        views_B = views_B[0]
        views_A_edge_num = torch.sum(views_A != 0).item()
        views_B_edge_num = torch.sum(views_B != 0).item()

        real_views_A = Variable(views_A.type(Tensor))
        real_views_B = Variable(views_B.type(Tensor))
        real_B_list.append(real_views_B)
        real_A_list.append(real_views_A)

        with torch.no_grad():
            gen_views_B,_,_= generator_1to2(real_views_A, real_views_A)
            gen_views_B = topk_adj(gen_views_B, views_B_edge_num, opt.is_neg)
            gen_B_list.append(gen_views_B)
            rec_loss = F.l1_loss(gen_views_B, real_views_B)
            rec1_list.append(rec_loss.data)
            # for item in gen_B_list:
            #     for i in range(len(item)):
            #         for j in range(len(item)):
            #             if torch.abs(item[i,j]) >= 0.5:
            #                 item[i,j] = torch.sign(item[i,j]) * 1
            #             else:
            #                 item[i,j] = 0

            #gen_B_list[torch.abs(gen_B_list) >= 0.5] = torch.sign(gen_B_list) * 1 
            #gen_B_list[torch.abs(gen_B_list) < 0.5] = 0
            gen_views_A,_,_= generator_2to1(real_views_B, real_views_B)
            gen_views_A = topk_adj(gen_views_A, views_A_edge_num, False)
            gen_A_list.append(gen_views_A)
            rec_loss = F.l1_loss(gen_views_A, real_views_A)
            rec2_list.append(rec_loss.data)
            '''for item in gen_A_list:
                for i in range(len(item)):
                    for j in range(len(item)):
                        if item[i,j] >= 0.5:
                            item[i,j] = 1
                        else:
                            item[i,j] = 0'''
            # gen_A_list[gen_A_list >= 0.5] = 1
            # gen_A_list[gen_A_list < 0.5] = 0
    
    # dataTest = torch.utils.data.DataLoader(
    #     dataset=BpDataset(train='test2'),
    #     batch_size=opt.batch_size,
    # )
    # for i, (views_A, views_B, _) in enumerate(dataTest):
    #     views_A = views_A[0]
    #     views_B = views_B[0]

    #     real_views_A = Variable(views_A.type(Tensor))
    #     real_views_B = Variable(views_B.type(Tensor))
    #     real_B_list.append(real_views_B)
    #     real_A_list.append(real_views_A)

    #     with torch.no_grad():
    #         gen_views_A = generator_2to1(real_views_B)
    #         gen_A_list.append(gen_views_A)
    #         rec_loss = opt.gamma * F.l1_loss(gen_views_A, real_views_A)
    #         rec2_list.append(rec_loss.data)

    print('VIEW A EVAL:')
    clean_lists()
    evaluate([view.cpu().numpy() for view in real_A_list] ,[view.cpu().numpy() for view in gen_A_list])
    print_stats()
    print('VIEW B EVAL:')
    clean_lists()
    evaluate([view.cpu().numpy() for view in real_B_list] ,[view.cpu().numpy() for view in gen_B_list])
    print_stats()
    print('TEST:')
    print(
        "[rec1 loss: %f] [rec2 loss: %f]"
        % (torch.mean(torch.stack(rec1_list)), torch.mean(torch.stack(rec2_list)))
    )
    # print_graph_stat(real_A_list, '[real view A]')
    # print_graph_stat(gen_A_list, '[gen view A]')
    # print_graph_stat(real_B_list, '[real view B]')
    # print_graph_stat(gen_B_list, '[gen view B]')


def to_adj(old_adj):
    new_adj = old_adj.cpu().numpy().copy()
    new_adj[new_adj != 0] = 1
    return Tensor(new_adj)


def adj(old_adj, view):
    if view == 'view1':
        '''for i in range(len(old_adj)):
            for j in range(len(old_adj)):
                if old_adj[i,j] < 0.5:
                    old_adj.data[i,j] = 0
                else:
                    old_adj.data[i,j] = 1'''
        torch.where(old_adj.data < 0.5, Tensor([0.0]), Tensor([1.0]))
    else:
        '''for i in range(len(old_adj)):
            for j in range(len(old_adj)):
                if torch.absolute(old_adj[i,j])< 0.5:
                    old_adj.data[i,j]= 0
                else:
                    old_adj.data[i,j]= torch.sign(old_adj.data[i, j]) * 1'''
        torch.where((old_adj.data < 0.5) & (old_adj.data > -0.5), Tensor([0.0]), 1.0 * torch.sign(old_adj.data))

    return old_adj


def top_n_indexes(arr, n):
	idx = np.argpartition(arr, arr.size - n, axis=None)[-n:] # set arr.size -n as partition point
	width = arr.shape[1]
	return [divmod(i, width) for i in idx]


def topk_adj(adj, k, has_neg = False):
    adj_ = adj.data.cpu().numpy()
    if has_neg:
        adj_sign = adj_
    else:
        adj_sign = np.abs(adj_)
    # adj_ = (adj_ - np.min(adj_)) / np.ptp(adj_)
    # adj_ -= np.diag(np.diag(adj_))
    # adj_ = np.triu(adj_)
    if has_neg:
        adj_ = np.abs(adj_)
    inds = top_n_indexes(adj_, k)
    adj.data[:,:] = 0
    for i, j in inds:
        adj.data[i, j] = adj_sign[i, j].tolist()
    return adj


def test_622(opt, generator_1to2, generator_2to1):
    generator_1to2.eval(); generator_2to1.eval()
    clean_lists()
    rec1_list, rec2_list = [], []
    real_A_list, real_B_list, gen_A_list, gen_B_list = [], [], [], []

    if opt.dataset != 'ppmi_622':
        print('must ppmi_622')
        exit(0)
        
    dataTest1 = torch.utils.data.DataLoader(
        dataset=Ppmi622Dataset(train='test1'),
        batch_size=1,
    )
    dataTest2 = torch.utils.data.DataLoader(
        dataset=Ppmi622Dataset(train='test2'),
        batch_size=1,
    )
    for i, (views_A, views_B, _) in enumerate(dataTest2):
        views_A = views_A[0]
        views_B = views_B[0]
        views_A_edge_num = torch.sum(views_A != 0).item()
        # views_B_edge_num = torch.sum(views_B != 0).item()

        real_views_A = Variable(views_A.type(Tensor))
        real_views_B = Variable(views_B.type(Tensor))
        # real_B_list.append(real_views_B)
        real_A_list.append(real_views_A)

        with torch.no_grad():
            '''
            gen_views_B = generator_1to2(real_views_A, real_views_A)
            gen_views_B = topk_adj(gen_views_B, views_B_edge_num, opt.is_neg)
            gen_B_list.append(gen_views_B)
            rec_loss = F.l1_loss(gen_views_B, real_views_B)
            rec1_list.append(rec_loss.data)
            '''
            gen_views_A = generator_2to1(real_views_B, real_views_B)
            gen_views_A = topk_adj(gen_views_A, views_A_edge_num)
            gen_A_list.append(gen_views_A)
            rec_loss = F.l1_loss(gen_views_A, real_views_A)
            rec1_list.append(rec_loss.data)

    for i, (views_A, views_B, _) in enumerate(dataTest1):
        views_A = views_A[0]
        views_B = views_B[0]
        # views_A_edge_num = torch.sum(views_A != 0).item()
        views_B_edge_num = torch.sum(views_B != 0).item()

        real_views_A = Variable(views_A.type(Tensor))
        real_views_B = Variable(views_B.type(Tensor))
        real_B_list.append(real_views_B)
        # real_A_list.append(real_views_A)

        with torch.no_grad():
            gen_views_B = generator_1to2(real_views_A, real_views_A)
            gen_views_B = topk_adj(gen_views_B, views_B_edge_num, opt.is_neg)
            gen_B_list.append(gen_views_B)
            rec_loss = F.l1_loss(gen_views_B, real_views_B)
            rec2_list.append(rec_loss.data)
            '''
            gen_views_A = generator_2to1(real_views_B, real_views_B)
            gen_views_A = topk_adj(gen_views_A, views_A_edge_num)
            gen_A_list.append(gen_views_A)
            rec_loss = F.l1_loss(gen_views_A, real_views_A)
            rec2_list.append(rec_loss.data)
            '''

    print('VIEW A EVAL:')
    clean_lists()
    evaluate([view.cpu().numpy() for view in real_A_list] ,[view.cpu().numpy() for view in gen_A_list])
    print_stats()
    print('VIEW B EVAL:')
    clean_lists()
    evaluate([view.cpu().numpy() for view in real_B_list] ,[view.cpu().numpy() for view in gen_B_list])
    print_stats()
    print('TEST:')
    print(
        "[rec1 loss: %f] [rec2 loss: %f]"
        % (torch.mean(torch.stack(rec1_list)), torch.mean(torch.stack(rec2_list)))
    )
    # print_graph_stat(real_A_list, '[real view A]')
    # print_graph_stat(gen_A_list, '[gen view A]')
    # print_graph_stat(real_B_list, '[real view B]')
    # print_graph_stat(gen_B_list, '[gen view B]')


def val(opt, generator_1to2, generator_2to1):
    generator_1to2.eval(); generator_2to1.eval()
    recAB_list, recBA_list = [], []

    if opt.dataset == 'bp':
        dataset = BpDataset(train='test', K=opt.k_fold)
    elif opt.dataset == 'hiv':
        dataset = HivDataset(train='test')
    elif opt.dataset == 'ppmi':
        dataset = PpmiDataset(train='test')
    elif  opt.dataset == 'ppmi_622':
        dataset = PpmiDataset(train='test')
    dataTest = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
    )
    for i, (views_A, views_B, _) in enumerate(dataTest):
        views_A = views_A[0]
        views_B = views_B[0]
        views_A_edge_num = torch.sum(views_A != 0).item()
        views_B_edge_num = torch.sum(views_B != 0).item()

        real_views_A = Variable(views_A.type(Tensor))
        real_views_B = Variable(views_B.type(Tensor))

        with torch.no_grad():
            gen_views_B = generator_1to2(real_views_A, real_views_A)
            gen_views_B = topk_adj(gen_views_B, views_B_edge_num, opt.is_neg)
            rec_loss = F.l1_loss(gen_views_B, real_views_B)
            recAB_list.append(rec_loss.data)

            gen_views_A = generator_2to1(real_views_B, real_views_B)
            gen_views_A = topk_adj(gen_views_A, views_A_edge_num, False)
            rec_loss = F.l1_loss(gen_views_A, real_views_A)
            recBA_list.append(rec_loss.data)
        
    print(
        "---> [recA2B loss: %f] [recB2A loss: %f] <---"
        % (torch.mean(torch.stack(recAB_list)), torch.mean(torch.stack(recBA_list)))
    )
 