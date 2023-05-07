import concurrent.futures
from collections import Counter
import os
from datetime import datetime
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
import matplotlib

import metrics.mmd as mmd


matplotlib.use('Agg')
PRINT_TIME = True
MAX_WORKERS = 48


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_visual(graph_ref_list, graph_pred_list, view: str, model_name: str):
    sample_ref = []
    sample_pred = []

    # in case an empty graph is generated
    graph_pred_list = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for deg_hist in executor.map(degree_worker, graph_ref_list):
            sample_ref.append(deg_hist)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for deg_hist in executor.map(degree_worker, graph_pred_list):
            sample_pred.append(deg_hist)

    max_length = max([len(item) for item in sample_ref + sample_pred])

    def draw_degree_bar_chart(data, offset, color, label):
        view_degree_histogram = np.zeros(max_length)
        for G in data:
            G = np.array(G)
            view_degree_histogram[:G.shape[0]] += G
        view_degree_histogram /= len(data)
        
        print(view_degree_histogram)
        x = np.arange(max_length)
        y = view_degree_histogram
        plt.bar(x+offset, height=y, width=0.4, color=color, label=label, alpha=0.5)

        return view_degree_histogram

    plt.figure(figsize=(20, 4), dpi=300)
    # draw view A
    sample_ref = draw_degree_bar_chart(sample_ref, -0.2, 'r', 'origin')
    # draw view B
    sample_pred = draw_degree_bar_chart(sample_pred, 0.2, 'b', 'generate')
    plt.legend(loc='upper right')

    pic_path = f'data_gen/visual/{model_name}/'
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    plt.savefig(pic_path + f'view{view}_degree.png')

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)


def weight_worker(G):
    weight_list = []
    for (u, v, wt) in G.edges.data('weight'):
        weight_list.append(int(wt / 0.01))
    counts = Counter(weight_list)
    out = [counts.get(i, 0) for i in range(max(counts) + 1)]

    return np.array(out)
    # return np.array(nx.degree_histogram(G))


def weight_visual(graph_ref_list, graph_pred_list, view: str, model_name: str):
    """
    Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """

    sample_ref = []
    sample_pred = []

    # in case an empty graph is generated
    graph_pred_list = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for deg_hist in executor.map(weight_worker, graph_ref_list):
            sample_ref.append(deg_hist)
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for deg_hist in executor.map(weight_worker, graph_pred_list):
            sample_pred.append(deg_hist)

    max_length = max([len(item) for item in sample_ref + sample_pred])

    def draw_degree_bar_chart(data, offset, color, label):
        view_degree_histogram = np.zeros(max_length)
        for G in data:
            G = np.array(G)
            view_degree_histogram[:G.shape[0]] += G
        view_degree_histogram /= len(data)
        
        print(view_degree_histogram)
        x = np.arange(max_length)
        y = view_degree_histogram
        plt.bar(x+offset, height=y, width=0.4, color=color, label=label, alpha=0.5)

        return view_degree_histogram

    plt.figure(figsize=(20, 4), dpi=300)
    # draw view A
    sample_ref = draw_degree_bar_chart(sample_ref, -0.2, 'r', 'origin')
    # draw view B
    sample_pred = draw_degree_bar_chart(sample_pred, 0.2, 'b', 'generate')
    plt.legend(loc='upper right')

    # weight_mmd = mmd.compute_mmd(
    #     [sample_ref], [sample_pred], mmd.gaussian_emd, n_jobs=MAX_WORKERS)
    # plt.title(f'view{view}: weight mmd = {weight_mmd}')

    pic_path = f'data_gen/visual/{model_name}/'
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    plt.savefig(pic_path + f'view{view}_weight.png')

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing weight mmd: ', elapsed)


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_visual(graph_ref_list, graph_pred_list, view: str, model_name: str, bins=100):
    sample_ref = []
    sample_pred = []
    graph_pred_list = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for clustering_hist in executor.map(clustering_worker,
                                            [(G, bins) for G in graph_ref_list]):
            sample_ref.append(clustering_hist)

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for clustering_hist in executor.map(clustering_worker,
                                            [(G, bins) for G in graph_pred_list]):
            sample_pred.append(clustering_hist)

    max_length = max([len(item) for item in sample_ref + sample_pred])

    def draw_degree_bar_chart(data, offset, color, label):
        view_degree_histogram = np.zeros(max_length)
        for G in data:
            G = np.array(G)
            view_degree_histogram[:G.shape[0]] += G
        view_degree_histogram /= len(data)
        
        print(view_degree_histogram)
        x = np.arange(max_length)
        y = view_degree_histogram
        plt.bar(x+offset, height=y, width=0.4, color=color, label=label, alpha=0.5)

    plt.figure(figsize=(20, 4), dpi=300)
    # draw view A
    draw_degree_bar_chart(sample_ref, -0.2, 'r', 'origin')
    # draw view B
    draw_degree_bar_chart(sample_pred, 0.2, 'b', 'generate')
    plt.legend(loc='upper right')
    
    pic_path = f'data_gen/visual/{model_name}/'
    if not os.path.exists(pic_path):
        os.makedirs(pic_path)
    plt.savefig(pic_path + f'view{view}_cluster.png')

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
