import os
import random
import shutil
from statistics import mean
import torch
import networkx as nx

import metrics.stats

LINE_BREAK = '----------------------------------------------------------------------\n'

node_count_avg_ref, node_count_avg_pred = [], []
edge_count_avg_ref, edge_count_avg_pred = [], []
degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd = [], [], [], []
weight_mmd = []


def clean_lists():
    node_count_avg_ref.clear(); node_count_avg_pred.clear()
    edge_count_avg_ref.clear(); edge_count_avg_pred.clear()
    degree_mmd.clear(); clustering_mmd.clear(); orbit_mmd.clear(); nspdk_mmd.clear()
    weight_mmd.clear()


def print_stats():
    print('Node count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(node_count_avg_ref), mean(node_count_avg_pred)))
    print('Edge count avg: Test - {:.6f}, Generated - {:.6f}'.format(
        mean(edge_count_avg_ref), mean(edge_count_avg_pred)))

    print('MMD Degree - {:.6f}, MMD Weight - {:.6f}, MMD Clustering - {:.6f}'.format(
        mean(degree_mmd), mean(weight_mmd), mean(clustering_mmd)))
    # print('MMD Degree - {:.6f}, MMD Clustering - {:.6f}, MMD Orbits - {:.6f}'.format(
    #     mean(degree_mmd), mean(clustering_mmd), mean(orbit_mmd)))
    # print('MMD NSPDK - {:.6f}'.format(mean(nspdk_mmd)))
    print(LINE_BREAK)


def evaluate(graphs_ref, graphs_pred):
    graphs_ref = [nx.from_numpy_matrix(G, create_using=nx.DiGraph()) for G in graphs_ref]
    graphs_pred = [nx.from_numpy_matrix(G, create_using=nx.DiGraph()) for G in graphs_pred]

    node_count_avg_ref.append(mean([len(G.nodes()) for G in graphs_ref]))
    node_count_avg_pred.append(mean([len(G.nodes()) for G in graphs_pred]))

    edge_count_avg_ref.append(mean([len(G.edges()) for G in graphs_ref]))
    edge_count_avg_pred.append(mean([len(G.edges()) for G in graphs_pred]))


    degree_mmd.append(metrics.stats.degree_stats(
        graphs_ref, graphs_pred))
    weight_mmd.append(metrics.stats.weight_stats(
        graphs_ref, graphs_pred))
    clustering_mmd.append(metrics.stats.clustering_stats(
        graphs_ref, graphs_pred))
    # orbit_mmd.append(metrics.stats.orbit_stats_all(
    #     graphs_ref, graphs_pred))

    # nspdk_mmd.append(metrics.stats.nspdk_stats(graphs_ref, graphs_pred))

    # print('Running average of metrics:\n')

    # print_stats(
    #     degree_mmd, clustering_mmd, orbit_mmd, nspdk_mmd
    # )