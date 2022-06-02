# -*- coding: utf-8 -*-
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn.neighbors import NearestNeighbors

from importlib import reload

import numpy as np
import matplotlib.pyplot as plt
import torch, gudhi
import sys, time, codecs, operator

import networkx as nx
from networkx.linalg.spectrum import laplacian_spectrum

from utils import load_word_vectors, distance_matrix, compute_distance, similarity_function, select_k, get_keys_pruned, isomap, create_nx_graph

try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range


def main():
    """
    main function
    """
    
    # Get vectors first and words sorted by frequency
    en_freq, en_vec = load_word_vectors(sys.argv[1])
    de_freq, de_vec = load_word_vectors(sys.argv[2])
    
    frequency = int(sys.argv[3])
    method = sys.argv[4]
    homo_dim = 1

    if method == 'gh':
        en_matrix = distance_matrix(en_freq, en_vec, frequency)
        de_matrix = distance_matrix(de_freq, de_vec, frequency)
        print ("[INFO] Gromov-Hausdorff: ", compute_distance(en_matrix, de_matrix, homo_dim))
        
    elif method == 'ev':
        # Initialise neighborhood graphs
        en_G = nx.Graph()
        de_G = nx.Graph()
        
        # Prepare data for nearest neighbour retrieval
        en_keys, en_pruned = get_keys_pruned(en_freq, en_vec, frequency)
        de_keys, de_pruned = get_keys_pruned(de_freq, de_vec, frequency)

        # Get nearest neighbours
        nbrs_en = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(en_pruned)
        distances_en, indices_en = nbrs_en.kneighbors(en_pruned)

        nbrs_de = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(de_pruned)
        distances_de, indices_de = nbrs_de.kneighbors(de_pruned)

        for index in indices_en:
            en_G.add_edge(en_keys[index[0]], en_keys[index[1]])

        for index in indices_de:
            de_G.add_edge(de_keys[index[0]], de_keys[index[1]])

        laplacian1 = nx.spectrum.laplacian_spectrum(en_G)
        laplacian2 = nx.spectrum.laplacian_spectrum(de_G)

        k1 = select_k(laplacian1)
        k2 = select_k(laplacian2)
        k = min(k1, k2)

        similarity = sum((laplacian1[:k] - laplacian2[:k]) ** 2)
        print ("[INFO] Laplacian:", similarity)
    
    elif method == 'sgm':
        # Prepare data for nearest neighbour retrieval
        en_keys, en_pruned = get_keys_pruned(en_freq, en_vec, frequency)
        de_keys, de_pruned = get_keys_pruned(de_freq, de_vec, frequency)

        similarities_of = similarity_function(np.array(tuple(en_pruned)), 1)
        similarities_tf = similarity_function(np.array(tuple(de_pruned)), 1)
        
        W_of, D_of, res_of = isomap(['knn', 5, 1], 300, similarities = similarities_of)
        W_tf, D_tf, res_tf = isomap(['knn', 5, 1], 300, similarities = similarities_tf)

        G_of = create_nx_graph(W_of)
        G_tf = create_nx_graph(W_tf)

        laplacian_of = laplacian_spectrum(G_of)
        laplacian_tf = laplacian_spectrum(G_tf)
        k_of = select_k(laplacian_of, 0.9)
        k_tf = select_k(laplacian_tf, 0.9)
        k = min(k_of, k_tf)
        similarity = sum((laplacian_of[:k] - laplacian_tf[:k]) ** 2)

        print('[INFO] Similarity: {}'.format(similarity))
    
    else:
        print('[ERROR] Please choose a valid method (gh, ev, sgm)')
        sys.exit(0)



if __name__ == "__main__":
    main()
