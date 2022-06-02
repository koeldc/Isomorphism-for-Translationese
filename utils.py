# -*- coding: utf-8 -*-
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import dijkstra
from importlib import reload

import numpy as np
import torch, gudhi
import sys, time, codecs, operator

import networkx as nx
try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range



def load_word_vectors(filepath):
    print("Loading vectors from", filepath)
    input_dic = {}

    with open(filepath, "r", encoding='utf-8') as in_file:
        lines = in_file.readlines()

    words = []
    vectors = []
    for line in lines[1:]:
        item = line.strip().split()
        dkey = item.pop(0)
        words.append(dkey)
        vector = np.array(item, dtype='float32')
        vectors.append(vector)

    npvectors = np.vstack(vectors)

    # Our words are stored in the list words and...
    # ...our vectors are stored in the 2D array npvectors

    # 1. Length normalize
    npvectors = normalize(npvectors, axis=1, norm='l2')

    # 2. Mean centering dimesionwise
    npvectors = npvectors - npvectors.mean(0)

    # 3. Length normalize again
    npvectors = normalize(npvectors, axis=1, norm='l2')

    for i in xrange(len(words)):
        word = words[i]
        vector = npvectors[i]
        input_dic[word] = vector

    print(len(input_dic),"vectors loaded from", filepath)
    return words, input_dic



def distance_matrix(xx_freq, xx_vec, frequency):
    xx_vectors = []
    for word in xx_freq[:frequency]:
        xx_vectors.append(xx_vec[word])
    xx_embed_temp = np.vstack(xx_vectors)
    xx_embed = torch.from_numpy(xx_embed_temp)
    xx_dist = torch.sqrt(2 - 2 * torch.clamp(torch.mm(xx_embed, torch.t(xx_embed)), -1., 1.))
    xx_matrix = xx_dist.cpu().numpy()
    return xx_matrix



def select_k(spectrum, minimum_energy = 0.9):
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)



def get_keys_pruned(frequency_list, vector, frequency):
    keys = []
    pruned = []
    for word in frequency_list[:frequency]:
        keys.append(word)
        pruned.append(vector[word])
    return keys, pruned

# GH specific functions

def compute_diagram(x, homo_dim = 1):
    rips_tree = gudhi.RipsComplex(x).create_simplex_tree(max_dimension = homo_dim)
    rips_diag = rips_tree.persistence()
    return [rips_tree.persistence_intervals_in_dimension(w) for w in range(homo_dim)]



def compute_distance(x, y, homo_dim):
    start_time = time.time()
    diag_x = compute_diagram(x, homo_dim=homo_dim)
    diag_y = compute_diagram(y, homo_dim=homo_dim)
    return min([gudhi.bottleneck_distance(x, y, e=0) for (x, y) in zip(diag_x, diag_y)])

# SGM specific functions

def similarity_function(X, sigma):
    N = len(X)
    dis = np.zeros((N,N))
    for i in range(N):
        dis[i,i] = 0
        for j in range(i+1,N):
            dis[i,j] = np.linalg.norm(X[i,:]-X[j,:])
            dis[j,i] = dis[i,j]
    return dis



def build_similarity_graph(graph_parameters,similarities):
    grf_type = graph_parameters[0]
    grf_thresh = graph_parameters[1]

    N = len(similarities)
    W = np.zeros((N,N));

    if grf_type == 'knn':
        tmp = np.ones((N,N))
        for i in range(N):
            ind = np.argsort(similarities[i,:])
            tmp[i,ind[grf_thresh+1:N]] = 0
        tmp = tmp + tmp.transpose()
        tmp = tmp != 0
        W = np.multiply(similarities,tmp)

    elif grf_type == 'eps':
        W = similarities
        W[W < grf_thresh] = 0
    else:
        print('Cannot recognize the type of graph')
    return W



def eigenvector(D,d):
    N = len(D)
    H = np.identity(N)-float(1)/N*np.ones((N,N))
    tau_D = -0.5*H.dot(np.multiply(D,D).dot(H))
    [eigval,eigvec] = np.linalg.eig(tau_D)
    sqrt_eigval = np.sqrt(eigval)
    res = eigvec.dot(np.diag(sqrt_eigval))
    return res[:,0:d]



def isomap(graph_parameters,d, data = None, similarities = None, normalize_data = True):
    if data is None:
        if similarities is None:
            print ("Isomap needs data")
    else:
        if normalize_data:
            print("----Normalization step----")
            data_norm = normalize(data, axis = 1)
        data_norm = data
        similarities = similarity_function(data_norm,1)
    print("----Building graph----")
    W = build_similarity_graph(graph_parameters,similarities)
    print("----Computing short paths----")
    D = dijkstra(W, directed = True)
    print("----Computing embeddings----")
    res = eigenvector(D,d)
    return W, D, res



def create_nx_graph(W):
    G = nx.Graph()

    for i, row in enumerate(W):
        i_h, j_h = 0, 0
        tmp_high = 0.0
        for j, item in enumerate(row):
            if (i != j):
                if (tmp_high < W[i,j]):
                    tmp_high = W[i,j]
                    i_h = i
                    j_h = j
        G.add_weighted_edges_from([(en_freq[i_h], de_freq[j_h], W[i_h,j_h])])
    return G
