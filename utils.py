import os
import numpy as np
import pandas as pd
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import Data
import scipy.sparse as sp
import random
import scanpy as sc
from operator import itemgetter
import pkg_resources
from scipy.stats import pearsonr
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import pairwise_kernels
#from torch_sparse import SparseTensor


#  load data
def load_data(data_path):
    """
        input args:
            data_path: normalize data with row: gene and column:cell
        return:
            cell * gene
    """
    data_csv = pd.read_csv(data_path, index_col=0)
    cells = data_csv.columns.values
    genes = data_csv.index.values
    data = data_csv.values.T
    return data, cells, genes

# construct graph
def get_adj(features, k=15, pca=50, mode="connectivity"):
    if pca:
        countp = dopca(features, dim=pca)
    else:
        countp = features
    A = kneighbors_graph(countp, k, mode=mode, metric="cosine", include_self=True)
    adj = A.toarray()
    adj_n = norm_adj(adj)
    adj_matrix_sp = sp.coo_matrix(adj)
    edge_index = torch.tensor(np.vstack((adj_matrix_sp.row, adj_matrix_sp.col)), dtype=torch.long)
    features = features.astype(float)
    features = torch.tensor(features, dtype=torch.float32)
    data = Data(x=features, edge_index=edge_index)
    return data

def degree_power(A, k):
    degrees = np.power(np.array(A.sum(1)), k).flatten()
    degrees[np.isinf(degrees)] = 0.
    if sp.issparse(A):
        D = sp.diags(degrees)
    else:
        D = np.diag(degrees)
    return D

def norm_adj(A):
    normalized_D = degree_power(A, -0.5)
    output = normalized_D.dot(A).dot(normalized_D)
    return output

def dopca(X, dim=10):
    pcaten = PCA(n_components=dim)
    X_10 = pcaten.fit_transform(X)
    return X_10

def impute_dropout(X, seed=None, drop_rate=0.1):
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    """

    X_zero = np.copy(X)
    i, j = np.nonzero(X_zero)
    if seed is not None:
        np.random.seed(seed)

    ix = np.random.choice(range(len(i)), int(
        np.floor(drop_rate * len(i))), replace=False)
    X_zero[i[ix], j[ix]] = 0.0

    return X_zero

def select_neighbours(hidden, k):
    num_cells = hidden.size(0)
    choose_cell = []
    for i in range(num_cells):
        # Compute cosine similarity for one cell against all others
        sim = torch.cosine_similarity(hidden[i].unsqueeze(0), hidden, dim=-1)
        sim[i] = 0.0  # Ignore self-similarity
        top_k = sim.topk(k, largest=True).indices.to('cpu').numpy()
        choose_cell.append(top_k)

    return choose_cell

def LS_imputation(drop_data, choose_cell, device, filter_noise=2):
    original_data = torch.FloatTensor(np.copy(drop_data)).to(device)
    dataImp = original_data.clone().to(device)

    for i in range(dataImp.shape[0]):
        nonzero_index = dataImp[i].nonzero()
        zero_index = (dataImp[i] == 0).nonzero()

        y = original_data[i, nonzero_index]
        x = original_data[choose_cell[i], nonzero_index]

        xtx = torch.matmul(x.T, x)
        rank = torch.linalg.matrix_rank(xtx)
        if rank != x.shape[-1]: # detect the singular matrix
            print('It is a singular matrix, use average imputation')
            return Average_imputation(drop_data, choose_cell, device, filter_noise)

        w = torch.matmul(torch.matmul(torch.linalg.inv(xtx), x.T), y)
        impute_data = torch.matmul(original_data[choose_cell[i], zero_index], w)
        impute_data[impute_data <= filter_noise] = 0   # filter noise
        dataImp[i, zero_index] = impute_data

    return dataImp.detach().cpu().numpy()

def Average_imputation(drop_data, choose_cell, device, filter_noise=2):
    original_data = torch.FloatTensor(np.copy(drop_data)).to(device)
    dataImp = original_data.clone().to(device)
    for i in range(dataImp.shape[0]):
        zero_index = (dataImp[i] == 0).nonzero()

        impute_data = torch.mean(original_data[choose_cell[i], zero_index], dim=1)
        # filter noise
        impute_data[impute_data <= filter_noise] = 0
        dataImp[i, zero_index] = impute_data.unsqueeze(-1)
    return dataImp.detach().cpu().numpy()
