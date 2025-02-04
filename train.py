import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, GATv2Conv
import utils
from model import Encoder, Model, drop_feature
from utils import *
import pandas as pd

class model_learning():

    def __init__(self, model:Model, data:Data):
        super(model_learning, self).__init__()
        self.model = model
        self.data = data

    def train(self):
        self.model.train()
        optimizer.zero_grad()
        edge_index_1 = dropout_adj(self.data.edge_index, p=drop_edge_rate_1)[0]
        edge_index_2 = dropout_adj(self.data.edge_index, p=drop_edge_rate_2)[0]
        x_1 = drop_feature(self.data.x, drop_feature_rate_1)
        x_2 = drop_feature(self.data.x, drop_feature_rate_2)
        z1 = self.model(x_1, edge_index_1)
        z2 = self.model(x_2, edge_index_2)
        loss = self.model.loss(z1, z2, batch_size=0)
        loss.backward()
        optimizer.step()
        return loss.item()

    # def test(self):
    #     self.model.eval()
    #     z = self.model(self.data.x, self.data.edge_index)
    #     return z

    def save_model(self):
        torch.save(self.model, data_path + args.dataset + '_model.pt')
        print("Model has saved")

    def impute(self, needImputed, if_training):
        if not if_training:
            z = np.load(data_path + args.dataset + "_embedding.npy")
            z = torch.tensor(z)
        else:
            self.model.eval()
            with torch.no_grad():
                z = self.model(self.data.x, self.data.edge_index)
        choose_cell = select_neighbours(z, k=20)
        imputed_data = LS_imputation(needImputed, choose_cell, device)
        return imputed_data, z.cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='test')
    parser.add_argument('--dropout', type=str, default='0')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    else:
        print("No GPU available, running on CPU")

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    if_training = config['if_training']
    learning_rate = config['learning_rate']
    ae_hidden = config['ae_hidden']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'GATv2Conv': GATv2Conv})[config['base_model']]
    num_layers = config['num_layers']
    pca_num = config['pca_num']
    k_neighbor = config['k_neighbor']
    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    data_path = config['data_path']


    normalize_path = data_path + "normalize_d" + args.dropout + ".csv"
    normalize, cells, genes = load_data(normalize_path)


    data = get_adj(normalize, k=k_neighbor, pca= pca_num)
    print(data.edge_index.shape)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    encoder = Encoder(data.num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    print(model)


    is_loadPr = False
    if is_loadPr:
        model = torch.load(data_path + args.dataset + '_model.pt')

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    start = t()
    prev = start
    model_learning = model_learning(model,data)

    # if if_training:
    #     for epoch in range(1, num_epochs + 1):
    #         loss = model_learning.train()
    #
    #         now = t()
    #         print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
    #               f'this epoch {now - prev:.4f}, total {now - start:.4f}')
    #         prev = now

    if if_training:
        for epoch in range(1, num_epochs + 1):
            loss = model_learning.train()
            if epoch % 10 == 0:
                now = t()
                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                      f'this epoch {now - prev:.4f}, total {now - start:.4f}')
                prev = now

        model_learning.save_model()

    print("=== Impute ===")
    needImputed = pd.read_csv(data_path + args.dataset + "_d" + args.dropout + ".csv", index_col=0).T
    imputed_data, z = model_learning.impute(needImputed, if_training)
    pd.DataFrame(imputed_data.T, index=genes, columns=cells).to_csv(data_path + "imputation.csv")
    np.save(data_path + args.dataset + "_embedding.npy", z)


