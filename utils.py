import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset




def load_adj(adj_file):
    ddi_label = pd.read_csv(adj_file, dtype=int)
    ddi_arr = ddi_label.iloc[:, [0, 1]].values
    label = ddi_label.iloc[:, 2].values
    return ddi_arr,label

def load_graph(adj_file):
    # load ddi_matrix to get ddi_array and label
    ddi_arr,_=load_adj(adj_file)
    adj = sp.coo_matrix((np.ones( ddi_arr.shape[0]), ( ddi_arr[:, 0], ddi_arr[:, 1])),
                        shape=(572, 572), dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class load_data(Dataset):
    def __init__(self, dataset):
        #use self_attention getting STE_feature.txt
        DF=pd.read_csv('data/ATT_and_DMDDI_result/STE_self_and_attention_1716.csv', dtype=float, header=None, index_col=0)
        DF=DF.values
        print("STE.shape",DF.shape)
        self.x=DF

        # self.x = np.loadtxt('./data/STE_feature.txt', dtype=float)
###when test different ddi class dateset,need to introduce  different file "ddi_class_X.csv"
        # y = pd.read_csv('./graph/ddi_class_3.csv', dtype=int)
        # label=y.iloc[:,2].values # (33214,)
        # self.y=label


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))






