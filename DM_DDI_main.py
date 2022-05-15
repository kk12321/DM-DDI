# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
from __future__ import print_function, division
import argparse
import csv
import os
from random import random

from keras import Input, Model
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import label_binarize
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from utils import load_data, load_graph
from GNN import GNNLayer
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_curve, auc, \
    roc_auc_score
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"
from utils import load_adj
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 设置随机数种子
setup_seed(3407)
# 预处理数据以及训练模型
# ..
loss = torch.nn.CrossEntropyLoss()

event_num = 65   #千万要注意这个event_num,不然维度可能不一致
droprate = 0.3
vector_size = 572

def DNN():

    # train_input = Input(shape=(vector_size * 2,), name='Inputlayer') #输入层 vector_size=572
    train_input = Input(shape=(130,), name='Inputlayer') #药物嵌入是65列时，为129，当为100维度时，设置200  attention sdcn is 128 ,the others is 100*2
    # train_input = Input(shape=(256,), name='Inputlayer') #药物嵌入是65列
    # train_input = Input(shape=(572,130), name='Inputlayer')  # 输入层 572*2
    train_in = Dense(512, activation='relu')(train_input)  #train_input输入，没有作为Dense（X...）参数
    train_in = BatchNormalization()(train_in) #批量标准化层（）
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(256, activation='relu')(train_in) #全连接层，用relu做激活函数
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)
    train_in = Dense(event_num)(train_in)
    out = Activation('softmax')(train_in) #输出用softmax作多分类
    model = Model(train_input, out) #input=train_input, output=out 用交叉损失来训练，不像机器学习中多分类
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

class AE(nn.Module):

    def __init__(self,n_input, n_enc_1,n_enc_2,n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.z_layer = Linear(n_enc_2,n_z)

        self.dec_1 = Linear(n_z,n_enc_2)
        self.dec_2 = Linear(n_enc_2, n_enc_1)
        self.x_bar_layer = Linear(n_enc_1, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        z = self.z_layer(enc_h2)  # 编码 3
        dec_h1 = F.relu(self.dec_1(z))  # 解码
        dec_h2 = F.relu(self.dec_2(dec_h1))
        x_bar = self.x_bar_layer(dec_h2)
        return x_bar, enc_h1, enc_h2, z

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z) #(572,2,1)
        beta = torch.softmax(w, dim=1) #(572,2,1),go for softmax in column
        return (beta * z).sum(1), beta  #sum[(572,2,1)*(572,2,3)=>(572,2,3)]===>(572,3);(572,2,1)

class DM_DDI(nn.Module):
    def __init__(self, n_enc_1, n_enc_2,
                 n_input, n_z, v=1):
        super(DM_DDI, self).__init__()
        self.ae = AE(
        n_input=1716,
        n_enc_1=2000,
        n_enc_2=256,
        n_z=args.n_z,

        )
        self.ae.load_state_dict(
            torch.load(args.pretrain_path, map_location='cpu'))
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_enc_2)
        self.gnn_3 = GNNLayer(n_enc_2,n_z)

        self.attention = Attention(n_z)
        self.predict = torch.nn.Linear(2 * n_z, n_z)
        self.cluster_layer = Parameter(torch.Tensor(n_z, n_enc_2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self.v = v

    def forward(self, x, adj):
        #AE Module
        x_bar,tra1, tra2, z = self.ae(x)
        sigma = 0.5
        # GCN Module
        h1 = self.gnn_1(x, adj)
        h2 = self.gnn_2((1 - sigma) * h1 + sigma * tra1, adj)
        h3 = self.gnn_3((1 - sigma) * h2 + sigma * tra2, adj, active=False)
        emb=torch.stack([h3,z],dim=1) #(572,2,65)
        emb1,att1=self.attention(emb) #(572,3)  (572,2,1)#use attention finish the final fusion
        C1, C2, label_train_y, label_test_y = combine_drugpairs(emb1,1)
        return emb1,att1,x_bar,C1, C2, label_train_y, label_test_y

def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    y_one_hot = label_binarize(y_test, np.arange(event_num))
    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    return result_all

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)  # 把每一类进行求精度得分，然后再求平均
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

#combine embedding of drug pairs
def combine_part_pairs(embedding,array,method_num):
    feature_pairs_train = []
    for i, j in enumerate(array):
            leftdrug = j[0]
            rightdrug = j[1]
            leftdrug = embedding[leftdrug]
            rightdrug = embedding[rightdrug]
            if method_num == 1:
                B = ( leftdrug+ rightdrug) / 2
            elif method_num == 2:
                B = ( leftdrug*  rightdrug)
            elif method_num == 3:
                B = (leftdrug - rightdrug)
            elif method_num == 4:
                B = torch.cat((leftdrug, rightdrug))
            feature_pairs_train.append(B)
    embedding_pairs=torch.stack(feature_pairs_train)  # stack the list to form tensor（37264,65）
    return embedding_pairs

def combine_drugpairs(embeding_drug,method_num):
    ddi_arr,label=load_adj(adj_file) #traverse adj matrix to get ddi_arr and label
    label = torch.LongTensor(label)
    all_edge_array = ddi_arr
    num_train = int(np.floor(len(all_edge_array) * 0.8))
    train_edge_array = all_edge_array[:num_train]  # 29811  from 0->29810
    test_edge_array = all_edge_array[num_train:]  # 7453    from 0->7452
    label_train_y = label[:num_train]  # get train_y
    C1=combine_part_pairs(embeding_drug,train_edge_array,method_num)
    label_test_y = label[num_train:]   # get test_y
    C2 = combine_part_pairs(embeding_drug,test_edge_array, method_num)
    return C1, C2, label_train_y, label_test_y


def train_dmddi(dataset):
    model = DM_DDI(2000,256,
                 n_input=args.n_input,
                 n_z=args.n_z,
                 v=1.0).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    adj = load_graph(adj_file)
    data = torch.Tensor(dataset.x).to(device)  #get three drug's feature data after combination (572,1716)
    train_losses = []
    test_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)


    for epoch in range(1000):
        model.train()
        emb,att,x_bar,C1, C2, label_train_y, label_test_y= model(data, adj)
        #att(572,2,1)is two drug'account, we can level it and analysis the whole attention distribution
        CE_loss = torch.nn.CrossEntropyLoss()
        ce_loss = CE_loss(C1, label_train_y)
        re_loss = F.mse_loss(x_bar, data)
        loss_train=ce_loss+re_loss
        loss_train.requires_grad_(True)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        train_losses.append(loss_train.item())


        model.eval()
        with torch.no_grad():
             emb,att,x_bar,C1, C2, label_train_y, label_test_y = model(data, adj)

        # re_loss = F.mse_loss(x_bar, data)
        loss_test = CE_loss(C2, label_test_y)
        test_losses.append(loss_test.item())
        train_loss = np.average(train_losses)
        valid_loss = np.average(test_losses)
        if epoch % 5 == 0:
            print("epoch===========", epoch)
            print_msg = (f'train_loss: {train_loss:.5f} ' + f'test_loss: {valid_loss:.5f} ')
            print(print_msg)
        #Save embedding when running to the last epoch
    if epoch == 999:
        emb_all= np.array(emb.detach().numpy())    #change tensor  to array（572，65）
        DF=pd.DataFrame(data=emb_all)
        # print("finish train and save drug embeddings")
        # DF.to_csv('DMDDI_emb_{}.csv'.format(event_num))
        return emb_all


def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    for j in range(event_num):
        index = np.where(label_matrix == j)
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed)
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):
            index_all_class[index[0][test_index]] = k_num
            k_num += 1
    return index_all_class


def combine_and_predict(embedding_look_up, ddi_adj, clf_type, event_num, seed, CV):
    All_drug_embedding_pairs=[]
    for edge in ddi_adj:
        leftdrug=edge[0]
        rightdrug=edge[1]
        node_u_emb = embedding_look_up[leftdrug]
        node_v_emb = embedding_look_up[rightdrug]
        feature_vector = np.append(node_u_emb, node_v_emb)
        All_drug_embedding_pairs.append(feature_vector)
    feature_matrix=pd.DataFrame(data=All_drug_embedding_pairs)

    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)
    index_all_class = get_index(label, event_num, seed, CV)

    for k in range(CV):  # CV=5
        print("Number of cross-validations is：", k)
        train_index = np.where(index_all_class != k)
        test_index = np.where(index_all_class == k)
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)
        x_train = feature_matrix.iloc[train_index]
        x_train = x_train.values
        y_train = label[train_index]
        x_test = feature_matrix.iloc[test_index]
        x_test = x_test.values
        y_test = label[test_index]
        # =====================one-hot===============================
        y_train_one_hot = np.array(y_train)

        y_train_one_hot = (np.arange(y_train_one_hot.max() + 1) == y_train[:, None]).astype(
            dtype='float32')
        y_test_one_hot = np.array(y_test)
        y_test_one_hot = (np.arange(y_test_one_hot.max() + 1) == y_test[:, None]).astype(dtype='float32')
        #
        # if clf_type == 'RF':
        #     clf = RandomForestClassifier(n_estimators=100)
        # elif clf_type == 'GBDT':
        #     clf = GradientBoostingClassifier()
        # elif clf_type == 'KNN':
        #     clf = KNeighborsClassifier(n_neighbors=4)
        # elif clf_type == 'LR':
        #     clf = LogisticRegression()
        if clf_type == 'DDIMDL':
            dnn = DNN()  # 调用DNN模型去训练与预测
            # 这个早停策略就是patience=10，当10个出现不变时，就认为是收敛，就结束
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
            dnn.fit(x_train, y_train_one_hot, batch_size=128, epochs=100, validation_data=(x_test, y_test_one_hot),
                    callbacks=[early_stopping])  # 之前epoch=100
            pred += dnn.predict(x_test)  # 输出7451*65个float,但每折划分的数据大小不一定相同
            continue
        elif clf_type == 'RF':
            # clf = RandomForestClassifier(n_estimators=100) 原来的
            clf = RandomForestClassifier(n_estimators=100)
        elif clf_type == 'GBDT':
            clf = GradientBoostingClassifier()
        elif clf_type == 'SVM':
            clf = SVC(probability=True)
        elif clf_type == 'FM':
            clf = GradientBoostingClassifier()
        elif clf_type == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=4)
        else:
            clf = LogisticRegression()

        clf.fit(x_train, y_train)
        pred += clf.predict_proba(x_test)

    pred_score = pred
    pred_type = np.argmax(pred_score, axis=1)
    y_true = np.hstack((y_true, y_test))
    y_pred = np.hstack((y_pred, pred_type))
    y_score = np.row_stack((y_score, pred_score))
    result_all= evaluate(y_pred, y_score, y_true, event_num)
    return result_all

def save_result(feature_name, event_num, result):

    with open(feature_name + '_' + str(event_num) + '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='ddi')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--n_input', default=1716, type=int)
    parser.add_argument('--n_z', default=3, type=int)
    parser.add_argument('--pretrain_path', type=str, default='pkl')
    parser.add_argument('--clf', default='DDIMDL', choices=['LR','KNN','RF','GBDT','DDIMDL'],type=str)
    parser.add_argument('--adj_file', default='graph/ddi_class_65.csv', type=str)
    parser.add_argument('--event_num', default=65, type=int)
    args = parser.parse_args()
    device = torch.device('cpu')
    #When the test is 65, no change is needed, otherwise it needs to be modified in function load_data()
    dataset = load_data(args.name)

    #Different DDI data requires modification of the corresponding parameters
    if args.name == 'ddi':
        args.n_z=65
        args.pretrain_path = 'data/pre_ae_1716_2000_256_65.pkl'
        clf_type=args.clf
        adj_file = 'graph/ddi_class_65.csv'
        event_num = 65
        print("parameters setting:", args)

    print("the DM_DDI Running......")
    embbedings=train_dmddi(dataset)  #get the  drug embedding after end-end training



