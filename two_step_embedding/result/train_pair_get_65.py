
from _csv import reader
import csv
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

event_num=30
seed = None
CV = 5

def DNN():

    droprate = 0.3
    # train_input = Input(shape=(vector_size * 2,), name='Inputlayer') #输入层 vector_size=572
    train_input = Input(shape=(60,), name='Inputlayer') #药物嵌入是65列时，为130，当为100维度时，设置200  attention sdcn is 128 ,the others is 100*2
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
    model = Model(train_input, out) #用交叉损失来训练，不像机器学习中多分类 nput=train_input, output=out
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_index(label_matrix, event_num, seed, CV):
    index_all_class = np.zeros(len(label_matrix))
    # event_num=65
    for j in range(event_num): #j从0-64
#（1）np.where()可以由label找到对应的index
#（2）对每个类别得到的索引进行训练与测试划分，有几折划分几次
#（3）把每折划分的测试集分别打上类别（0-4）
        index = np.where(label_matrix == j) #把类别为0，1...的索引都找出来，以类别为0的9000个索引为例
        kf = KFold(n_splits=CV, shuffle=True, random_state=seed) #对每个类别都进行5折划分
        k_num = 0
        for train_index, test_index in kf.split(range(len(index[0]))):#对9000个索引划分训练与测试集
            index_all_class[index[0][test_index]] = k_num #把测试集中边所在索引位置置为K_num，因为每次不会重复，所以5次后可把这类数据填满
            #即对65类药物对，第一类都划分为1-5个类别
            k_num += 1
    return index_all_class


def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        #返回AUC得分
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
            score[c] = binary_metric(y_true_c, y_score_c) #把每一类进行求精度得分，然后再求平均
        return np.average(score)
    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)



def evaluate(pred_type, pred_score, y_test, event_num):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    print("y真实的标签target:",y_test)
    print("y预测的标签:", pred_type)
    y_one_hot = label_binarize(y_test, np.arange(event_num))  #label_binarize()把一维变两维，对应置1，把真实的label变为one-hot
    result_all[0] = accuracy_score(y_test, pred_type) #精度，预测的类型与真实类型
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro') #自己编写函数，得到 ROC曲线得分，最后求一次平均
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro') #因为有5折，求5次平均
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')
    result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    result_all[5] = f1_score(y_test, pred_type, average='micro')
    result_all[6] = f1_score(y_test, pred_type, average='macro')
    result_all[7] = precision_score(y_test, pred_type, average='micro')
    result_all[8] = precision_score(y_test, pred_type, average='macro')
    result_all[9] = recall_score(y_test, pred_type, average='micro')
    result_all[10] = recall_score(y_test, pred_type, average='macro')
    return result_all
#first save at local,then move to corresbangding file
def save_result(feature_name, result_type, clf_type, result):
    with open(feature_name+ str(result_type) + '_' + clf_type+ '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0


#读入药物嵌入表示
############## need change the event_numb  ###########################
data=pd.read_csv('30/DMDDI_emb_30.csv', header=0, index_col=0)
# data1=pd.read_csv('./sdcn_attention_65.csv')
data=data.values #(572, 65)

#当65类数据时
# fddi=np.genfromtxt('ddi_neigh_after.csv', dtype=int, encoding='UTF-8')
# label=np.genfromtxt('../ddi_label.txt', dtype=int, encoding='UTF-8', usecols=0)
#当只是前10类数据

# ddi_adj=np.genfromtxt('./ddi_neigh_after.csv')
# label=np.genfromtxt('./ddi_label.csv')
ddi_label = pd.read_csv('30/ddi_advance30.csv', dtype=int, header=None)  # DataFrame (33264, 3)
# ddi_label1 = pd.read_csv('./ddi_advance65.csv', dtype=int) #(37263,2)
ddi_adj = ddi_label.iloc[:, [0, 1]].values  # array  (33214, 2)
label = ddi_label.iloc[:, 2].values  # (33214,)

def combine_and_predict(embedding_look_up, label_matrix, clf_type, event_num, seed, CV):
    X_train=[]
    for edge in ddi_adj:  # 遍历所有边
        leftdrug=edge[0]
        rightdrug=edge[1]
        node_u_emb = data[leftdrug]  # 经验证SDNE嵌入可以直接映射得到特征表示
        node_v_emb = data[rightdrug]
        # 对药物组合方式可以进行选择
        feature_vector = np.append(node_u_emb, node_v_emb)  # 把两个列表append,，转化为array，65维变成130维
        X_train.append(feature_vector)  # 把多个130维的药物对array加入到list中，再以data形式，变成DF
    feature_matrix=pd.DataFrame(data=X_train)
    print("output feature_matrix.shape",feature_matrix.shape)
    # feature_matrix.to_csv('emb_pairs.csv')
    # pd1['label']=label
    # # pd1.to_csv("drugpairs_hope_10.csv")
    # print("嵌入药物对与label保存完成！")
    # print("pd.shape",pd1.shape) #(33214, 131)

    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)  # np.zeros(11,1)生成11行，1列的0
    y_true = np.array([])  # 定义一个array数组
    y_pred = np.array([])
    y_score = np.zeros((0, event_num), dtype=float)  # 0行，65列，表只有列名
    # 可以理解这个index_all_class只不过是手工实现交叉验证数据划分，中间复杂过程可不理解
    index_all_class = get_index(label, event_num, seed, CV)  # 37264个类别的索引，形如【4，4，4，4，2，2，2，0，0，0...】

    count = 0
    for k in range(CV):  # CV=5
        print("交叉验证折数：", k)  # 使得每此交叉划分数据时，都是29790：7474
        train_index = np.where(index_all_class != k)  # 26567个索引 where()找到满足条件的索引，即类别不为0，1，2...的索引
        test_index = np.where(index_all_class == k)  # 7474
        pred = np.zeros((len(test_index[0]), event_num), dtype=float)  # 7474*65
        x_train = feature_matrix.iloc[train_index]  # (29790,3432)的DF，前有index索引    #由多个索引得到对应的嵌入
        x_train = x_train.values  # 变成规范的数组，但string
        y_train = label[train_index]
        # y_train=y_train.values
        x_test = feature_matrix.iloc[test_index]  # (6647,130) #由index在DF中获得嵌入
        x_test = x_test.values
        y_test = label[test_index]

        # =====================one-hot===============================
        y_train_one_hot = np.array(y_train)  # 29790个label
        a=y_train_one_hot.max()
        print("a max",a)
        y_train_one_hot = (np.arange(y_train_one_hot.max() + 1) == y_train[:, None]).astype(
            dtype='float32')  # (29790,65),
        y_test_one_hot = np.array(y_test)  # 7474
        y_test_one_hot = (np.arange(y_test_one_hot.max() + 1) == y_test[:, None]).astype(dtype='float32')
        if clf_type == 'DDIMDL':
            dnn = DNN()  # 调用DNN模型去训练与预测
            # 这个早停策略就是patience=10，当10个出现不变时，就认为是收敛，就结束
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')

            dnn.fit(x_train, y_train_one_hot, batch_size=128, epochs=100, validation_data=(x_test, y_test_one_hot),
                    callbacks=[early_stopping])  # 之前epoch=100
            pred += dnn.predict(x_test)  # 输出7451*65个float,但每折划分的数据大小不一定相同
            print("pred +=", pred)
            count = count + 1
            print("次数", count)
            continue
        elif clf_type == 'RF':
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
        # pu_estimator = PUAdapter(clf)  # 如果PU为1，则用PU_estimator来训练
        # pu_estimator.fit(x_train, y_train) #(29790,130)
        # pred += pu_estimator.predict(x_test) #(7474,130)
        # #
        # pred_score = pred / len(feature_matrix)
    pred_score = pred  # （7531,65）
    pred_type = np.argmax(pred_score, axis=1)  # 7431 # 得分最高那个即为类别
    y_true = np.hstack((y_true, y_test))  # y_test为测试所在的索引标签，series类型，前而y_true为空array
    y_pred = np.hstack((y_pred, pred_type))  # 前面的y_pred没有值，pre_type为array
    y_score = np.row_stack((y_score, pred_score))  # y_score=np.zeros((0, event_num)为空的，然后把后面的值赋它
    result_all= evaluate(y_pred, y_score, y_true, event_num)
    return result_all

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--classifier", choices=["RF", "KNN", "LR", 'DDIMDL', 'GBDT'], default=["DDIMDL"],
                        help="classifiers to use", nargs="+")  # GBDT,SVM can't be used
    args = vars(parser.parse_args())
    print("参数为：", args)
    clf = args['classifier']
    clf=str(clf[0])#取出列表中值，变成字符串，以便后面拼接
    featureName="+".join(['S','T','E']) #变成特征数组，再变成str
    featureName="DM-DDI_embedding"

    all_result=combine_and_predict(data,label,clf,event_num,seed,CV) #进行组合与预测
    save_result(featureName,event_num, clf, all_result)

#用法
#1. #119  导入SDCN学到的嵌入
#2. #128   导入 ddi_after
#3  #226   保存的文件名进行更改