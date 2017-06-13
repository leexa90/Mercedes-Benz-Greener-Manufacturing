
import pandas as pd
import numpy as np
import xgboost as xgb
 
train=pd.read_csv('train4b.csv')
test=pd.read_csv('test4b.csv')
train=train[train.y < 200]
train = train.T.drop_duplicates().T
predictors=[i for i in train.keys() if i not in ['y']]
test=test[predictors]

# -*- coding: utf-8 -*-
"""
@author: Faron
"""
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold

ID = 'ID'
TARGET = 'y'


TRAIN_FILE = "train4b.csv"
TEST_FILE = "test4b.csv"

SEED = 0
NFOLDS = 5
NTHREADS = 4

xgb_params = {
    'seed': 0,
    'colsample_bytree': 1,
    'silent': 0,
    'subsample': 1.0,
    'learning_rate': 1.0,
    'objective': 'reg:linear',
    'max_depth': 500,
    'num_parallel_tree': 1,
    'min_child_weight': -1,
    'nthread': NTHREADS,
    'nrounds': 1
}


def get_data():
    train = pd.read_csv(TRAIN_FILE)
    train = train[train.y < 200]
    test = pd.read_csv(TEST_FILE)
    print train.head()
    y_train = train[TARGET].ravel()

    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID, TARGET], axis=1, inplace=True)

    ntrain = train.shape[0]
    train_test = pd.concat((train, test)).reset_index(drop=True)



    x_train = np.array(train_test.iloc[:ntrain, :])
    x_test = np.array(train_test.iloc[ntrain:, :])

    return x_train, y_train, x_test


def get_oof(clf, x_train, y_train, x_test):
    ntrain = x_train.shape[0]
    oof_train = np.zeros((ntrain,))

    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED).split(x_train)

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)

    clf.train(x_train, y_train)
    oof_test = clf.predict(x_test)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def get_sparse_ohe(x_train, x_test, min_obs=10):
    ntrain = x_train.shape[0]

    train_test = np.concatenate((x_train, x_test)).reshape(-1, )

    # replace infrequent values by nan
    val = dict((k, np.nan if v < min_obs else k) for k, v in dict(Counter(train_test)).items())
    print dict((k, np.nan if v < 3 else k) for k, v in dict(Counter(train_test[0:500])).items())
    k, v = np.array(list(zip(*sorted(val.items()))))
    train_test = v[np.digitize(train_test, k, right=True)]

    ohe = csr_matrix(pd.get_dummies(train_test, dummy_na=False, sparse=True))

    x_train_ohe = ohe[:ntrain, :]
    x_test_ohe = ohe[ntrain:, :]

    return x_train_ohe, x_test_ohe


class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 1000)

    def train(self, x_train, y_train, x_valid=None, y_valid=None, sample_weights=None, verbose=1):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)
    
    # pred_leaf=True => getting leaf indices
    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x), pred_leaf=True).astype(int)


x_train, y_train, x_test = get_data()

dtrain = xgb.DMatrix(x_train, label=y_train)

xg = XgbWrapper(seed=SEED, params=xgb_params)
xg_cat_embedding_train, xg_cat_embedding_test = get_oof(xg, x_train, y_train, x_test)

xg_cat_embedding_ohe_train, xg_cat_embedding_ohe_test = get_sparse_ohe(xg_cat_embedding_train, xg_cat_embedding_test)

print("OneHotEncoded XG-Embeddings: {},{}".format(xg_cat_embedding_ohe_train.shape, xg_cat_embedding_ohe_test.shape))

die 
target='y'
plst =  {
    'seed': 0,
    'colsample_bytree': 1,
    'silent': 1,
    'subsample': 1.0,
    'learning_rate': 1.0,
    'objective': 'reg:linear',
    'max_depth': 300,
    'num_parallel_tree': 1,
    'min_child_weight': 50,
    'nthread': 6,
    'nrounds': 1
}

from sklearn.metrics import r2_score
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)+100
def r2_customised(preds, dtrain): # reduces rmse of ( individal_r2 - mean_r2)^2
    dtrain = dtrain.get_label()
    dtrain=dtrain/100
    preds=preds/100
    SS_res = ( dtrain - preds )**2
    SS_tot = ( dtrain - np.mean(dtrain))**2
    SumSS_res =  np.sum(( dtrain - preds )**2)
    SumSS_tot = np.sum(( dtrain - np.mean(dtrain))**2 )
    #r2 = 1 - (SumSS_res/(SumSS_tot+0.0001))
    preds =  1 - (SS_res/(SS_tot+0.0001))
    grad = (preds - max(preds))*0.001
    hess = grad*0+10
    return grad, hess

dictt={}
for depth in [3,4,5,6]:
    name='xMSE2'
    print depth
    dictt[depth]=[]
    rounds=0
    for num in range(0,10):
        train=train.sample(frac=1)
        train[name+str(depth)+'_'+str(num)]=0
        test[name+str(depth)+'_'+str(num)]=0
        max_n = 5
        if rounds ==0:
            for validation in range(0,max_n):
                dcv=train.iloc[validation::max_n]
                dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
                dtrain=train.iloc[dtrain_index]
                xgtrain = xgb.DMatrix(dtrain[predictors], label=dtrain[target])
                xgcv = xgb.DMatrix(dcv[predictors], label=dcv[target])
                xgtest = xgb.DMatrix(test[predictors])
                watchlist  = [ (xgtrain,'train'),(xgcv,'cv')]
                a={}
                model=xgb.train(plst,xgtrain,1,watchlist,early_stopping_rounds=500,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
                print a['train']['r2'][model.best_iteration],a['cv']['r2'][model.best_iteration],model.best_iteration
                rounds+=1.0/max_n
                print model.predict(xgcv, pred_leaf=True)[0],len(model.predict(xgcv, pred_leaf=True)[0])
                print rounds
        for validation in range(0,max_n):
            dcv=train.iloc[validation::max_n]
            dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
            dtrain=train.iloc[dtrain_index]
            xgtrain = xgb.DMatrix(dtrain[predictors], label=dtrain[target])
            xgcv = xgb.DMatrix(dcv[predictors], label=dcv[target])
            xgtest = xgb.DMatrix(test[predictors])
            watchlist  = [ (xgtrain,'train'),(xgcv,'cv')]
            a={}
            model=xgb.train(plst,xgtrain,1,watchlist,early_stopping_rounds=5000000,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
            train=train.set_value(dcv.index,name+str(depth)+'_'+str(num),model.predict(xgcv))
            test.set_value(test.index,name+str(depth)+'_'+str(num),test[name+str(depth)+'_'+str(num)]+model.predict(xgtest)/max_n)
            dictt[depth] += [a['cv']['r2'][model.best_iteration],]
        print r2_score(train.y,train[name+str(depth)+'_'+str(num)])
        print '\n'
 
#test['y']=map(lambda x:np.mean(x),np.array(test[['depth_4_0' ,  'depth_4_1',   'depth_4_2','depth_4_3','depth_4_4']]))
#test[['ID','y']].to_csv('first.csv',index=0)
             
predictors=[i for i in train.keys() if (name[0:5] in i) or i in ['ID','y']]
train[predictors].to_csv('train_xgbMSE.csv',index=0)
predictors=[i for i in test.keys() if (name[0:5] in i) or i in ['ID','y']]
test[predictors].to_csv('test_xgbMSE.csv',index=0)
'''
0.5734342 0.00993828205311 3
0.5816492 0.00993222393561 4
0.579851 0.00999689565915 5
0.5632972 0.0184149400565 6
0.5734342 0.00993828205311 7
'''
