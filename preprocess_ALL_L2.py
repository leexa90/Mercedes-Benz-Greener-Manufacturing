
# coding: utf-8

# Hi, Kagglers!
# 
# Hereafter I will try to publish **some basic approaches to climb up the Leaderboard**
# 
# **Competition goal**
# 
# In this competition, Daimler is challenging Kagglers to tackle the curse of dimensionality and reduce the time that cars spend on the test bench.
# <br>Competitors will work with a dataset representing different permutations of Mercedes-Benz car features to predict the time it takes to pass testing. <br>Winning algorithms will contribute to speedier testing, resulting in lower carbon dioxide emissions without reducing Daimler’s standards. 
# 
# **The Notebook offers basic Keras skeleton to start with**
# 
# ### Stay tuned, this notebook will be updated on a regular basis
# **P.s. Upvotes and comments would let me update it faster and in a more smart way :)**

# 

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# preprocessing/decomposition
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA

# keras 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, ModelCheckpoint

# model evaluation
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# supportive models
from sklearn.ensemble import GradientBoostingRegressor
# feature selection (from supportive model)
from sklearn.feature_selection import SelectFromModel
from keras import backend as K
# to make results reproducible
seed = 42 # was 42


##train = pd.read_csv('train4b.csv')
##test = pd.read_csv('test4b.csv')
##train = train.T.drop_duplicates().T
##train=train[train.y<200]
##predictors=[i for i in train.keys() if i not in ['y',]]
##test=test[predictors]
##test['y']=np.nan
##print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))
##
##

from sklearn.metrics import r2_score
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return (1-(SS_res/(SS_tot+0.0001)))*-1+100
def xgb_r2_score(preds, dtrain):
    return r2_score(dtrain,preds)



from EXAMPLE import model_XA
import os

def lg(y):return np.log10(y)
def inv_lg(y):return 10**y

def sqr(y):return y**.5
def inv_sqr(y):return y**2

def exp(y):return 1.05**y
def inv_exp(y):return K.log(y)/K.log(1.05)

dictt = {(None,None):'None',
         (lg,inv_lg) : 'lg',
         (sqr,inv_sqr): 'sqr',
         (exp,inv_exp) : 'exp'}


'''
MODEL XA is my wrapper script for keras that takes in these inputs
model_XA(train_n='train4b.csv',test_n='test4b.csv',name='test',act_func='relu',\
             loss='mean_squared_error',seed=seed,Y_transform=None,Y_invtransform=None)

train_n = train data file
test_n = test data file
name = name you desire your dataset to have
act_func = activation functions for neural network , supported are : relu, sigmoid, tanh
loss = 'loss function you want', supported are : mean_squared error, mean_absolute_error , mean_percent_error https://keras.io/losses/#usage-of-loss-functions
seed = random seed (modified for my usage)
Y_transform = function to Y transform variable, default = None
Y_invtransform = function to inverse transform Y back to origin value, default = None

Function default monitors r2 correlation function.

Code trains a neural network and returns predicted values for L2 ensembling
1. Runs 3 X 3fold cross validation to get optimal number of training iterations
2. Runs 29 * 3fold training to output predicted values for Test and Train set

# USED 3 fold cross validation since R2 was was sensitive to sampling of data, thus
used a large test fold. 

Outputs 29 * 3 predicted values for both train and test set
'''
## MSE ###
for act_func in ['relu','tanh','sigmoid'][0:1]:
    for objective in [(None,None),(lg,inv_lg),(sqr,inv_sqr),(exp,inv_exp)][:-1]:
        name='L2_keras_predict_%s_%s'%(dictt[objective],act_func)
        if 'test_%s.csv'%name not in [x for x in os.listdir('.') ] and \
           'train_%s.csv'%name not in [x for x in os.listdir('.') ]:
            rounds =0
            train2,test2,predictors = model_XA('trainLvl2b.csv','testLvl2b.csv',name,act_func=act_func,\
                                               loss='mean_squared_error',Y_transform=objective[0],Y_invtransform=objective[1]\
                                               ,seed=0)
            train3,test3,predictors = model_XA('trainLvl2b.csv','testLvl2b.csv',name,act_func=act_func,\
                                               loss='mean_squared_error',Y_transform=objective[0],Y_invtransform=objective[1]\
                                               ,seed=20)
            train2 = train2.merge(train3,on=['ID','y'])
            test2 = test2.merge(test3,on='ID')
            predictors =[ x for x in train2.keys() if x[0:5] in name]
            train2[predictors+['ID','y',]].to_csv('train_%s.csv' %name,index=0)    
            test2[predictors+['ID',]].to_csv('test_%s.csv'%name,index=0)
            del test2,train2
            import gc
            gc.collect()
            quit()
        else:
            print name
die
### MAE ###
for act_func in ['relu','tanh','sigmoid'][:]:
    for objective in [(None,None),(lg,inv_lg),(sqr,inv_sqr),(exp,inv_exp)][:-1]:
        name='keras_MAE2_%s_%s'%(dictt[objective],act_func)
        if 'test_%s.csv'%name not in [x for x in os.listdir('.') ] and \
           'train_%s.csv'%name not in [x for x in os.listdir('.') ]:
            rounds =0
            train2,test2,predictors = model_XA('train4b.csv','test4b.csv',name,act_func=act_func,\
                                               loss='mean_squared_error',Y_transform=objective[0],Y_invtransform=objective[1]\
                                               ,seed=0)
            train3,test3,predictors = model_XA('train4b.csv','test4b.csv',name,act_func=act_func,\
                                               loss='mean_squared_error',Y_transform=objective[0],Y_invtransform=objective[1]\
                                               ,seed=20)
            train2 = train2.merge(train3,on=['ID','y'])
            test2 = test2.merge(test3,on='ID')
            predictors =[ x for x in train2.keys() if x[0:5] in name]
            train2[predictors+['ID','y',]].to_csv('train_%s.csv' %name,index=0)    
            test2[predictors+['ID',]].to_csv('test_%s.csv'%name,index=0)
            del test2,train2
            import gc
            gc.collect()
            quit()
        else:
            print name
### R2 ###
for act_func in ['relu','tanh','sigmoid'][:]:
    for objective in [(None,None),(lg,inv_lg),(sqr,inv_sqr),(exp,inv_exp)][:-1]:
        name='keras_MSE2_%s_%s'%(dictt[objective],act_func)
        if 'test_%s.csv'%name not in [x for x in os.listdir('.') ] and \
           'train_%s.csv'%name not in [x for x in os.listdir('.') ]:
            rounds =0
            train2,test2,predictors = model_XA('train4b.csv','test4b.csv',name,act_func=act_func,\
                                               loss='mean_squared_error',Y_transform=objective[0],Y_invtransform=objective[1]\
                                               ,seed=0)
            train3,test3,predictors = model_XA('train4b.csv','test4b.csv',name,act_func=act_func,\
                                               loss='mean_squared_error',Y_transform=objective[0],Y_invtransform=objective[1]\
                                               ,seed=20)
            train2 = train2.merge(train3,on=['ID','y'])
            test2 = test2.merge(test3,on='ID')
            predictors =[ x for x in train2.keys() if x[0:5] in name]
            train2[predictors+['ID','y',]].to_csv('train_%s.csv' %name,index=0)    
            test2[predictors+['ID',]].to_csv('test_%s.csv'%name,index=0)
            del test2,train2
            import gc
            gc.collect()
            quit()
        else:
            print name


