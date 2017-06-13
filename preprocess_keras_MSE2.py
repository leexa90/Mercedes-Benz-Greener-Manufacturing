
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
for act_func in ['relu','tanh','sigmoid'][1:]:
    name='keras_MSE2_'+act_func
    rounds =0
    #train['y'] = np.log10(train['y'])
    train2,test2,predictors = model_XA('train4b.csv','test4b.csv',name,act_func=act_func,\
                                       loss='mean_squared_error',Y_transform=None)
    train2[predictors+['ID','y',]].to_csv('train_%s.csv' %name,index=0)    
    test2[predictors+['ID',]].to_csv('test_%s.csv'%name,index=0)
    del test2,train2
    import gc
    gc.collect()




