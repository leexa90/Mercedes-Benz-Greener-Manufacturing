
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


# 

# In[2]:

# Read datasets
train = pd.read_csv('train4b.csv')
test = pd.read_csv('test4b.csv')
train = train.T.drop_duplicates().T
train=train[train.y<200]
predictors=[i for i in train.keys() if i not in ['y',]]
test=test[predictors]
# save IDs for submission
id_test = test['ID'].copy()

# glue datasets together
total = pd.concat([train, test], axis=0)
print('initial shape: {}'.format(total.shape))

# binary indexes for train/test set split
is_train = ~total.y.isnull()

# find all categorical features, uncomment if cat var exist
##cf = total.select_dtypes(include=['object']).columns
##print cf
### make one-hot-encoding convenient way - pandas.get_dummies(df) function
##dummies = pd.get_dummies(
##    total[cf],
##    drop_first=False # you can set it = True to ommit multicollinearity (crucial for linear models)
##)
##
##print('oh-encoded shape: {}'.format(dummies.shape))
##
##### get rid of old columns and append them encoded
####total = pd.concat( 
####    [
####        total.drop(cf, axis=1), # drop old
####        dummies # append them one-hot-encoded
####    ],
####    axis=1 # column-wise
####)
##
##print('appended-encoded shape: {}'.format(total.shape))

# recreate train/test again, now with dropped ID column
train, test = total[is_train], total[~is_train]

# drop redundant objects
del total

# check shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))


# 

# In[3]:

# Calculate additional features: dimensionality reduction components
n_comp=10 # was 10

# uncomment to scale data before applying decompositions
# however, all features are binary (in [0,1] interval), i don't know if it's worth something
train_scaled = train.drop('y', axis=1).copy()
test_scaled = test.copy()
'''
ss = StandardScaler()
ss.fit(train.drop('y', axis=1))

train_scaled = ss.transform(train.drop('y', axis=1).copy())
test_scaled = ss.transform(test.copy())
'''

### PCA
##pca = PCA(n_components=n_comp, random_state=seed)
##pca2_results_train = pca.fit_transform(train_scaled)
##pca2_results_test = pca.transform(test_scaled)
##
### ICA
##ica = FastICA(n_components=n_comp, random_state=seed,max_iter=500)
##ica2_results_train = ica.fit_transform(train_scaled)
##ica2_results_test = ica.transform(test_scaled)
##
### Append it to dataframes
##for i in range(1, n_comp+1):
##    train['pca_' + str(i)] = pca2_results_train[:,i-1]
##    test['pca_' + str(i)] = pca2_results_test[:, i-1]
##    
##    train['ica_' + str(i)] = ica2_results_train[:,i-1]
##    test['ica_' + str(i)] = ica2_results_test[:, i-1]
   


# 

### In[4]:
##
### create augmentation by feature importances as additional features
##t = train['y']
##tr = train.drop(['y'], axis=1)
##
### Tree-based estimators can be used to compute feature importances
##clf = GradientBoostingRegressor(
##                max_depth=4, 
##                learning_rate=0.005, 
##                random_state=seed, 
##                subsample=0.95, 
##                n_estimators=200
##)
##
### fit regressor
##clf.fit(tr, t)
##
### df to hold feature importances
##features = pd.DataFrame()
##features['feature'] = tr.columns
##features['importance'] = clf.feature_importances_
##features.sort_values(by=['importance'], ascending=True, inplace=True)
##features.set_index('feature', inplace=True)
##
### select best features
##model = SelectFromModel(clf, prefit=True)
##train_reduced = model.transform(tr)
##
##
##test_reduced = model.transform(test.copy())
##die
### dataset augmentation
##train = pd.concat([train, pd.DataFrame(train_reduced)], axis=1)
##test = pd.concat([test, pd.DataFrame(test_reduced)], axis=1)
##
# check new shape
print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))


# 

# In[5]:

# define custom R2 metrics for Keras backend
from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return (1-(SS_res/(SS_tot+0.0001)))*-1+100

def r2_keras_array(y_true, y_pred):
    SS_res =  np.sum(( y_true - y_pred )**2) 
    SS_tot = np.sum(( y_true - np.mean(y_true))**2 )
    print SS_res,SS_tot
    return ( 1 - (SS_res/(SS_tot+0.0001)))
# 

# In[6]:

# base model architecture definition
def model():
    model = Sequential()
    #input layer
    model.add(Dense(input_dims, input_dim=input_dims))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    # hidden layers
    model.add(Dense(input_dims))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    #model.add(Dropout(0.2))
    
    model.add(Dense(input_dims//2))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.2))
    
    model.add(Dense(input_dims//4))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.2))

    model.add(Dense(input_dims//10))
    model.add(BatchNormalization())
    model.add(Activation(act_func))
    model.add(Dropout(0.2))
    
    model.add(Dense(input_dims//10, activation=act_func))
    
    # output layer (y_pred)
    model.add(Dense(1, activation='linear'))
    
    # compile this model
    model.compile(loss='mean_squared_error', # one may use 'mean_absolute_error' as alternative
                  optimizer='adam',
                  metrics=[r2_keras] # you can add several if needed
                 )
    
    # Visualize NN architecture
    #print(model.summary())
    return model


# In[7]:

# initialize input dimension
input_dims = train.shape[1]-2 #ID and y

# make np.seed fixed
np.random.seed(seed)

# initialize estimator, wrap model in KerasRegressor
estimator = KerasRegressor(
    build_fn=model, 
    nb_epoch=100, 
    batch_size=20,
    verbose=1
)


# In[8]:

# X, y preparation
X, y = train.drop('y', axis=1).values, train.y.values
print(X.shape)

# X_test preparation
X_test = test
X_te = test.values
print(X_test.shape)

# train/validation split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=seed
)


# In[9]:

# define path to save model
import os
import h5py as h5py
model_path = 'keras_model.h5'

# prepare callbacks
callbacks = [
    EarlyStopping(
        monitor='val_r2_keras', 
        patience=100, # was 10
        verbose=1),
    
    ModelCheckpoint(
        model_path, 
        monitor='val_r2_keras', 
        save_best_only=True, 
        verbose=0)
]

# fit estimator
# train/validation split


from sklearn.metrics import r2_score
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true - y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return (1-(SS_res/(SS_tot+0.0001)))*-1+100
def xgb_r2_score(preds, dtrain):
    return r2_score(dtrain,preds)

#activation functions for hidden layers
act_func = 'tanh' # could be 'relu', 'sigmoid', ...
for i in ['relu','tanh','sigmoid']:
    act_func = i # could be 'relu', 'sigmoid', ...
    name='keras_MSE2_lgY_'+act_func
    rounds =0
    for tries in range(0,30):
        max_n=3
        # X, y preparation
        train=train.sample(frac=1).reset_index(drop=1)
        #new_index=train['y'].apply(lambda x : int(x/5)*5).sort_values().index #stable sorted acording to fives
        #train=train.iloc[new_index].reset_index(drop=1)
        predictors = [x for x in train.keys() if name[0:5] not in x]
        X, y = train[predictors].drop(['ID','y'], axis=1).values, train.y.values
        print(X.shape)
        # X_test preparation
        X_test = test
        X_te = test[predictors].drop(['ID','y'], axis=1).values
        print(X_test.shape)
        pred_train=train[['ID']].copy()
        pred_test=test[['ID']].copy()
        if rounds ==0 or tries ==1:
            for validation in range(0,max_n):
                dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
                X_tr, X_val = X[dtrain_index],X[validation::max_n]
                y_tr, y_val = y[dtrain_index],y[validation::max_n]          
                hist=estimator.fit(
                    X_tr, 
                    y_tr, 
                    epochs=80, # increase it to 20-100 to get better results
                    validation_data=(X_val, y_val),
                    verbose=2,
                    callbacks=callbacks,
                    shuffle=True
                )
                rounds += 0.5*(len(hist.history['loss'])-10)/max_n
                print rounds
        elif tries >=1:
            pred_train[name+str(tries)]=0
            pred_test[name+str(tries)]=0
            for validation in range(0,max_n):
                dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
                X_tr, X_val = X[dtrain_index],X[validation::max_n]
                y_tr, y_val = y[dtrain_index],y[validation::max_n]          
                hist=estimator.fit(
                    X_tr, 
                    y_tr, 
                    epochs=int(rounds), # increase it to 20-100 to get better results
                    validation_data=(X_val, y_val),
                    verbose=2,
                    shuffle=True
                )
                rounds = len(hist.history['loss'])-10
                temp=pred_train.set_value(
                    range(validation,len(train)
                          ,max_n),
                    name+str(tries),
                    estimator.predict(X_val,verbose=0)
                    )
                temp=pred_test.set_value(
                    test.index,
                    name+str(tries),
                    pred_test[name+str(tries)]+estimator.predict(X_te,verbose=0)/max_n
                    )
        train=train.merge(pred_train,on='ID')
        test=test.merge(pred_test,on='ID')
        print xgb_r2_score(train[name+str(tries)],train.y)
    train['ID']=train['ID'].astype(np.int32)
    test['ID']=test['ID'].astype(np.int32)
    predictors=[x for x in train.keys() if name[0:5] in x]
    train[predictors+['ID','y',]].to_csv('train_%s.csv' %name,index=0)    
    test[predictors+['ID',]].to_csv('test_%s.csv'%name,index=0)
die

# 

# In[10]:

# if best iteration's model was saved then load and use it
if os.path.isfile(model_path):
    estimator = load_model(model_path, custom_objects={'r2_keras': r2_keras})

# check performance on train set
print('MSE train: {}'.format(mean_squared_error(y_tr, estimator.predict(X_tr))**0.5)) # mse train
print('R^2 train: {}'.format(r2_score(y_tr, estimator.predict(X_tr)))) # R^2 train

# check performance on validation set
print('MSE val: {}'.format(mean_squared_error(y_val, estimator.predict(X_val))**0.5)) # mse val
print('R^2 val: {}'.format(r2_score(y_val, estimator.predict(X_val)))) # R^2 val
pass


# 

# In[11]:

# predict results
res = estimator.predict(X_test.values).ravel()

# create df and convert it to csv
output = pd.DataFrame({'id': id_test, 'y': res})
output.to_csv('keras-baseline.csv', index=False)


# In[12]:



