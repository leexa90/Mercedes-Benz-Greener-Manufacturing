


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

# define custom R2 metrics for Keras backend
from keras import backend as K
from guppy import hpy



def model_XA(train_n='train4b.csv',test_n='test4b.csv',name='test',act_func='relu',\
             loss='mean_squared_error',seed=seed,Y_transform=None,Y_invtransform=None):
    temp = pd.read_csv(train_n,nrows=100)
    dtypes ={}
    for i in temp.keys():
        if temp[i].dtype == np.int64:
            if i =='y':
                dtypes[i]=temp[i].dtype
            if np.mean(temp[i]) < 1:
                dtypes[i]=np.int8
            else:
                dtypes[i]=np.int32
        elif temp[i].dtype == np.float64:
            dtypes[i]=np.float16
            #print i
        else:
            dtypes[i]=temp[i].dtype
            print i
    
        
    train = pd.read_csv(train_n,dtype  = dtypes,usecols=dtypes.keys())
    test = pd.read_csv(test_n,dtype  = dtypes,usecols=dtypes.keys())
    train = train.T.drop_duplicates().T
    train=train[train.y<200].reset_index(drop=1)
    predictors=[i for i in train.keys() if i not in ['y',]]
    test=test[predictors]
    test['y']=np.nan
    print('\nTrain shape: {}\nTest shape: {}'.format(train.shape, test.shape))
    if Y_transform != None:
        train['y']= Y_transform(train['y'])
        def r2_keras(y_true, y_pred):
            y_true = Y_invtransform(y_true)
            y_pred = Y_invtransform(y_pred)
            SS_res =  K.sum(K.square( y_true - y_pred )) 
            SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
            return (1-(SS_res/(SS_tot+0.0001)))*-1+100
##    if Y_transform == 'lg':
##        train['y']= np.log10(train['y'])
##    elif Y_transform == 'sqr':
##        train['y']= train['y']**.5
##    elif Y_transform == 'exp':
##        def log_105(x):
##            return np.log(x)/np.log(1.05)
##        train['y']= 1.05**train['y']
    else:
        def r2_keras(y_true, y_pred):
            SS_res =  K.sum(K.square( y_true - y_pred )) 
            SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
            return (1-(SS_res/(SS_tot+0.0001)))*-1+100


    def model():
        model = Sequential()
        #input layer
        model.add(Dense(input_dims, input_dim=input_dims))
        model.add(BatchNormalization())
        model.add(Activation(act_func))
        model.add(Dropout(0.2))
        # hidden layers
        model.add(Dense(input_dims))
        model.add(BatchNormalization())
        model.add(Activation(act_func))
        model.add(Dropout(0.2))
        
        model.add(Dense(input_dims//2))
        model.add(BatchNormalization())
        model.add(Activation(act_func))
        model.add(Dropout(0.2))
        
        model.add(Dense(input_dims//5))
        model.add(BatchNormalization())
        model.add(Activation(act_func))
        #model.add(Dropout(0.2))
        
        model.add(Dense(input_dims//10))
        model.add(BatchNormalization())
        model.add(Activation(act_func))
        #model.add(Dropout(0.3))
        
        model.add(Dense(input_dims//10, activation=act_func))
        
        # output layer (y_pred)
        model.add(Dense(1, activation='linear'))
        
        # compile this model
        model.compile(loss=loss, # one may use 'mean_absolute_error' as alternative
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
    import gc
    model_path = 'keras_model.h5'
    
    # prepare callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_r2_keras', 
            patience=20, # was 10
            verbose=1),
        
        ModelCheckpoint(
            model_path, 
            monitor='val_r2_keras', 
            save_best_only=True, 
            verbose=0)
    ]



    from sklearn.metrics import r2_score
    def xgb_r2_score(preds, dtrain):
        SS_res =  np.sum(( dtrain - preds )**2) 
        SS_tot = np.sum(( dtrain - np.mean(dtrain))**2 )
        return ( 1 - (SS_res/(SS_tot+0.0001)))
    rounds =[]
    for tries in range(seed,seed+20):
        estimator = KerasRegressor(
        build_fn=model, 
        nb_epoch=100, 
        batch_size=20,
        verbose=1
        )
        hp = hpy()
        max_n=3
        # X, y preparation
        #new_index=train['y'].apply(lambda x : int(x/5)*5).sort_values().index #stable sorted acording to fives
        #train=train.iloc[new_index].reset_index(drop=1)
        predictors = [x for x in train.keys() if name[0:5] not in x]
        X, y = train[predictors].drop(['ID','y'], axis=1).values, train.y.values
        print(X.shape)
        # X_test preparation
        X_test = test
        X_te = test[predictors].drop(['ID','y'], axis=1).values
        print(X_test.shape)
        if len(rounds)<9:
            while len(rounds) < 9:
                indexes=np.random.choice(range(0,len(train)),len(train),replace=False)

                for validation in range(0,max_n):
                    dtrain_index= [ i for i in indexes if i%max_n != validation]
                    dtest_index = [ i for i in indexes if i%max_n == validation]
                    X_tr, X_val = X[dtrain_index],X[dtest_index]
                    y_tr, y_val = y[dtrain_index],y[dtest_index]          
                    hist=estimator.fit(
                        X_tr, 
                        y_tr, 
                        epochs=120, # increase it to 20-100 to get better results
                        validation_data=(X_val, y_val),
                        verbose=2,
                        callbacks=callbacks,
                        shuffle=True
                    )
                    rounds += [(len(hist.history['loss'])-20),]
                    print rounds
                    K.clear_session()
        else:
            #print hp.heap()
            train[name+str(tries)]=0
            test[name+str(tries)]=0
            indexes=np.random.choice(range(0,len(train)),len(train),replace=False)
            for validation in range(0,max_n):
                dtrain_index= [ i for i in indexes if i%max_n != validation]
                dtest_index = [ i for i in indexes if i%max_n == validation]
                X_tr, X_val = X[dtrain_index],X[dtest_index]
                y_tr, y_val = y[dtrain_index],y[dtest_index]     
                estimator.fit(
                    X_tr, 
                    y_tr, 
                    epochs=int(np.mean(rounds)), # increase it to 20-100 to get better results
                    validation_data=(X_val, y_val),
                    verbose=0,
                    shuffle=True
                    )
                train=train.set_value(
                    dtest_index,
                    name+str(tries),
                    estimator.predict(X_val,verbose=0)
                    )
                test=test.set_value(
                    test.index,
                    name+str(tries),
                    test[name+str(tries)]+estimator.predict(X_te,verbose=0)/max_n
                    )
                del X_tr, X_val, y_tr, y_val
                K.clear_session()
            train[name+str(tries)]=train[name+str(tries)].astype(np.float32)
            print train[name+str(tries)].head()
            test[name+str(tries)]=train[name+str(tries)].astype(np.float32)
            gc.collect()
            print xgb_r2_score(train[name+str(tries)],train.y)
            #print hp.heap()
        train['ID']=train['ID'].astype(np.int32)
        test['ID']=test['ID'].astype(np.int32)
        predictors=[x for x in train.keys() if name[0:5] in x]
        #train[predictors+['ID','y',]].to_csv('train_%s.csv' %name,index=0)    
        #test[predictors+['ID',]].to_csv('test_%s.csv'%name,index=0)
##    if Y_transform == 'lg':
##        train['y']= 10**(train['y'])
##        train[predictors]=train[predictors].applymap(lambda x :10**x)
##    elif Y_transform == 'sqr':
##        train['y']= train['y']**2
##        train[predictors]=train[predictors].applymap(lambda x :x**2)
##    elif Y_transform == 'exp':
##        def log_105(x):
##            return np.log(x)/np.log(1.05)
##        train['y']= log_105(train['y'])
##        train[predictors]=train[predictors].applymap(log_105)
    if Y_transform != None:
        train['y'] = Y_invtransform(train['y'])
    return train[predictors+['ID','y']],test[predictors+['ID',]],predictors



