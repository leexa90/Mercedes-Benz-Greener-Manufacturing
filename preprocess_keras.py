import numpy as np
np.random.seed(123)

import pandas as pd
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from sklearn.metrics import r2_score
'''
PART 1
'''

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
##
##
'''
convert string to interger
'''
for i in train.keys():
    if train[i].dtype=='O':
        print i
        #test first
        temp=train.groupby(i).y.mean().to_dict()
        temp2={}
        counter=0
        for j in sorted([ (temp[x],x)  for x  in temp]):
            temp2[j[1]]=counter
            counter +=1
        temp=train.groupby(i).y.mean().to_dict()
        #test[i+'_2']=test[i].map(temp)        
        test[i]=test[i].map(temp2)
        train[i]=train[i].map(temp2)

train.to_csv('train3.csv',index=0)
test.to_csv('test3.csv',index=0)
'''
Part2
'''
train=pd.read_csv('train3.csv')
test=pd.read_csv('test3.csv')
from scipy import stats

for i in train.keys():
    if len(train[i].unique())==1:
        print i        
        del train[i]
        del test[i]
    else:
        try:
            if np.mean(test[i].isnull()) !=0.0:
                missing= test[test[i].isnull()][i].index
                print test[test[i].isnull()]
                # fill missing with mode
                test.set_value(missing,i,stats.mode(train[i])[0][0])
                print test.iloc[missing]
            
        except KeyError:
            print i
train=train.fillna(0)
test=test.fillna(0)
## Batch generators ##################################################################################################################################

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################

index = list(train.index)
print index[0:10]
np.random.shuffle(index)
print index[0:10]
train = train.iloc[index]
## set test y to NaN
test['y'] = np.nan
y = train['y']
id_train = train['ID'].values
id_test = test['ID'].values
ntrain = train.shape[0]
tr_te = pd.concat((train, test), axis = 0)
f_cat = [f for f in tr_te.columns if  f in \
         ['X0','X1','X2','X3','X4','X5','X6','X7','X8']]
sparse_data = []
# cat data
for f in f_cat:
    dummy = pd.get_dummies(tr_te[f].astype('category'))
    tmp = csr_matrix(dummy)
    sparse_data.append(tmp)
# continous data
f_num = [f for f in tr_te.columns if f not  in \
         ['X0','X1','X2','X3','X4','X5','X6','X7','X8','y','ID']]
scaler = StandardScaler()
tmp = csr_matrix(tr_te[f_num])
sparse_data.append(tmp)
#del(tr_te, train, test)

## sparse train and test data
xtr_te = hstack(sparse_data, format = 'csr')
xtr_te = np.array(tr_te[f_cat+f_num])
xtrain = xtr_te[:ntrain, :]
xtest = xtr_te[ntrain:, :]

print('Dim train', xtrain.shape)
print('Dim test', xtest.shape)

#del(xtr_te, sparse_data, tmp)
import gc
gc.collect()

def nn_model():
    model = Sequential()
    model.add(Dense(600, input_dim=xtrain.shape[1], kernel_initializer='TruncatedNormal', activation='relu'))
    model.add(Dense(300, kernel_initializer='TruncatedNormal', activation='relu'))
    model.add(Dense(150, kernel_initializer='TruncatedNormal', activation='relu'))
    model.add(Dense(50, kernel_initializer='TruncatedNormal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    from keras import optimizers
    sgd = optimizers.SGD(lr=0.000005, decay=0, momentum=0, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
##    model.add(Dense(55, input_dim = xtrain.shape[1], kernel_initializer = 'he_normal'))
##    model.add(PReLU())
##    model.add(BatchNormalization())
##    model.add(Dropout(0.3))
##        
####    model.add(Dense(100, init = 'he_normal'))
####    model.add(PReLU())
####    model.add(BatchNormalization())    
####    model.add(Dropout(0.2))
##    
##    model.add(Dense(50, kernel_initializer = 'he_normal'))
##    model.add(PReLU())
##    model.add(BatchNormalization())    
##    model.add(Dropout(0.2))
##    from keras import losses
##    model.add(Dense(1, kernel_initializer = 'he_normal'))
##    model.compile(loss=losses.mean_absolute_error, optimizer = 'adadelta')
##    return(model)

## cv-folds
nfolds = 10
folds = KFold(len(y), n_folds = nfolds, shuffle = True, random_state = 111)

## train models
i = 0
nbags = 1
nepochs = 50
pred_oob = np.zeros(xtrain.shape[0])
pred_test = np.zeros(xtest.shape[0])

def samples(x_source, y_source, size):
    while True:
        for i in range(0, x_source.shape[0], size):
            j = i + size
            
            if j > x_source.shape[0]:
                j = x_source.shape[0]
                
            yield (x_source[i:j], y_source[i:j])
#model.fit_generator(samples(x_train, x_train, nb_batch),x_train.shape[0], nb_epoch, verbose=1)
for (inTr, inTe) in folds:
    xtr = xtrain[inTr]
    ytr = y[inTr]
    xte = xtrain[inTe]
    yte = y[inTe]
    pred = np.zeros(xte.shape[0])
    for j in range(nbags):
        model = nn_model()
        #model = model.fit(xtr, ytr)
        fit = model.fit_generator(samples(xtr, ytr, 128), \
                                  steps_per_epoch=xtr.shape[0]/128,epochs=400, \
                                  validation_data=samples(xte, yte, 128),\
                                  validation_steps=xte.shape[0]/128,verbose=2)
        pred += model.predict_generator(generator = batch_generatorp(xte, 800, False), val_samples = xte.shape[0])[:,0]
        pred_test += model.predict_generator(generator = batch_generatorp(xtest, 800, False), val_samples = xtest.shape[0])[:,0]
    pred /= nbags
    pred_oob[inTe] = pred
    score = mean_absolute_error(yte, pred)
    i += 1
    print('Fold ', i, '- MAE:', score)

print('Total - MAE:', mean_squared_error(y, pred_oob))
