import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
l2_models = [x for x in os.listdir('.') if 'L2' in x and x.endswith('csv') and x.startswith('train')]
l2_models = sorted(l2_models)

train  = pd.read_csv(l2_models[-1])
test = pd.read_csv('test'+l2_models[-1][5:])
train['ID'] = train['ID'].astype(np.int32)
test['ID'] = test['ID'].astype(np.int32)


for i in l2_models[:-1]:
    temp = pd.read_csv(i)
    train = train.merge(temp,on = ['ID','y'])
    temp = pd.read_csv('test'+i[5:])
    test = test.merge(temp,on = ['ID'])
    print test.head()

def get_name(str):
    if len(str) >= 2:
        if str[-2] in ['1','2','3','4','5','6','7','8','9','0']:
            return str[0:-2]
        elif str[-1] in ['1','2','3','4','5','6','7','8','9','0']:
            return str[0:-1]
        else: None

['L2_keras_predict_sqr_tanh', 'L2_keras_predict_None_tanh', \
 'L2_keras_predict_lg_tanh']


new  = []
for i in list(set(map(get_name,train.keys()))):
    if i != None:
        predictors2 =[ x for x in train.keys() if i in x]
        test[i+'_ALL']=map(lambda x: np.mean(x), \
                                             np.array(test[predictors2]))
        train[i+'_ALL']=map(lambda x: np.mean(x), \
                                             np.array(train[predictors2]))
        print i,r2_score(train.y,train[i+'_ALL'])
        new += [i+'_ALL',]
predictors1 =[ x for x in train.keys() if 'L2_keras_predict_sqr_tanh' in x]
train['L2_keras_predict_sqr_tanh_ALL'] = map(lambda x: np.mean(x)**2, \
                                             np.array(train[predictors1]**.5))

predictors2 =[ x for x in train.keys() if 'L2_keras_predict_None_tanh' in x]
train['L2_keras_predict_None_tanh_ALL'] = map(lambda x: np.mean(x), \
                                             np.array(train[predictors2]))

predictors3 =[ x for x in train.keys() if 'L2_keras_predict_lg_tanh' in x]
train['L2_keras_predict_lg_tanh_ALL'] = map(lambda x: 10**np.mean(x), \
                                             np.array(np.log10(train[predictors3])))

print r2_score(train.y,train['L2_keras_predict_None_tanh_ALL'])
print r2_score(train.y,train['L2_keras_predict_sqr_tanh_ALL'])
print r2_score(train.y,train['L2_keras_predict_lg_tanh_ALL'])
