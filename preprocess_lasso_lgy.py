import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
def xgb_r2_score(preds, dtrain):
    labels = dtrain
    return 'r2', r2_score(labels, preds)

train=pd.read_csv('train4b.csv')
test=pd.read_csv('test4b.csv')
train=train[train.y < 200]
train['y']=np.log10(train['y'])
train = train.T.drop_duplicates().T
predictors=[i for i in train.keys() if i not in ['y']]
test=test[predictors]
from sklearn import linear_model
max_n = 10
dictt={}
name='lasso_lgy'
for alpha in [0.01,0.05,0.1,0.3,0.5,0.7,1,1.5,2,2.5,3,4,5,7,10,12.5,15,17.5,20,25,30,35,40,45,50,60,70,80,90,100,110][0:5]:
    for repeat in range(0,10):
        train[name+'_'+str(alpha)+'_'+str(repeat)]=0
        test[name+'_'+str(alpha)+'_'+str(repeat)]=0
        train=train.sample(frac=1).reset_index(drop=1)
        for validation in range(0,max_n):
            dcv=train.iloc[validation::max_n]
            dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
            dtrain=train.iloc[dtrain_index]
            dtest=test
            iter_n = 2000
            if alpha < 0.1:
                iter_n=4000
            reg = linear_model.Lasso (alpha = alpha,max_iter=iter_n)
            reg.fit(np.array(dtrain[predictors]),dtrain.y.values)
            #print xgb_r2_score(reg.predict(np.array(dcv[predictors])),dcv.y.values)
            train=train.set_value(
                dcv.index,
                name+'_'+str(alpha)+'_'+str(repeat),
                reg.predict(np.array(dcv[predictors]))
                )
            test.set_value(
                test.index,
                name+'_'+str(alpha)+'_'+str(repeat),
                test[name+'_'+str(alpha)+'_'+str(repeat)]+reg.predict(np.array(dtest[predictors]))/max_n)
        #print str(alpha),str(repeat),xgb_r2_score(dtrain[name+'_'+str(alpha)+'_'+str(repeat)],dtrain.y.values)
        if name+'_'+str(alpha) not in dictt:
            dictt[alpha]=[xgb_r2_score(dtrain[name+'_'+str(alpha)+'_'+str(repeat)],dtrain.y.values)[1],]
        else:
            dictt[alpha]+=[xgb_r2_score(dtrain[name+'_'+str(alpha)+'_'+str(repeat)],dtrain.y.values)[1],]
        print alpha,xgb_r2_score(10**train[name+'_'+str(alpha)+'_'+str(repeat)],10**train.y.values)
    dictt[alpha]=np.mean(dictt[alpha])
    print '\n'
train['y']=10**(train['y'])
pred = [x for x in train.keys() if 'lasso' in x]
train['ID']=train['ID'].astype(np.int32)
test['ID']=test['ID'].astype(np.int32)
train[pred+['ID','y']].to_csv('train_lassolgy.csv',index=0)
test[pred+['ID']].to_csv('test_lassolgy.csv',index=0)
'''
0.5734342 0.00993828205311 3
0.5816492 0.00993222393561 4
0.579851 0.00999689565915 5
0.5632972 0.0184149400565 6
0.5734342 0.00993828205311 7
'''
