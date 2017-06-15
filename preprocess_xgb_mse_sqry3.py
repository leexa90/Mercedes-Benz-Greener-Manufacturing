import pandas as pd
import numpy as np
import xgboost as xgb

train=pd.read_csv('train4b.csv')
test=pd.read_csv('test4b.csv')
predictors=[i for i in train.keys() if i not in ['y']]
train=train[train.y < 200]
target='y'
params = {}

params["objective"] = "reg:linear"
params["eta"] = 0.01
params["min_child_weight"] = 1
params["subsample"] = 0.6
params["colsample_bytree"] = 0.6
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 6
#params["nthread"] = 6
params["nthread"] = 6
params['seed']=927
params['silent']=1
from sklearn.metrics import r2_score
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels**2, preds**2)+100
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
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess
def r2(preds,dtrain):
    return r2_score(dtrain,preds)

train['y']=(train['y'])**.5
params['eval_metric']='rmse'
plst = list(params.items())
dictt={}
for depth in [3,4,5,6]:
    for child in [5,20]:
        params["min_child_weight"] = child
        name = 'x_mse_sqrY'+str(child)
        print depth,child
        dictt[depth]=[]
        params["max_depth"] = depth
        plst = list(params.items())
        rounds=0
        for num in range(0,5):
            train=train.sample(frac=1)
            train[name+str(depth)+'_'+str(num)]=0
            test[name+str(depth)+'_'+str(num)]=0
            max_n = 10
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
                    model=xgb.train(plst,xgtrain,4000,watchlist,early_stopping_rounds=500,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
                    print a['train']['r2'][model.best_iteration],a['cv']['r2'][model.best_iteration],model.best_iteration
                    rounds+=model.best_iteration/max_n
            for validation in range(0,max_n):
                dcv=train.iloc[validation::max_n]
                dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
                dtrain=train.iloc[dtrain_index]
                xgtrain = xgb.DMatrix(dtrain[predictors], label=dtrain[target])
                xgcv = xgb.DMatrix(dcv[predictors], label=dcv[target])
                xgtest = xgb.DMatrix(test[predictors])
                watchlist  = [ (xgtrain,'train'),(xgcv,'cv')]
                a={}
                model=xgb.train(plst,xgtrain,int(rounds),watchlist,early_stopping_rounds=5000000,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
                train=train.set_value(dcv.index,name+str(depth)+'_'+str(num),model.predict(xgcv)**2)
                test.set_value(test.index,name+str(depth)+'_'+str(num),test[name+str(depth)+'_'+str(num)]+model.predict(xgtest)**2/max_n)
                dictt[depth] += [a['cv']['r2'][model.best_iteration],]
            print '\n'
            print r2_score(train.y**2,train[name+str(depth)+'_'+str(num)])
for i in dictt:
	print np.mean([float(x) for x in dictt[i]]),np.std([float(x) for x in dictt[i]])/len(dictt[depth])**.5

#test['y']=map(lambda x:np.mean(x),np.array(test[['depth_4_0' ,  'depth_4_1',   'depth_4_2','depth_4_3','depth_4_4']]))
#test[['ID','y']].to_csv('first.csv',index=0)
train['y']=10**(train['y'])         
predictors=[i for i in train.keys() if (name[0:5] in i) or i in ['ID','y']]
train[predictors].to_csv('train_xgmse_lgy.csv',index=0)
predictors=[i for i in test.keys() if (name[0:5] in i) or i in ['ID','y']]
test[predictors].to_csv('test_xgmse_lgy.csv',index=0)
'''
0.5734342 0.00993828205311 3
0.5816492 0.00993222393561 4
0.579851 0.00999689565915 5
0.5632972 0.0184149400565 6
0.5734342 0.00993828205311 7
'''
