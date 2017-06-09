import pandas as pd
import numpy as np
import xgboost as xgb
train=pd.read_csv('trainLvl2.csv')
test=pd.read_csv('testLvl2.csv')
train = train.T.drop_duplicates().T
predictors=[i for i in train.keys() if i not in ['y']]
test=test[predictors]

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
    return 'r2', r2_score(labels, preds)+100
def xgb_r2_score2(preds, dtrain):
    #labels = dtrain.get_label()
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
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess
params['eval_metric']='rmse'
plst = list(params.items())
dictt={}
for depth in [3,4,5,6]:
    print depth
    dictt[depth]=[]
    params["max_depth"] = depth
    plst = list(params.items())
    for num in range(0,5):
        train=train.sample(frac=1)
        train['depth_AllCA_'+str(depth)+'_'+str(num)]=0
        test['depth_AllCA_'+str(depth)+'_'+str(num)]=0
        max_n = 10
        for validation in range(0,max_n):
            dcv=train.iloc[validation::max_n]
            dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
            dtrain=train.iloc[dtrain_index]
            xgtrain = xgb.DMatrix(dtrain[predictors], label=dtrain[target])
            xgcv = xgb.DMatrix(dcv[predictors], label=dcv[target])
            xgtest = xgb.DMatrix(test[predictors])
            watchlist  = [ (xgtrain,'train'),(xgcv,'cv')]
            a={}
            model=xgb.train(plst,xgtrain,700,watchlist,early_stopping_rounds=5000,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
            print a['train']['r2'][model.best_iteration],a['cv']['r2'][model.best_iteration],model.best_iteration
            model=xgb.train(plst,xgtrain,model.best_iteration,watchlist,early_stopping_rounds=500,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
            train=train.set_value(dcv.index,'depth_AllCA_'+str(depth)+'_'+str(num),model.predict(xgcv))
            test.set_value(test.index,'depth_AllCA_'+str(depth)+'_'+str(num),test['depth_AllCA_'+str(depth)+'_'+str(num)]+model.predict(xgtest)/max_n)
            dictt[depth] += [a['cv']['r2'][model.best_iteration],]
        print '\n'
    print np.mean([float(x) for x in dictt[depth]]),np.std([float(x) for x in dictt[depth]])/len(dictt[depth])**.5
for i in dictt:
	print np.mean([float(x) for x in dictt[i]]),np.std([float(x) for x in dictt[i]])/len(dictt[depth])**.5

#test['y']=map(lambda x:np.mean(x),np.array(test[['depth_4_0' ,  'depth_4_1',   'depth_4_2','depth_4_3','depth_4_4']]))
#test[['ID','y']].to_csv('first.csv',index=0)
            
predictors=[i for i in train.keys() if ('depth' in i) or i in ['ID','y']]
train[predictors].to_csv('train_AllCA.csv',index=0)
predictors=[i for i in test.keys() if ('depth' in i) or i in ['ID','y']]
test[predictors].to_csv('test_AllCA.csv',index=0)
'''
0.5734342 0.00993828205311 3
0.5816492 0.00993222393561 4
0.579851 0.00999689565915 5
0.5632972 0.0184149400565 6
0.5734342 0.00993828205311 7
'''
