
import pandas as pd
import numpy as np
import xgboost as xgb

def get_mean_for_cat_var(file_n,fold1,fold2): #gets median for cat data
    data=pd.read_csv(file_n)
    data=data[data.y < 200]
    train=data.iloc[fold1]
    cv =data.iloc[fold2]
    test = pd.read_csv('test.csv')
    pred =['ID',]
    for i in ['X0', 'X2']:#train.keys():
        if train[i].dtype=='O' and len(train[i].unique()) > 5:
            print i
            pred += [i+'y',]
            for value in data[i].unique():
                temp1=train[train[i]==value]
                temp2=cv[cv[i]==value]
                temp_Test=test[test[i]==value]
                List1=list(train[train[i]==value].index)
                List2=list(cv[cv[i]==value].index)
                median1=np.median(temp1['y'])
                median2=np.median(temp2['y'])
                if len(List1)+len(List2) >40:
                    train=train.set_value(temp1.index,i+'y',median2)
                    cv=cv.set_value(temp2.index,i+'y',median1)
                    test=test.set_value(temp_Test.index,i+'y',0.5*median1+0.5*median2)
                    # bayesian modelling, giving value of y given particular category of X0-X8
            train.set_value(train[i+'y'].isnull(),i+'y',np.mean(train[i+'y'])) #or median
            cv.set_value(cv[i+'y'].isnull(),i+'y',np.mean(train[i+'y'])) #or median
            test.set_value(test[i+'y'].isnull(),i+'y',np.mean(train[i+'y'])) #or median
    return pd.concat([train[pred+['y']],cv[pred+['y']]]),test[pred]
 
train=pd.read_csv('train4c.csv')
#train=train.merge(pd.read_csv('train2b.csv'),on='ID')
test=pd.read_csv('test4c.csv')
#test=test.merge(pd.read_csv('test2b.csv'),on='ID')
train=train[train.y < 200]
train = train.T.drop_duplicates().T
predictors=[i for i in train.keys() if i not in ['y']]
test=test[predictors]
 
target='y'
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.003
params["min_child_weight"] = 15
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
predictors += ['X0y', 'X2y']#,'X1y', 'X3y', 'X5y', 'X6y', 'X8y']
predictors = [x for x in predictors if x != 'ID']
for depth in [2,4,7,10]:
    for child in [5,20]:
        params["min_child_weight"] = child
        name='xMSE4'+str(child)+'_'
        print depth
        dictt[depth]=[]
        params["max_depth"] = depth
        plst = list(params.items())
        rounds=0
        for num in range(0,30):
            if num%10==0:
                rounds = 0
            train=train.sample(frac=1)
            train[name+str(depth)+'_'+str(num)]=0
            test[name+str(depth)+'_'+str(num)]=0
            max_n = 2
            if rounds ==0:
                for validation in range(0,2):#has to be 2
                    dcv_index = [ i for i in range(len(train)) if i%max_n == validation]
                    dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
                    dcv=train.iloc[dcv_index]
                    dtrain=train.iloc[dtrain_index]
                    train_temp,test_temp = get_mean_for_cat_var('train.csv',dtrain_index,dcv_index)
                    dtrain = pd.merge(dtrain , train_temp , on =['ID','y'])
                    dcv = pd.merge(dcv , train_temp , on =['ID','y'])
                    dtest = pd.merge(test , test_temp , on =['ID'])
                    
                    xgtrain = xgb.DMatrix(dtrain[predictors], label=dtrain[target])
                    xgcv = xgb.DMatrix(dcv[predictors], label=dcv[target])
                    xgtest = xgb.DMatrix(dtest[predictors])
                    watchlist  = [ (xgtrain,'train'),(xgcv,'cv')]
                    a={}
                    model=xgb.train(plst,xgtrain,8000,watchlist,early_stopping_rounds=500,\
                                    evals_result=a, maximize=1,verbose_eval=1000,feval=xgb_r2_score)
                    print a['train']['r2'][model.best_iteration],a['cv']['r2'][model.best_iteration],model.best_iteration
                    rounds+=model.best_iteration/max_n
                    print rounds
            for validation in range(0,2):
                dcv_index = [ i for i in range(len(train)) if i%max_n == validation]
                dtrain_index= [ i for i in range(len(train)) if i%max_n != validation]
                dcv=train.iloc[dcv_index]
                dtrain=train.iloc[dtrain_index]
                train_temp,test_temp = get_mean_for_cat_var('train.csv',dtrain_index,dcv_index)
                dtrain = pd.merge(dtrain , train_temp , on =['ID','y'])
                dcv = pd.merge(dcv , train_temp , on =['ID','y'])
                dtest = pd.merge(test , test_temp , on =['ID'])
                
                xgtrain = xgb.DMatrix(dtrain[predictors], label=dtrain[target])
                xgcv = xgb.DMatrix(dcv[predictors], label=dcv[target])
                xgtest = xgb.DMatrix(dtest[predictors])
                watchlist  = [ (xgtrain,'train'),(xgcv,'cv')]
                a={}
                model=xgb.train(plst,xgtrain,int(rounds),watchlist,early_stopping_rounds=5000000,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
                cv_index = train[train['ID'].isin(dcv.ID.values)].index
                train=train.set_value(cv_index,name+str(depth)+'_'+str(num),model.predict(xgcv))
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
