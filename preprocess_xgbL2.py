import pandas as pd
import numpy as np
import xgboost as xgb
train=pd.read_csv('trainLvl2b.csv')
test=pd.read_csv('testLvl2b.csv')
train = train.T.drop_duplicates().T
predictors=[i for i in train.keys() if i not in ['y','ID']]
test=test[predictors]

target='y'
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.01
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
dictt={}
for depth in [3,6,9]:
    for child in [5,15]:
        name='L2_predict_'+str(depth)+str(child)+'_'
        print depth,child
        dictt[depth]=[]
        params["max_depth"] = depth
        plst = list(params.items())
        rounds=0
        for num in range(0,20):
            train=train.sample(frac=1)
            train[name+str(depth)+'_'+str(num)]=0
            test[name+str(depth)+'_'+str(num)]=0
            max_n = 5
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
                    model=xgb.train(plst,xgtrain,8000,watchlist,early_stopping_rounds=500,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
                    print a['train']['r2'][model.best_iteration],a['cv']['r2'][model.best_iteration],model.best_iteration
                    rounds+=model.best_iteration/max_n
                    print rounds
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
                train=train.set_value(dcv.index,name+str(depth)+'_'+str(num),model.predict(xgcv))
                test.set_value(test.index,name+str(depth)+'_'+str(num),test[name+str(depth)+'_'+str(num)]+model.predict(xgtest)/max_n)
                dictt[depth] += [a['cv']['r2'][model.best_iteration],]
            print r2_score(train.y,train[name+str(depth)+'_'+str(num)])
            print '\n'
 
#test['y']=map(lambda x:np.mean(x),np.array(test[['depth_4_0' ,  'depth_4_1',   'depth_4_2','depth_4_3','depth_4_4']]))
#test[['ID','y']].to_csv('first.csv',index=0)
train['ID'] = train.I
test['ID'] = test.I
predictors=[i for i in train.keys() if (name[0:5] in i) or i in ['ID','y']]
train[predictors].to_csv('train_%s.csv' %name,index=0)
predictors=[i for i in test.keys() if (name[0:5] in i) or i in ['ID','y']]
test[predictors].to_csv('test_%s.csv' %name,index=0)
'''
gblinear3_ ('r2', 0.56455627384897611)
gblinear4_ ('r2', 0.56231826068569757)
gblinear5_ ('r2', 0.56646811125111962)
gblinear6_ ('r2', 0.56520535013907325)
keras_MAE2_lgY_relu ('r2', 0.5647133768384911)
keras_MAE2_relu ('r2', 0.55966247078466769)
keras_MSE2_relu ('r2', 0.576911022072226)
keras_MSE2_tanh ('r2', 0.5733064301468066)
keras_per_relu ('r2', 0.53998518411222873)
keras_r2_relu ('r2', 0.57500453624382097)
keras_r2b ('r2', 0.5559808314247574)
lasso_0.01_ ('r2', 0.58710550646571935)
lasso_0.05_ ('r2', 0.58226933556598615)
lasso_0.1_ ('r2', 0.56575506771932371)
lasso_0.3_ ('r2', 0.5341519399327439)
lasso_0.5_ ('r2', 0.52153174213969611)
predM ('r2', 0.55879111198961096)
predPer ('r2', 0.55203360792319711)
ridge_20.0_ ('r2', 0.58285984350427489)
ridge_30.0_ ('r2', 0.58377114148993614)
ridge_40.0_ ('r2', 0.58293085837611347)
xMSE23_ ('r2', 0.59249262263799585)
xMSE24_ ('r2', 0.59037845984396675)
xMSE25_ ('r2', 0.58824823076052313)
xMSE26_ ('r2', 0.58709960214826706)
x_mse_expY3_ ('r2', 0.53331470967750472)
x_mse_expY4_ ('r2', 0.54486893057182062)
x_mse_expY5_ ('r2', 0.55097506057227186)
x_mse_expY6_ ('r2', 0.55219303912264595)
x_mse_sqrY3_ ('r2', 0.59350046182740512)
x_mse_sqrY4_ ('r2', 0.59180121656851892)
x_mse_sqrY5_ ('r2', 0.59014810148350549)
x_mse_sqrY6_ ('r2', 0.58837503020650317)
'''
