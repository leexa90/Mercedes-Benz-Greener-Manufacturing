import pandas as pd
import numpy as np
import xgboost as xgb
'''
PART 1
'''
##train=pd.read_csv('train.csv')
##test=pd.read_csv('test.csv')
##
##
##'''
##convert string to interger
##'''
##for i in train.keys():
##    if train[i].dtype=='O':
##        print i
##        #test first
##        temp=train.groupby(i).y.mean().to_dict()
##        test[i]=test[i].map(temp)
##
##        
##        for value in train[i].unique():
##            temp=train[train[i]==value]
##            List=list(train[train[i]==value].index)
##            # bayesian modelling, giving value of y given particular category of X0-X8
##            for num in range(len(List)):
##                wanted=List[0:num]+List[num+1:]
##                temp2=temp[temp.index.isin(wanted)]
##                temp3=np.mean(temp2.y)
##                train=train.set_value(List[num],i,temp3)
##        train[i]=train[i].astype(np.float64)
##train.to_csv('train2.csv',index=0)
##test.to_csv('test2.csv',index=0)
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

y_train = train["y"]
y_mean = np.mean(y_train)    
        
##        for value in train[i].unique():
##            temp=train[train[i]==value]
##            List=list(temp.index)
##            # bayesian modelling, giving value of y given particular category of X0-X8
##            for num in range(len(List)):
##                wanted=List[0:num]+List[num+1:]
##                temp2=temp[temp.index.isin(wanted)]
##                temp3=np.mean(temp2.y)
##                train=train.set_value(List[num],i+'_2',temp3)
        #train[i]=train[i].astype(np.float64)
train.to_csv('train3.csv',index=0)
test.to_csv('test3.csv',index=0)
'''
Part2
'''
train=pd.read_csv('train3.csv')
test=pd.read_csv('test3.csv')
for i in train.keys():
    if len(train[i].unique())==1:
        print i        
        del train[i]
        del test[i]
from sklearn.decomposition import PCA, FastICA
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
n_comp = 10

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]

train.to_csv('train4.csv',index=0)
test.to_csv('test4.csv',index=0)
train=pd.read_csv('train4.csv')
test=pd.read_csv('test4.csv')
predictors=[i for i in train.keys() if i not in ['ID','y']]
target='y'
params = {}
params["objective"] = "reg:linear"
params["eta"] = 0.005
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
    return 'r2', r2_score(labels, preds)
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
        train['depth_PCA_'+str(depth)+'_'+str(num)]=0
        test['depth_PCA_'+str(depth)+'_'+str(num)]=0
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
            model=xgb.train(plst,xgtrain,10000,watchlist,early_stopping_rounds=500,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
            print a['train']['r2'][model.best_iteration],a['cv']['r2'][model.best_iteration],model.best_iteration
            model=xgb.train(plst,xgtrain,model.best_iteration,watchlist,early_stopping_rounds=500,evals_result=a,feval=xgb_r2_score, maximize=1,verbose_eval=False)
            train=train.set_value(dcv.index,'depth_PCA_'+str(depth)+'_'+str(num),model.predict(xgcv))
            test.set_value(test.index,'depth_PCA_'+str(depth)+'_'+str(num),test['depth_PCA_'+str(depth)+'_'+str(num)]+model.predict(xgtest)/max_n)
            dictt[depth] += [a['cv']['r2'][model.best_iteration],]
        print '\n'
    print np.mean([float(x) for x in dictt[depth]]),np.std([float(x) for x in dictt[depth]])/len(dictt[depth])**.5
for i in dictt:
	print np.mean([float(x) for x in dictt[i]]),np.std([float(x) for x in dictt[i]])/len(dictt[depth])**.5

#test['y']=map(lambda x:np.mean(x),np.array(test[['depth_4_0' ,  'depth_4_1',   'depth_4_2','depth_4_3','depth_4_4']]))
#test[['ID','y']].to_csv('first.csv',index=0)
            
predictors=[i for i in train.keys() if ('depth' in i) or i in ['ID','y']]
train[predictors].to_csv('train_PCA.csv',index=0)
predictors=[i for i in test.keys() if ('depth' in i) or i in ['ID','y']]
test[predictors].to_csv('test_PCA.csv',index=0)
'''
0.5734342 0.00993828205311 3
0.5816492 0.00993222393561 4
0.579851 0.00999689565915 5
0.5632972 0.0184149400565 6
0.5734342 0.00993828205311 7
'''
