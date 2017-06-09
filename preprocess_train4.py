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
test['y']=np.nan
train_id =len(train)
total=pd.concat([train,test], axis=0)
##
##
# find all categorical features
cf = total.select_dtypes(include=['object']).columns

# make one-hot-encoding convenient way - pandas.get_dummies(df) function
dummies = pd.get_dummies(
    total[cf],
    drop_first=False # you can set it = True to ommit multicollinearity (crucial for linear models)
)

# get rid of old columns and append them encoded
total = pd.concat(
    [
        total.drop(cf, axis=1), # drop old
        dummies # append them one-hot-encoded
    ],
    axis=1 # column-wise
)
train=total.iloc[0:train_id]
test=total.iloc[train_id:]
for i in sorted(train.keys()):
    if train[i].dtype=='O':
        print i
        temp ={}
        counter=0
        for j in sorted(train[i].unique()):
            #test first
            temp[j]=counter
            counter +=1
        if i!='y' :
            test[i]=test[i].map(temp)
            train[i]=train[i].map(temp)
corr=train.corr()
pairs=[]
for A in range(len(corr)):
    for B in range(A+1,len(corr)):
        i,j=corr.keys()[A],corr.keys()[B]
        if i!=j:
            if (corr[i][j]**2)**.5>0.98:
                print i,j,corr[i][j]
                if j in train and '_' not in j:
                    train=train.drop(j,axis=1)
                    test=test.drop(j,axis=1)

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
            print i,'error'
train=train.fillna(0)
test=test.fillna(0)
n_comp = 10
test['y']=np.nan
train_id =len(train)
total=pd.concat([train,test], axis=0)
# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y",'ID'], axis=1))
pca2_results_test = pca.transform(test.drop(["y",'ID'], axis=1))

pca2 = PCA(n_components=100, random_state=42)
temp_train = pca2.fit_transform(train.drop(["y",'ID'], axis=1))
temp_test = pca2.transform(test.drop(["y",'ID'], axis=1))

# ICA
ica = FastICA(n_components=n_comp, random_state=42,max_iter=1000)
ica2_results_train = ica.fit_transform(train.drop(["y",'ID'], axis=1))
ica2_results_test = ica.transform(test.drop(["y",'ID'], axis=1))

# TNSE
from sklearn.manifold import TSNE
tsne_comp=3
tsne = TSNE(n_components=tsne_comp, random_state=42)
tsne_all = tsne.fit_transform(total.drop(["y",'ID'], axis=1))
tsne2_results_train = tsne_all[0:train_id]
tsne2_results_test =  tsne_all[train_id:]




# Append decomposition components to datasets
for i in range(1, n_comp+1):
    train['pca_' + str(i)] = pca2_results_train[:,i-1]
    test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    train['ica_' + str(i)] = ica2_results_train[:,i-1]
    test['ica_' + str(i)] = ica2_results_test[:, i-1]
    if i <= tsne_comp:
        train['tnse_' + str(i)] = tsne2_results_train[:,i-1]
        test['tnse_' + str(i)] = tsne2_results_test[:, i-1]


train.to_csv('train4b.csv',index=0)
test.to_csv('test4b.csv',index=0)
##train=pd.read_csv('train4b.csv')
##test=pd.read_csv('test4b.csv')


