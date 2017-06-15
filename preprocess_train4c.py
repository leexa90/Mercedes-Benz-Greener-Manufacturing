import pandas as pd
import numpy as np
import xgboost as xgb
'''
PART 1
'''
train=pd.read_csv('train4b.csv')
test=pd.read_csv('test4b.csv')
train=train[train.y < 200]
test['y']=np.nan
train_id =len(train)
total=pd.concat([train,test], axis=0)



'''
ADD PAIRWISE ITNERACTIONS THAT GIE MOST GAIN
X115|X261
X136|X261
X162|X314
X0_az|X314
X261|ica_7
X261|pca_6
X313|ica_7
X136|X314
X115|ica_2
X118|X314
X234|X313
X275|ica_2
X314|ica_7
X315|ica_2
X127|ica_7
X118|X261
ica_2|pca_6
X127|X136
X118|ica_7
X118|ica_2
'''
feat  = [['X115','X261'],['X136','X261'],['X162','X314'],['X0_az','X314'],
         ['X136','X314'],['X118','X314'],['X234','X313'],['X118','X261'],
         ['X127','X136']]
feat2 = [['X127','X136','X261'],['X162','X314','X315']]

for pair in feat:
    total['%s_%s'%(pair[0],pair[1])] = 0
    counter = 0
    for i in total[pair[0]].unique():
        for j in total[pair[1]].unique():
            temp = total[(total[pair[0]]==i) & (total[pair[1]]==j)]
            total.set_value(temp.index,'%s_%s'%(pair[0],pair[1]),  str(counter))
            counter += 1


for pair in feat2:
    total['%s_%s_%s'%(pair[0],pair[1],pair[2])] = 0
    counter = 0
    for i in total[pair[0]].unique():
        for j in total[pair[1]].unique():
            for k in total[pair[2]].unique():
                temp = total[(total[pair[0]]==i) & (total[pair[1]]==j) & (total[pair[2]]==k)]
                total.set_value(temp.index,'%s_%s_%s'%(pair[0],pair[1],pair[2]),  str(counter))
                counter += 1

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
train.to_csv('train4c.csv',index=0)
test.to_csv('test4c.csv',index=0)
