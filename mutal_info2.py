import pandas as pd
import numpy as np
train = pd.read_csv('train4b.csv')
test  = pd.read_csv('test4b.csv')


from sklearn import metrics
#feature_selection.mutual_info_classif(train.X0.reshape(-1, 1),train.X1.reshape(-1, 1))
predictors = [x for x in train.keys() if x not in ['yy']]
#result=np.load('mutual_info.npy').item()
#all_result=sorted([(round(result[x],2),x) for x in result if x[0]!=x[1] and x in train.keys()],reverse=True)
print '\n'
#print all_result[0:10]
result = {}
#It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
for i in range(1,len(predictors)):
    print predictors[i]
    for j in range(i+1,len(predictors)):
        feat1,feat2 = predictors[i],predictors[j]
        #if (feat1,feat2) not in result:
        result[(feat1,feat2)]=metrics.mutual_info_score(train[feat1],train[feat2])
       
#np.save('mutual_info2.npy',result) 

def non_PCA (x):
    cannot = ['pca_1', 'ica_1', 'tnse_1', 'pca_2', 'ica_2',\
              'tnse_2', 'pca_3', 'ica_3', 'tnse_3', 'pca_4', \
              'ica_4', 'pca_5', 'ica_5', 'pca_6', 'ica_6',\
              'pca_7', 'ica_7', 'pca_8', 'ica_8', 'pca_9', \
              'ica_9', 'pca_10', 'ica_10']
    if x[0] in cannot:
        return False
    if x[1] in cannot:
        return False
    return True

all_result=sorted([(round(result[x],2),x) for x in result if x[0]!=x[1] and non_PCA(x)],reverse=True)
print '\n'
print all_result[0:10]
