import pandas as pd
import numpy as np
train = pd.read_csv('train4b.csv')
test  = pd.read_csv('test4b.csv')


from sklearn import feature_selection
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
        result[(feat1,feat2)]=feature_selection.mutual_info_regression(train[[feat1,feat2]],np.array(train['y']))[0]

            
#np.save('mutual_info2.npy',result) 
