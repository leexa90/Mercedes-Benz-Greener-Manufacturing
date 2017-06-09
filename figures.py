import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# figures
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
''' mutual info'''
for i in sorted(train.keys()):
    if train[i].dtype=='O':
        temp ={}
        counter=1
        for j in sorted(train[i].unique()):
            #test first
            temp[j]=counter
            counter +=1
        print temp
        test[i]=test[i].map(temp)
        train[i]=train[i].map(temp)

from sklearn import feature_selection
#feature_selection.mutual_info_classif(train.X0.reshape(-1, 1),train.X1.reshape(-1, 1))
predictors = [x for x in train.keys() if x not in ['yy']]
result=np.load('mutual_info.npy').item()
all_result=sorted([(round(result[x],2),x) for x in result if x[0]!=x[1]],reverse=True)
print '\n'
print all_result[0:10]
#It is equal to zero if and only if two random variables are independent, and higher values mean higher dependency.
##for feat1 in predictors:
##    for feat2 in predictors:
##        if (feat1,feat2) not in result:
##            result[(feat1,feat2)]=feature_selection.mutual_info_regression(train[[feat2]],np.array(train[feat1]))[0]
##        if len(result)%378==0:
##            print feat1
##            
#np.save('mutual_info.npy',result)       

# X0 and impacts
fig, ax1 = plt.subplots(figsize=(27, 27))
fig.tight_layout(pad=4.0, w_pad=4, h_pad=4)
temp =[]
name = []
feat1 = 'X2'
feat2 = 'X250'
if feat2 != '':
    for i in sorted(train[feat1].unique()):
        for j in sorted(train[feat2].unique()):
            temp += [train[(train[feat1] == i) & (train[feat2] == j)]['y']]
            name += [str(i)+'_'+str(j),]
else:
    for i in sorted(train[feat1].unique()):
        j=''
        temp += [train[(train[feat1] == i) ]['y']]
        name += [str(i)+'_'+str(j),]
bp=plt.boxplot(temp,notch=0, sym='+', vert=1, whis=1.5)
x_axis=0
for i in sorted(train[feat1].unique()):
    x = np.random.normal(x_axis+1, 0.02, size=len(temp[x_axis]))
    plt.plot(x,temp[x_axis] , 'r.', alpha=0.002)
    x_axis += 1
plt.xlabel('level')
plt.ylabel('y')
plt.setp(bp['boxes'], color='black')
plt.setp(bp['whiskers'], color='black')
plt.setp(bp['fliers'], color='red', marker='+')
xtickNames = plt.setp(ax1, xticklabels=name)
plt.tight_layout()
plt.grid(True)
plt.savefig(feat1+feat2)
plt.show()
