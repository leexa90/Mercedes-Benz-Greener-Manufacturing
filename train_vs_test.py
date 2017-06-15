
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.cross_validation import cross_val_score
'''
PART 1
'''
train=pd.read_csv('train4b.csv')

test=pd.read_csv('test4b.csv')

train=train[train.y < 200]

train_d = train.drop(['y'], axis=1)
test_d = test.drop(['y'], axis=1)

# To make sure we can distinguish between two classes
train_d['Target'] = 1
test_d['Target'] = 0

# We concatenate train and test in one big dataset
data = pd.concat((train_d, test_d))
predictors = [x for x in data.keys() if  x !='Target' and  x!='ID']
x_train , y_train = data.iloc[::2][predictors] ,data.iloc[::2]['Target']
x_test , y_test = data.iloc[1::2][predictors] ,data.iloc[1::2]['Target']
clf = LogisticRegression()
clf.fit(x_train, y_train)
pred = clf.predict_proba(x_test)[:,1]
auc = AUC(y_test, pred)
print "Logistic Regression AUC: {:.2%}".format(auc)
#Logistic Regression AUC: 49.27%
