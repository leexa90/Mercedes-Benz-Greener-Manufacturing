import pandas as pd
import numpy as np

depth =pd.read_csv('train_depth.csv')
poisson = pd.read_csv('train_poisson.csv')
keras = pd.read_csv('train_keras.csv')
lasso = pd.read_csv('train_lasso.csv')
from sklearn.metrics import r2_score
def xgb_r2_score(preds, dtrain):
    labels = dtrain
    return 'r2', r2_score(labels, preds)
find_combined=0
if find_combined:
    for i in [3,4,5,6]:
        for j in [x for x in depth.keys() if type(x) != int]:
            if 'depth' in j and str(i) == j.split('_')[1]:
                if i in depth:
                    depth[i] += depth[j]/5
                else:
                    depth[i] = depth[j]/5 
        print i, xgb_r2_score(depth[i],depth.y.values)


    for i in [3,4,5,6]:
        for j in [x for x in poisson.keys() if type(x) != int]:
            if 'depth' in j and str(i) == j.split('_')[2]:
                if i in poisson:
                    poisson[i] += poisson[j]/5
                else:
                    poisson[i] = poisson[j]/5 
        print i, xgb_r2_score(poisson[i],poisson.y.values)
    pred = [x for x in keras.keys() if 'pred' in x]
    keras['all']=map(lambda x :np.mean(x), np.array(keras[pred]))
    for i in keras:
        print i, xgb_r2_score(keras[i],keras.y.values)

    for i in [0.01,0.05,0.1,0.3,0.5,1]:
        for j in [x for x in lasso.keys() if type(x) == str]:
            if 'lasso' in j and str(i) == j.split('_')[1]:
                if i in lasso:
                    lasso[i] += lasso[j]/10
                else:
                    lasso[i] = lasso[j]/10
        print i, xgb_r2_score(lasso[i],lasso.y.values)
    pred = [x for x in keras.keys() if 'pred' in x]
    keras['all']=map(lambda x :np.mean(x), np.array(keras[pred]))
train = pd.read_csv('train4.csv')
test = pd.read_csv('test4.csv')
for i in [depth,poisson,lasso,keras]:
    train= pd.merge(train,i,on=['ID','y'])
    test= pd.merge(train,i,on=['ID','y'])

'''
# depth rmse
3 ('r2', 0.57237382209669785)
4 ('r2', 0.57020425775518424)
5 ('r2', 0.55798494873472926)
6 ('r2', 0.56200018569323928)
# poisson objective function
3 ('r2', 0.56986100795504513)
4 ('r2', 0.56548764975960686)
5 ('r2', 0.56274212421154801)
#keras
pred0 ('r2', 0.54295588086295099)
pred1 ('r2', 0.53961706814728239)
pred2 ('r2', 0.53578621527511805)
pred3 ('r2', 0.53626750856917094)
pred4 ('r2', 0.54360345773231467)
pred5 ('r2', 0.54093617164528662)
pred6 ('r2', 0.54048595302752123)
pred7 ('r2', 0.54047428605675196)
pred8 ('r2', 0.5415848436056877)
pred9 ('r2', 0.54183830990998816)
all ('r2', 0.55508147618918646)
lasso
lasso_0.01_0 ('r2', 0.55807905856136386)
lasso_0.01_1 ('r2', 0.56185551055922989)
lasso_0.01_2 ('r2', 0.56220847416226483)
lasso_0.01_3 ('r2', 0.55421137399816778)
lasso_0.01_4 ('r2', 0.55726614737381031)
lasso_0.01_5 ('r2', 0.55854266007799747)
lasso_0.01_6 ('r2', 0.56101219766847177)
lasso_0.01_7 ('r2', 0.55915752339925895)
lasso_0.01_8 ('r2', 0.56211136558504715)
lasso_0.01_9 ('r2', 0.55886519738862184)
all ('r2', 0.56124984205926287)
0.01 ('r2', 0.56175088465174927)
0.05 ('r2', 0.55866088996763752)
0.1 ('r2', 0.54293469830872954)
0.3 ('r2', 0.51111365408503606)
0.5 ('r2', 0.49635561514626569)
1 ('r2', 0.44715340226893341)
'''
