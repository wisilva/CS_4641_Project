from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing as pre
from sklearn import impute
from sklearn.pipeline import Pipeline 

import pandas as pd
import scipy.sparse as sp

X = []
y = []

data =  pd.read_csv('./data/train_transaction.csv')
df = pd.DataFrame(data)



y =list(df.isFraud)


X = df.drop(columns=['TransactionID', 'isFraud'])

#Running into issues, so going to try splitting the data into objects and numerics

# X = X[['ProductCD', 'card1', 'card2', 'card3','card4','card5',
#        'card6','addr1','addr2','P_emaildomain','R_emaildomain', 
#        'M1','M2','M3','M4','M5','M6','M7','M8','M9']]

types = X.dtypes
X_nums = []
X_objs = X

for i in range(len(types)):
    if types[i] is not np.dtype('O'):
        X_nums.append(X[X.keys()[i]])
        X_objs = X_objs.drop(X.keys()[i], axis=1)

X_nums = pd.DataFrame(X_nums).T



num_trans = Pipeline([('imputer', impute.SimpleImputer())])
num_trans = num_trans.fit(X_nums)

X_nums = num_trans.transform(X_nums)


obj_trans = Pipeline([ ('onehot', pre.OneHotEncoder()) ])
obj_trans = obj_trans.fit(X_objs)

#After this, they are one-hot-encoded in an array of length 164
X_objs = obj_trans.transform(X_objs).toarray()

X = np.concatenate( [np.asarray(X_nums), np.asarray(X_objs)], axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print("Accuracy for %d points: %d"
                % (X_test.shape[0], (y_test != y_pred).sum()))