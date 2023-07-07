from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing as pre
from sklearn import impute, metrics
from sklearn.pipeline import Pipeline 

import pandas as pd
import scipy.sparse as sp

data =  pd.read_csv('./data/train_transaction_clean_downsampled_onehot.csv')
df = pd.DataFrame(data)

X = df.drop(columns=['0'])
y =np.array(df['0'])
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc * 100, "% Accuracy")
balanced_acc = metrics.balanced_accuracy_score(y_test, y_pred)
print(balanced_acc * 100, '% Balanced Accuracy')
