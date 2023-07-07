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

data =  pd.read_csv('./data/train_data_clean_reduced_encoded.csv')
df = pd.DataFrame(data)


X = df

#Downsampling

#mult is ratio of legitimate data to fraudulent data
mult = 1

fClass = X[X['0'] == 1]
lClass = X[X['0'] == 0]

fCount = int(len(fClass) * mult)

#downsample
lClass = lClass.sample(fCount, axis=0)

X = pd.concat([fClass.reset_index(drop=True), lClass.reset_index(drop=True)],axis=0)

#shuffle data
X.sample(frac = 1)
y =np.array(X['0'])
X = X.drop(columns=['0'])

#End downsampling


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
gnb = GaussianNB()
fit = gnb.fit(X_train, y_train)
y_pred = fit.predict(X_test)
acc = metrics.accuracy_score(y_test,y_pred)
print(acc * 100, "% Accuracy")
balanced_acc = metrics.balanced_accuracy_score(y_test, y_pred)
print(balanced_acc * 100, '% Balanced Accuracy')

















#Future work
# test = pd.read_csv('./data/test_data_clean_onehot.csv')
# test_df = pd.DataFrame(test)

# TEST_X = test_df.drop(columns=['0'])
# TEST_y = np.array(test_df['0'])

# TEST_pred = fit.predict(TEST_X)
# TEST_acc = metrics.accuracy_score(TEST_y, TEST_pred)
# print(TEST_acc * 100, "% Accuracy")
# balanced_TEST_acc = metrics.balanced_accuracy_score(TEST_y, TEST_pred)
# print(balanced_TEST_acc * 100, '% Balanced Accuracy')

