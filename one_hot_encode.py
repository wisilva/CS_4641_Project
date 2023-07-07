from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import csv
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing as pre
from sklearn import impute, metrics,utils, neighbors
from sklearn.pipeline import Pipeline 
import pandas as pd
import scipy.sparse as sp
from imblearn import over_sampling


data =  pd.read_csv('data/train_transaction_clean_downsampled_unencoded.csv')
df = pd.DataFrame(data)

TEST_DATA =  pd.read_csv('data/test_data_unencoded.csv')
TEST_SET = pd.DataFrame(TEST_DATA)


y = df.isFraud.values
X = df.drop(columns = ['isFraud']).fillna("None")

X_objs = X.select_dtypes(np.object_).values
X_nums = X.select_dtypes(np.number).values

obj_trans = Pipeline([ ('onehot', pre.OneHotEncoder(sparse_output = False)) ])
X_objs = obj_trans.fit_transform(X_objs)

categories = obj_trans['onehot'].categories_

X = np.concatenate( [X_nums,X_objs], axis=1)


new_df = pd.DataFrame(np.concatenate([np.transpose(y).reshape(len(y),1),np.array(X)], axis=1) )

new_df.to_csv(f"data/train_transaction_clean_downsampled_onehot.csv",index=False)




TEST_X = TEST_SET.drop(columns=['isFraud'])
TEST_y = TEST_SET.isFraud.values

TEST_trans = Pipeline([('onehot', pre.OneHotEncoder(categories = categories, sparse_output = False, handle_unknown = 'ignore')) ])

TEST_objs = TEST_X.select_dtypes(np.object_).fillna("None")
TEST_nums = TEST_X.select_dtypes(np.number)

TEST_objs = TEST_trans.fit_transform(TEST_objs)

TEST_X = np.concatenate( [TEST_nums,TEST_objs], axis=1)

TEST_df = pd.DataFrame(np.concatenate([np.transpose(TEST_y).reshape(len(TEST_y),1),np.array(TEST_X)], axis=1) )
TEST_df.to_csv(f"data/test_data_onehot.csv",index=False)

