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
import prince

np.set_printoptions(threshold=np.inf)

#unfinished. Need to finish building the components. The issue with this is although we have the top components for creating the reduced data, I do not know how to 
#do it with our categorical data. Potentially one hot encode and then scale the whole group 

data =  pd.read_csv('data/train_transaction_clean_unencoded.csv')
df = pd.DataFrame(data)

#This analysis yielded a new n = 159

best_n = 159
y = df.isFraud.values
X = df.drop(columns=['isFraud'])
 

  
X_objs = X.select_dtypes(np.object_).columns

d = len(X.columns)

algo = prince.FAMD(
    n_components=best_n,
    n_iter=3,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error"  # same parameter as sklearn.preprocessing.OneHotEncoder
)
algo = algo.fit(X)

X = X.drop(columns = X_objs)

# summ = np.array(algo.eigenvalues_summary)
# for i in range(len(summ)):
#     print(i)
#     print(summ[i])

#Column contributions are barely out of order for some reason, but they are perfectly ordered before the categorical values, so it does not matter
cols = algo.column_contributions_
obj_inds = np.array([np.where(cols.index == obj)[0][0] for obj in X_objs])
min_in = np.min(obj_inds)


objs = cols.iloc[min_in:,:].values
cols = cols.iloc[:min_in,:].values
transform = np.matmul(X, cols)

objs = np.sum(objs,axis = 0)
transform = np.add(transform,objs)

red_df = pd.DataFrame(np.concatenate([np.transpose(y).reshape(len(y),1),np.array(transform)], axis=1))
red_df.to_csv(f"data/train_data_clean_reduced_encoded.csv",index=False)


TEST_DATA =  pd.read_csv('data/test_data_clean_unencoded.csv')
TEST_SET = pd.DataFrame(TEST_DATA)


TEST_y = TEST_SET.isFraud.values
TEST_X = TEST_SET.drop(columns=['isFraud'])
 
TEST_X = TEST_X.drop(columns = X_objs)

transform = np.matmul(TEST_X, cols)

transform = np.add(transform,objs)

TEST_red_df = pd.DataFrame(np.concatenate([np.transpose(TEST_y).reshape(len(TEST_y),1),np.array(transform)], axis=1))
TEST_red_df.to_csv(f"data/test_data_clean_reduced_encoded.csv",index=False)


