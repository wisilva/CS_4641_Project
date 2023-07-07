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

mult = 1.0 #downsamples the legitimate cases to this ratio of the number of fraudulent cases 
bucket_size = 5000 #higher value increases time to run KNN at a quadratic rate but should provide more accurate nearest neighbors

data =  pd.read_csv('./data/train_transaction.csv')
df = pd.DataFrame(data)


#Untouched Test Set, about 20% of the data
TEST_SET = df.sample(n=100000,axis=0)

#Just to visually confirm that the Test dataset has a similar ratio
fraud_ratio = len(df[df.isFraud == 1]) / len(df[df.isFraud == 0])

test_ratio = len(TEST_SET[TEST_SET.isFraud == 1]) / len(TEST_SET[TEST_SET.isFraud == 0])




df = df.drop(TEST_SET.index)

#normalize (min/max or standardization?) -> downsample -> knn impute with buckets of size bucket_size-> SMOTENC upsample (includes PCA)

#Min/max normalization
#scaler = pre.MinMaxScaler()

#Standardization
scaler = pre.StandardScaler()


y = df.isFraud
X = df.drop(columns=['TransactionID', 'isFraud', 'TransactionDT'])

obj_inds = np.where(type(df.iloc(0)) == np.dtype('O'))

X_nums = X.select_dtypes(np.number) 
num_types = X_nums.columns
X_objs = X.select_dtypes(np.object_)
obj_types = X_objs.columns

#Standardize the numerical values
X_nums = pd.DataFrame(scaler.fit_transform(X_nums.values), columns= num_types)


X = pd.concat([X_nums.reset_index(drop=True), X_objs.reset_index(drop=True)], axis = 1)

#Unnecessary, this is actually just the last 14 elements
obj_inds = np.array( [np.where(X.columns == obj) for obj in obj_types]).flatten()

#add y back before downsample
df = pd.concat([y.reset_index(drop=True), X.reset_index(drop=True)], axis = 1)

x_types = X.columns


fClass = X[df.isFraud == 1]
lClass = X[df.isFraud == 0]

fCount = int(len(fClass) * mult)

#downsample
lClass = lClass.sample(fCount, axis=0)


#split by label
f_nums = fClass.select_dtypes(np.number) 
f_objs = fClass.select_dtypes(np.object_)


l_nums = lClass.select_dtypes(np.number) 
l_objs = lClass.select_dtypes(np.object_)

num_trans = Pipeline([('imputer', impute.KNNImputer(n_neighbors=20, keep_empty_features=True))])

#For legitimate data
num_buckets = len(l_nums) // bucket_size

l_split = np.array_split(l_nums,num_buckets, axis= 0)
for i in range(num_buckets):
    l_split[i] = num_trans.fit_transform(l_split[i])

l_nums = np.concatenate(l_split, axis =0)

lClass = pd.DataFrame(np.concatenate([l_nums, l_objs], axis = 1), columns = x_types)

lLabels = np.zeros(len(l_nums))
yLegitimate = pd.DataFrame(lLabels, columns = ['isFraud'])

lClass = pd.concat([yLegitimate.reset_index(drop=True),lClass.reset_index(drop=True)], axis =1 )


#For fraudulent Data

num_buckets = len(f_nums) // bucket_size

f_split = np.array_split(f_nums,num_buckets, axis= 0)
for i in range(num_buckets):
    f_split[i] = num_trans.fit_transform(f_split[i])

f_nums = np.concatenate(f_split, axis =0)

fClass = pd.DataFrame(np.concatenate([f_nums, f_objs], axis = 1), columns = x_types)

fLabels = np.ones(len(f_nums))
yFraud = pd.DataFrame(fLabels, columns = ['isFraud'])

fClass = pd.concat([yFraud.reset_index(drop=True),fClass.reset_index(drop=True)], axis =1 )

X = pd.concat([fClass.reset_index(drop=True), lClass.reset_index(drop=True)],axis=0)
#Write unencoded data
X_unenc = X.fillna("None")
X_unenc = X_unenc.sample(frac=1)
X_unenc.to_csv(f"data/train_transaction_clean_downsampled_unencoded.csv",index=False)


TEST_X = TEST_SET.drop(columns=['TransactionID', 'isFraud', 'TransactionDT'])
TEST_y = TEST_SET.isFraud

#some unnecessary code here. Y doesnt need to be dropped at all currently
TEST_objs = TEST_X.select_dtypes(np.object_).fillna("None")
TEST_nums = TEST_X.select_dtypes(np.number)


test_scaler = pre.StandardScaler()
TEST_nums = pd.DataFrame(test_scaler.fit_transform(TEST_nums.values), columns= num_types)


TEST_df = pd.concat([TEST_nums.reset_index(drop=True),TEST_objs.reset_index(drop=True)],axis = 1)
TEST_df = pd.concat([TEST_y.reset_index(drop=True),TEST_df.reset_index(drop=True)],axis = 1)
TEST_df = TEST_df.sample(frac = 1)

TEST_df.to_csv(f"data/test_data_unencoded.csv",index=False)

