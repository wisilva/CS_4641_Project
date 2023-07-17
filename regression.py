import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_file = 'train_transaction_clean'

data =  pd.read_csv(f'./data/{data_file}.csv')
df = pd.DataFrame(data)

data = df.select_dtypes([np.number]).dropna(axis=1, how='all')

np_data = data.to_numpy()

X = np_data[:, 2:]
y = np_data[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


model = Ridge()
model.fit(X_train, y_train)

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict_sigmoid = 1 / (1 + np.exp(-train_predict))
test_predict_sigmoid = 1 / (1 + np.exp(-test_predict))
for i in range(100):
    cutoff = 0.01 + i * 0.01

    print("cutoff: ", cutoff)
    
    y_train_hat = np.where(train_predict_sigmoid > cutoff, 1, 0)

    train_accuracy = accuracy_score(y_train, y_train_hat)
    print("Train accuracy:", train_accuracy)

    y_test_hat = np.where(test_predict_sigmoid > cutoff, 1, 0)

    test_accuracy = accuracy_score(y_test, y_test_hat)

    print("Test accuracy: ", test_accuracy)

    N = y_test.shape[0]
    y_n = y_test==0
    y_nh = y_test_hat==0
    y_p = y_test==1
    y_ph = y_test_hat==1

    true_negative = np.sum(y_n & y_nh)
    true_positive = np.sum(y_p & y_ph)
    false_negative = np.sum(y_p & y_nh)
    false_positive = np.sum(y_n & y_ph)

    print('True negative:', true_negative)
    print('True positive:', true_positive)
    print('False negative:', false_negative)
    print('False positive:', false_positive)
    print(N)