import pandas as pd
import numpy as np
import time

#change file_name and K as needed
file_name = "train_transaction.csv"
K = 10

data =  pd.read_csv(f'./data/{file_name}.csv')
df = pd.DataFrame(data)

data = df.select_dtypes([np.number]).dropna(axis=1, how='all')

np_data = data.to_numpy()

def pairwise_dist(x, y):
    """
    Args:
        x: N x D numpy array
        y: M x D numpy array
    Return:
        dist: N x M array, where dist[i, j] is the euclidean distance between
        x[i, :] and y[j, :]
    """
    xx = np.sum(x**2, axis=1)[:, np.newaxis]
    yy = np.sum(y**2, axis=1)[np.newaxis, :]
    xy =  x @ np.transpose(y)
    squared_dist = -2 * xy + xx + yy
    squared_dist[squared_dist < 0] = 0
    return np.sqrt(squared_dist)

def kNN(data, K, indices, index):
    """
        Function to clean or "fill in" NaN values in incomplete data points based on
        the average value for that feature for the K-nearest neighbors in the complete data points. 

        Args:
            data: N x D+1 numpy array, all the data
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            indices: list of complete indices over which none are nan from which to calculate nearest neighbors (d COLUMNS)
            index: index with nan values to be replaced (COLUMN)
        Return:
            nothing; data is cleaned in place
    """
    nan_indices = np.isnan(data[:, index])
    non_nan_indices = (nan_indices==False).nonzero()[0]
    complete_points = data[non_nan_indices[:, np.newaxis], indices[np.newaxis, :]] #m x d array where n is # points without nan
    incomplete_points = np.transpose(data[nan_indices, indices[:, np.newaxis]]) #n x d array where m is # points with nan; this broadcast worked weird so I had to transpose it

    
    if non_nan_indices.shape[0] <= K:
        closest_non_nan = np.repeat(data[non_nan_indices, index][np.newaxis, :], incomplete_points.shape[0], axis=0)
    else:
        distances = pairwise_dist(incomplete_points, complete_points) #n x m array of distances between incomplete and complete points
        closest_non_nan = data[non_nan_indices[np.argpartition(distances, K, axis=1)[:, :K]], index]
        # np.argpartition(distances, K, axis=1)[:, :K] is the index of the K complete points in distance to each incomplete points; n x K array
        # non_nan_indices converts this to indices in the main data set
        # data[] grabs the values at the index in which points may be NaN
    
    data[nan_indices, index] = np.mean(closest_non_nan, axis=1)
    return

def get_indices_order(data):
    """
        Function to get the order of indices for which to clean the data.
        Sorts the data by number of NaNs in a column. Columns with less NaNs should be cleaned first, more NaNs should be cleaned last.

        Args:
            data: N x D numpy array, the data with ONLY the features
        Return:
            number_full: number of columns with NO nans
            order: N length numpy array, the order of indices to be cleaned
    """
    count_nans = np.sum(np.isnan(data), axis=0)
    return count_nans.shape[0] - np.count_nonzero(count_nans), np.argsort(count_nans)

def clean_data(data, K):
    """
        Function to clean the data by replacing all NaN values with the average of values near it.

        Args:
            data: N x D numpy array, the data getting cleaned containing ONLY the features. All should be of the same label.
            K: the number of near neighbors to average.
        Return:
            nothing; data is changed in place
    """
    i, order = get_indices_order(data)
    column_count = order.shape[0]
    
    while i < column_count:
        indices = order[:i]
        #print(i)
        #print(indices)
        kNN(data, K, indices, order[i])
        i = i + 1

    return


non_fraud_indices = (np_data[:, 1]==0).nonzero()[0]
fraud_indices = (np_data[:, 1]==1).nonzero()[0]

non_fraud_data = np_data[non_fraud_indices, 2:]
fraud_data = np_data[fraud_indices, 2:]

clean_data(non_fraud_data, K)
clean_data(fraud_data, K)

np_data[non_fraud_indices, 2:] = non_fraud_data
np_data[fraud_indices, 2:] = fraud_data

new_df = pd.DataFrame(np_data, columns=data.columns)
new_df.to_csv(f"data/{file_name}_clean.csv",index=False)