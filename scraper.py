#%%

from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

def import_wine_review(nbr_class=5, subset_size=0, verbose=False):
    # Open the two datasets as pandas DataFrames
    if (verbose) : print("Open the two datasets as pandas DataFrames")
    df1 = pd.read_csv("wine-reviews/winemag-data_first150k.csv")
    df2 = pd.read_csv("wine-reviews/winemag-data-130k-v2.csv")

    # Drop the columns only in one of the datasets
    if (verbose) : print("Drop the columns only in one of the datasets")
    df2 = df2.drop(['taster_name', 'taster_twitter_handle', 'title'], axis=1)

    # Concatenate the datasets with new indicies
    if (verbose) : print("Concatenate the datasets with new indicies")
    df = pd.concat([df1,df2], ignore_index=True)
    del df1, df2
    if (verbose) : print("    Size of the dataset : {}".format(len(df.index)))

    # Drop useless columns
    if (verbose) : print("Drop useless columns")
    df = df[['country', 'price', 'province', 'variety', 'description', 'points']]

    # Replace the description and designation with their length
    if (verbose) : print("Replace the description and designation with their length")
    df = df.fillna(value={'description':'','designation':''})
    df['description_len'] = df['description'].apply(len)
    df = df.drop(['description'], axis=1)
    # df['designation_len'] = df['designation'].apply(len)
    # df = df.drop(['description', 'designation'], axis=1)


    # Normalize
    if (verbose) : print("Normalize")
    min_max_scaler = preprocessing.MinMaxScaler()
    temp = df[['description_len']].values
    temp_scaled = min_max_scaler.fit_transform(temp)
    df[['description_len']] = pd.DataFrame(temp_scaled)
    # temp = df[['designation_len']].values
    # temp_scaled = min_max_scaler.fit_transform(temp)
    # df[['designation_len']] = pd.DataFrame(temp_scaled)
    # temp = df[['price']].values
    # temp_scaled = min_max_scaler.fit_transform(temp)
    # df[['price']] = pd.DataFrame(temp_scaled)
    del min_max_scaler, temp, temp_scaled

    # Drop the NaN values
    if (verbose) : print("Drop the NaN values")
    df = df.dropna()

    # Sample the dataset
    if (verbose) : print("Sample the dataset")
    # if (subset_size>0) : df = df[:subset_size]
    if (subset_size>0) : df = shuffle(df, n_samples=subset_size)
    if (verbose) : print("    Size of the dataset : {}".format(len(df.index)))

    # Bin the points : creating the classes
    if (verbose) : print("Bin the points : creating the classes")
    bins = list(np.arange(80,100,20/nbr_class))+[100]
    labels = range(nbr_class)
    df['binned'] = pd.cut(df['points'], bins=bins, labels=labels, include_lowest=True)
    df = df.drop('points', axis=1)
    del bins, labels

    # OneHot encode the string values
    if (verbose) : print("OneHot encode the string values")
    df = pd.get_dummies(df,columns=['country', 'province', 'variety'])

    # Split the data
    if (verbose) : print("Split the data")
    X = df.loc[:, df.columns != 'binned']
    # X = df[['price']]

    y = df['binned']

    # print("y = ")
    # print(y)
    # plt.hist(y.values,bins=5)
    # plt.show()

    return [X, y]

def import_wine_quality(subset_size=0, verbose=False):
    # Open the  dataset as a pandas DataFrame
    if (verbose) : print("Open the  dataset as a pandas DataFrame")
    df = pd.read_csv("wine-quality/wine_white.csv", sep=';')

    # Sample the dataset
    if (verbose) : print("Sample the dataset")
    if (subset_size>0) : df = df[:subset_size]
    if (verbose) : print("    Size of the dataset : {}".format(len(df.index)))

    # Split the data
    print("Split the data")
    x = df.loc[:, df.columns != 'quality']
    y = df[['quality']]

    # Normalize X
    if (verbose) : print("Normalize X")
    min_max_scaler = preprocessing.MinMaxScaler()
    for tonorm in x.columns:
        temp = x[[tonorm]].values
        temp_scaled = min_max_scaler.fit_transform(temp)
        x[[tonorm]] = pd.DataFrame(temp_scaled)

    # Crop y
    remap = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 4,
        10: 5
    }
    y = y['quality'].map(remap)

    return x,y


#%%
# # XXX, yyy = import_wine_review(nbr_class=5, subset_size=100, verbose=False)
# XXX, yyy = import_wine_quality(subset_size=0, verbose=False)

# #%%
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score

# neigh = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='manhattan')
# scores = cross_val_score(neigh, XXX, yyy, cv=2)

# print("     the minimum is {}".format(min(scores)))
# print("     the accuracy is {}".format(sum(scores)/len(scores)))
# print("     the maximum is {}".format(max(scores)))


#%%