
from math import log2, log10, inf
import pandas as pd
import re
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

def all(data):
    data.drop(columns=['name'], inplace=True)
    y = data.loc[:, 'type'].values
    X = data.drop(columns=['type'])
    features = X.columns.to_list()
    df = pd.DataFrame(data={'features': features})
    top_features = df.features.to_list()
    return top_features


def model_based_selection(data):
    print("Feature selection by model based selection")
    data.drop(columns=['name'], inplace=True)
    y = data.loc[:, 'type'].values
    X = data.drop(columns=['type'])
    features = X.columns.to_list()
    X = X.values
    rf = RandomForestClassifier()
    rf.fit(X, y)
    df = pd.DataFrame(data={'features': features,
                      'ranking': rf.feature_importances_})
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)
    top_features = df.features.to_list()
    return top_features


def pca_based_selection(data):
    print("Feature selection by ExtraTreesClassifier")
    data.drop(columns=['name'], inplace=True)
    y = data.loc[:, 'type'].values
    X = data.drop(columns=['type'])
    features = X.columns.to_list()
    X = X.values
    rf = ExtraTreesClassifier()
    rf.fit(X, y)
    df = pd.DataFrame(data={'features': features,
                      'ranking': rf.feature_importances_})
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)
    top_features = df.features.to_list()
    return top_features


def recursive_feature_elimination(data):
    print("Feature selection by recursive feature elimination")
    data.drop(columns=['name'], inplace=True)
    y = data.loc[:, 'type'].values
    # y = y.astype(int)
    X = data.drop(columns=['type'])
    features = X.columns.to_list()
    X = X.values
    # use linear regression as the model
    lr = LinearRegression()
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(lr, n_features_to_select=1)
    rfe.fit(X, y)
    df = pd.DataFrame(data={'features': features,
                      'ranking': rfe.ranking_})
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)
    top_features = df.features.to_list()
    return top_features


def by_mean_decrease_impurity(data):
    print("Feature selection by mean decrease impurity")
    data.drop(columns=['name'], inplace=True)
    y = data.loc[:, 'type'].values
    # y = y.astype(int)
    X = data.drop(columns=['type'])
    features = X.columns.to_list()
    X = X.values
    # X = X.astype(int)
    rf = RandomForestRegressor()
    rf.fit(X, y)
    df = pd.DataFrame(data={'features': features,
                      'ranking': rf.feature_importances_})
    df.sort_values(["ranking"], axis="rows", ascending=[False], inplace=True)
    top_features = df.features.to_list()
    return top_features


def by_info_gain(data):
    # the function a dataframe with the features and the labels, and the number of features to select
    # returns a list of features which are ordered by information gain score
    data.drop(columns=['name'], inplace=True)
    # caculating the entropy before the split
    number_of_samples = len(data.loc[:, 'type'])
    percentage_type_0 = len(data[data['type'] == 0])/number_of_samples
    percentage_type_1 = len(data[data['type'] == 1])/number_of_samples
    entropy_before_split = - \
        (percentage_type_0*(log2(percentage_type_0) if percentage_type_0 != 0 else -inf) +
         percentage_type_1*(log2(percentage_type_1) if percentage_type_1 != 0 else -inf))
    df = pd.DataFrame(columns=['feature', 'info_gain'])

    # caculating the entropy after the split, for every sample
    for i in data.columns:
        if (i != 'type'):

            data_ = data.loc[:, [i, 'type']]

            # getting the percentage of every type in the left node and in the right node
            left = data_.loc[data_.loc[:, i] == 0]
            len_left = len(left)
            left_percentage = len_left/number_of_samples

            right = data_.loc[data_.loc[:, i] == 1]
            len_right = len(right)
            right_percentage = len_right/number_of_samples

            if (len_left == 0):
                len_left = 1
            if (len_right == 0):
                len_right = 1

            # getting the percentage of every type in the children of the left node and the right node
            left_0_percentage = len(left[left.loc[:, 'type'] == 0])/len_left
            left_1_percentage = len(left[left.loc[:, 'type'] == 1])/len_left
            right_0_percentage = len(
                right[right.loc[:, 'type'] == 0])/len_right
            right_1_percentage = len(
                right[right.loc[:, 'type'] == 1])/len_right

            # calculating entropy at each side
            entropy_left = 0
            if ((left_0_percentage == 0) or (left_1_percentage == 0)):
                entropy_left = 0
            else:
                entropy_left = -(left_0_percentage*log2(left_0_percentage) +
                                 left_1_percentage*log2(left_1_percentage))
            entropy_right = 0
            if ((right_0_percentage == 0) or (right_1_percentage == 0)):
                entropy_right = 0
            else:
                entropy_right = -(right_0_percentage*log2(right_0_percentage) +
                                  right_1_percentage*log2(right_1_percentage))

            # calculating the entropy after split
            entropy_after_split = (
                left_percentage*entropy_left+right_percentage*entropy_right)

            # getting the information gain
            info_gain = entropy_before_split-entropy_after_split
            df2 = pd.DataFrame([[i, info_gain]], columns=[
                               'feature', 'info_gain'])
            df = pd.concat([df, df2])
    df.sort_values(by='info_gain', ascending=False, inplace=True)
    top_features = df['feature'].to_list()
    return top_features


if __name__ == "__main__":
    """    
    df=pd.DataFrame(columns=['type','perm_rate','a','b','c','d','e'])
    df.loc[0]=[0,0.6,1,1,1,1,0]
    df.loc[1]=[0,0.2,1,1,1,1,1]
    df.loc[2]=[0,0.3,0,1,1,1,0]
    df.loc[3]=[0,0.5,0,1,0,1,0]
    df.loc[4]=[0,0.2,0,1,0,1,0]
    df.loc[5]=[1,0.3,0,0,1,0,0]
    df.loc[6]=[1,0.1,0,0,0,1,1]
    df.loc[7]=[1,0.5,0,0,0,0,0]
    df.loc[8]=[1,0.1,0,0,1,0,1]
    df.loc[9]=[1,0.2,0,0,0,0,0]
    print(by_tf_idf(df))
    """
    df = pd.DataFrame(columns=['name', 'type', 'a', 'b', 'c', 'd', 'e'])
    df.loc[0] = ['a', 0, 1, 1, 1, 1, 0]
    df.loc[1] = ['b', 0, 1, 1, 1, 1, 1]
    df.loc[2] = ['c', 0, 0, 1, 1, 1, 0]
    df.loc[3] = ['d', 0, 0, 1, 0, 1, 0]
    df.loc[4] = ['e', 0, 0, 1, 0, 1, 0]
    df.loc[5] = ['f', 1, 0, 0, 1, 0, 0]
    df.loc[6] = ['g', 1, 0, 0, 0, 1, 1]
    df.loc[7] = ['h', 1, 0, 0, 0, 0, 0]
    df.loc[8] = ['i', 1, 0, 0, 1, 0, 1]
    df.loc[9] = ['j', 1, 0, 0, 0, 0, 0]
    print(by_info_gain(df))
