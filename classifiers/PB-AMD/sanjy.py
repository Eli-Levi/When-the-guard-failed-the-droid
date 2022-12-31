
import reports
import machine
import algorithms
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split


def run(data, reports_):

    # setting the paths of the reports and their titles
    reports_path = [reports_ + 'random_forest.csv',
                    reports_ + 'cart.csv', reports_ + 'c45.csv']

    reports_items = [['group', 'attack', ' ', 'num_of_features', 'features', 'num_of_trees', 'criterion', 'min_samples_split', '', 'test_malicious_recall', 'test_malicious_confusion_matrix', ' ', 'test_manipulated_recall', 'test_manipulated_confusion_matrix', ' ', 'num of observations'],
                     ['group', 'attack', ' ', 'num_of_features', 'features', 'criterion', 'min_samples_split', '', 'test_malicious_recall',
                         'test_malicious_confusion_matrix', ' ', 'test_manipulated_recall', 'test_manipulated_confusion_matrix', ' ', 'num of observations'],
                     ['group', 'attack', ' ', 'num_of_features', 'features', ' ', 'test_malicious_recall', 'test_malicious_confusion_matrix', ' ', 'test_manipulated_recall', 'test_manipulated_confusion_matrix', ' ', 'num of observations']]

    reports_ = tuple(zip(reports_path, reports_items))

    machine.build_reports(reports_)

    # getting the data needed for the machine to work
    data = pd.read_csv(data)

    data = machine.get_data(data, ['perm:'])

    # separating the data

    train, test, mani = machine.divide_data(data)
    
    # running the machine
    features = []
    import feature_selection
    for i in range(5):
        features.append(feature_selection.by_mean_decrease_impurity(train[i]))

    for group in range(5):
        mani_group = 0
        m, t = machine.keep_same_apks(mani[group][mani_group], test[group])
        mani[group][mani_group].drop(columns=['name'])
        n_of_f = 0
        if group == 0 :
            n_of_f = 200
        elif group == 1:
            n_of_f = 160
        elif group == 2:
            n_of_f = 110
        elif group == 3:
            n_of_f = 200
        elif group == 4:
            n_of_f = 200

        for num_of_features in [n_of_f]:
            X_train, X_test, X_mani, y_train, y_test, y_mani, top_features = machine.get_X_y_features(
                num_of_features, features, group, t, train, m)
            print('Running random forest...')
            reports.random_forest(num_of_features, top_features, str(group), str(
                mani_group+1), reports_[0][0], X_train, X_test, X_mani, y_train, y_test, y_mani)
            

if __name__ == "__main__":

    # getting paths
    data = "data.csv"
    reports_ = ""  # "reports\\"

    # running the machine
    run(data, reports_)
