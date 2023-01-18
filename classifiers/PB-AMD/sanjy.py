
import reports
import machine
import algorithms
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV


def run(data, reports_):

    # setting the paths of the reports and their titles
    reports_path = [reports_ + 'random_forest.csv',
                    reports_ + 'cart.csv', reports_ + 'c45.csv']
    reports_items = [['group', ' ', 'num_of_features', 'features', 'num_of_trees', 'criterion', 'min_samples_split', '',
                      'test_malicious_recall', 'test_malicious_confusion_matrix', ' ', 'tpr'],
                     ['group', 'attack', ' ', 'num_of_features', 'features', 'criterion', 'min_samples_split', '', 'test_malicious_recall',
                         'test_malicious_confusion_matrix', ' ', 'test_manipulated_recall', 'test_manipulated_confusion_matrix', ' ', 'num of observations'],
                     ['group', 'attack', ' ', 'num_of_features', 'features', ' ', 'test_malicious_recall', 'test_malicious_confusion_matrix', ' ', 'test_manipulated_recall', 'test_manipulated_confusion_matrix', ' ', 'num of observations']]

    reports_ = tuple(zip(reports_path, reports_items))
    machine.build_reports(reports_)

    # getting the data needed for the machine to work
    data = pd.read_csv(data)
    data = machine.get_data(data, ['perm:'])

    # separating the data

    train, test = machine.divide_data(data)

    # running the machine
    features = []
    import feature_selection
    for i in range(5):
        features.append(feature_selection.model_based_selection(train[i]))

    for group in range(5):
        mani_group = 0
        t = machine.keep_same_apks(test[group])
        for num_of_features in [110, 160]:
            X_train, X_test, y_train, y_test, top_features = machine.get_X_y_features(
                num_of_features, features, group, t, train)
            print('Running random forest...')
            reports.random_forest(num_of_features, top_features, str(
                group), reports_[0][0], X_train, X_test, y_train, y_test)


if __name__ == "__main__":

    # getting paths
    data = "dataset.csv"
    reports_ = ""  # "reports\\"

    # running the machine
    run(data, reports_)
