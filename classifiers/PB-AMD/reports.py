import algorithms
import concurrent.futures
import math


def random_forest(num_of_features, features, group, group_mani, report, X_train, X_test, X_test_manipulated, y_train, y_test, y_test_manipulated):

    dict1 = {'num of trees': 0, 'criterion': 0,
             'min_samples_split': 0, 'recall': 0, 'confusion matrix': 0}
    dict2 = {'recall': 0, 'confusion matrix': 0}
    # Number of trees in random forest
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(algorithms.random_forest_, X_train, y_train, X_test, y_test, X_test_manipulated, n, criteria,  min_samples_split, features)
                   for n in range(10, 131, 10) for criteria in ['entropy', 'gini'] for min_samples_split in [3]]

    max1, max2 = 0, 0
    for f_ in concurrent.futures.as_completed(results):
        result = f_.result()
        get_perfomances(result[6], result[0], dict1)
        get_perfomances(result[6], result[1], dict2)
        dict1['num of trees'] = (result[2])
        dict1['criterion'] = (result[3])
        dict1['min_samples_split'] = (result[4])
        x = [group, group_mani, ' ', num_of_features, features, dict1['num of trees'], dict1['criterion'], dict1['min_samples_split'], ' ',
             dict1['recall'], dict1['confusion matrix'], ' ', dict2['recall'], dict2['confusion matrix'], ' ', X_test_manipulated.shape[0]]
        r1, r2 = dict1['recall'], dict2['recall']
        if r1 > max1:
            max1 = r1
            x1 = x
        if r2 > max2:
            max2 = r2
            x2 = x

    if max1 != 0:
        write_to_csv(report, x1)
    if max2 != 0:
        write_to_csv(report, x2)


def old_report_rf(num_of_features, features, group, group_mani, report, X_train, X_test, X_test_manipulated, y_train, y_test, y_test_manipulated):
    print("Old rf")
    dict1 = {'num of trees': 0, 'criterion': 0,
             'min_samples_split': 0, 'recall': 0, 'confusion matrix': 0}
    dict2 = {'recall': 0, 'confusion matrix': 0}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = [executor.submit(algorithms.random_forest_, X_train, y_train, X_test, y_test, X_test_manipulated, n, criteria,  min_samples_split, features) for n in range(
            20, 100, 20) for criteria in ['entropy', 'gini'] for min_samples_split in [3, 11, 21, 51, 101, 201, 401, 601, 801]]

    max1, max2 = 0, 0

    for f_ in concurrent.futures.as_completed(results):
        result = f_.result()
        get_perfomances(y_test, result[0], dict1)
        get_perfomances(y_test_manipulated, result[1], dict2)
        dict1['num of trees'] = (result[2])
        dict1['criterion'] = (result[3])
        dict1['min_samples_split'] = (result[4])
        x = [group, group_mani, ' ', num_of_features, features, dict1['num of trees'], dict1['criterion'], dict1['min_samples_split'], ' ',
             dict1['recall'], dict1['confusion matrix'], ' ', dict2['recall'], dict2['confusion matrix'], ' ', X_test_manipulated.shape[0]]
        r1, r2 = dict1['recall'], dict2['recall']
        if r1 > max1:
            max1 = r1
            x1 = x
        if r2 > max2:
            max2 = r2
            x2 = x

    if max1 != 0:
        write_to_csv(report, x1)
    if max2 != 0:
        write_to_csv(report, x2)


def get_perfomances(y, y_pred, dict):

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
    import numpy

    recall = recall_score(y, y_pred, zero_division=1)

    dict['confusion matrix'] = (numpy.array_str(confusion_matrix(y, y_pred)))
    dict['recall'] = (round(recall_score(y, y_pred, zero_division=1), 2))


def write_to_csv(file_name, list_):
    import csv
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(list_)
