import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline



def mlp_classifier(X_train, y_train, X_test, X_test_manipulated, num, criterion, min_samples_split):
    # clf = MLPClassifier(solver='adam', alpha=1e-5,
    #                     hidden_layer_sizes=(15, ), random_state=1, max_iter=301, warm_start=True)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    return (clf.predict(X_test), clf.predict(X_test_manipulated), criterion, min_samples_split)


def gradient_boosting_trees(X_train, y_train, X_test, y_test, X_test_manipulated, num, criterion, min_samples_split, features):
    clf = GradientBoostingClassifier(
        n_estimators=num, criterion=criterion, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)
    # pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=5))])
    # clf.fit(pipeline.fit_transform(X_train), pipeline.fit_transform(y_train))
    return (clf.predict(X_test), clf.predict(X_test_manipulated), num, criterion, min_samples_split)


def pca_(X, n):
    pca = PCA(n_components=n)
    pca.fit(X)
    return pca.fit_transform(X)


def kmeans_(X, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters)
    return kmeans.fit_predict(X)

def random_forest_(X_train, y_train, X_test, y_test, X_test_manipulated, num, criterion, min_samples_split, features):
    clf = RandomForestClassifier(n_estimators=num,
                                 criterion=criterion, min_samples_split=min_samples_split)
    # clf = SVC(kernel="linear")
    # clf = make_pipeline(StandardScaler(), LinearSVC(random_state=0, tol=1e-5, dual = True))
    clf.fit(X_train, y_train)
    return (clf.predict(X_test), clf.predict(X_test), num, criterion, min_samples_split)



def decision_tree_(X_train, y_train, X_test, X_test_manipulated, criterion, min_samples_split):
    clf = DecisionTreeClassifier(
        criterion=criterion, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)

    return (clf.predict(X_test), clf.predict(X_test_manipulated), criterion, min_samples_split)


def kirin_(X):
    import pandas as pd
    groups = [['perm: SET_DEBUG_APP'],
              ['perm: READ_PHONE_STATE', 'perm: RECORD_AUDIO', 'perm: INTERNET'],
              ['perm: PROCESS_OUTGOING_CALLS',
                  'perm: RECORD_AUDIO', 'perm: INTERNET'],
              ['perm: ACCESS_FINE_LOCATION', 'perm: INTERNET',
               'perm: RECEIVE_BOOT_COMPLETED'],
              ['perm: ACCESS_COARSE_LOCATION', 'perm: INTERNET',
               'perm: RECEIVE_BOOT_COMPLETED'],
              ['perm: RECEIVE_SMS', 'perm: WRITE_SMS'],
              ['perm: SEND_SMS', 'perm: WRITE_SMS'],
              ['perm: UNINSTALL_SHORTCUT', 'perm: INSTALL_SHORTCUT'],
              ['perm: SET_PREFERRED_APPLICATIONS', 'call__']]

    y_pred = [0]*X.shape[0]
    for i in range(X.shape[0]):
        row = X.iloc[i, :]
        for group in groups:
            if all(elem in X.columns.to_list() for elem in group) == True:
                r = row[group]
                if sum(r) == len(group):
                    y_pred[i] = 1

    y_pred = pd.Series(y_pred)
    return y_pred


def dl_(X_train, y_train, X_test, X_test_manipulated, layer_1_nodes, layer_2_nodes, input_nodes):
    import keras
    from keras.models import Sequential
    from keras.layers import Dense

    # Initialising the ANN
    classifier = Sequential()

    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units=layer_1_nodes, kernel_initializer='uniform',
                   activation='relu', input_dim=input_nodes))

    # Adding the second hidden layer
    classifier.add(Dense(units=layer_2_nodes,
                   kernel_initializer='uniform', activation='relu'))

    # Adding the output layer
    classifier.add(
        Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    # Compiling the ANN
    classifier.compile(
        optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size=40, epochs=100)

    # Part 3 - Making the predictions and evaluating the model

    # Predicting the Test set results
    y_pred_test = classifier.predict(X_test)
    y_pred_test = (y_pred_test > 0.5)

    y_pred_test_manipulated = classifier.predict(X_test_manipulated)
    y_pred_test_manipulated = (y_pred_test_manipulated > 0.5)

    return y_pred_test, y_pred_test_manipulated
