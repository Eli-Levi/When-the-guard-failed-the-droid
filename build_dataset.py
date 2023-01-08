from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from androguard.misc import AnalyzeAPK


def dividedataset():
    print("Dividing dataset into 80 20")
    df = pd.read_csv('data.csv')
    df = df[df.category != 2]
    df = df.drop_duplicates(subset=['name'])
    df_b = df[df.type == 0]
    df_m = df[df.type == 1]
    df_b.group_num = 0
    df_m.group_num = 0
    df_m = df_m.sample(frac = 0.2)
    df_m.category = 0
    df_b.category = 0
    print(len(df_b))
    print(len(df_m))
    # split the benign into train and test
    train_b = df_b.sample(frac = 0.6)
    train_b.category = 0
    test_b = df_b.drop(train_b.index)
    test_b.category = 1

    shuffled_train_b = train_b.sample(frac=1)
    result_train_b = np.array_split(shuffled_train_b, 5)
    shuffled_test_b = test_b.sample(frac=1)
    result_test_b = np.array_split(shuffled_test_b, 5)
    # split the malicious into train and test
    train_m = df_m.sample(frac = 0.6)
    train_m.category = 0
    test_m = df_m.drop(train_m.index)
    test_m.category = 1

    shuffled_train_m = train_m.sample(frac=1)
    result_train_m = np.array_split(shuffled_train_m, 5)
    shuffled_test_m = test_m.sample(frac=1)
    result_test_m = np.array_split(shuffled_test_m, 5)
    for i in range(5):
        result_train_b[i].group_num = i
        result_test_b[i].group_num = i
        result_train_m[i].group_num = i
        result_test_m[i].group_num = i
        if i == 0:
            train_b = result_train_b[i]
            test_b = result_test_b[i]
            train_m = result_train_m[i]
            test_m = result_test_m[i]
        else:
            train_b = train_b.combine_first(result_train_b[i])
            test_b = test_b.combine_first(result_test_b[i])
            train_m = train_m.combine_first(result_train_m[i])
            test_m = test_m.combine_first(result_test_m[i])
    
    df_dataframe = train_b.combine_first(train_m.combine_first(test_b.combine_first(test_m)))
    df_dataframe.to_csv("dataset.csv")
    return df


# df1 = get_permissions("train_bingn", 0, 0, "train_bingn")
# df2 = get_permissions("train_malware", 1, 0, "train_malware")
# df3 = get_permissions("test_bingn", 0, 1, "test_bingn")
# df4 = get_permissions("test_malware", 1, 1, "test_malware")
# df5 = get_permissions("test_manipulated", 1, 2, "test_manipulated")
# df1 = df1.combine_first(df2.combine_first(df3.combine_first(df4.combine_first(df5))))
# df1 = df1.fillna(0)
# df1.index.name = "name"
# df1.to_csv("all.csv")
dividedataset()
