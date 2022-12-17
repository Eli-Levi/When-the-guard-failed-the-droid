from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np
from androguard.misc import AnalyzeAPK


def get_permissions(filename, type, category, dir):
    print("running androguard")
    perm_list = []
    files = []
    df = pd.DataFrame()
    first = 0
    for apk in Path('./'+dir).glob("*.apk"):
        print("working on "+str(apk))
        a, d, dx = AnalyzeAPK(str(apk))
        f = a.get_permissions()
        count = 0
        perm_list = []
        for line in f:
            count = count + 1
            apk_perms = pd.Series(line)
            apk_perms = apk_perms.apply(lambda perm: "perm:" + perm[19:])
            perm_list.append(' '.join(list(apk_perms)))

        list_of_names = [apk]*count
        tfidf = CountVectorizer(stop_words=None, lowercase=False)
        print(perm_list)
        if len(perm_list) > 0:
            temp_list = tfidf.fit_transform(perm_list)
            dfapp = pd.DataFrame.sparse.from_spmatrix(
                temp_list, index=list_of_names, columns="perm: " + tfidf.get_feature_names_out())
            dfapp.drop("perm: perm", axis=1, inplace=True)
            dfapp = dfapp.fillna(0)
            if first == 0:
                first = 1
                df = dfapp
            else:
                df = df.combine_first(dfapp)
                df = df.fillna(0)
    df = df.assign(type=type)
    df = df.assign(group_num=None)
    df = df.assign(group_mani=None)
    df = df.assign(category=category)
    df = df.assign(perm_rate=None)
    df = df.assign(call__=None)
    df = df.fillna(0)
    return df


os.chdir("apks")
df1 = get_permissions("train_bingn", 0, 0, "train_bingn")
df2 = get_permissions("train_malware", 1, 0, "train_malware")
df3 = get_permissions("test_bingn", 0, 1, "test_bingn")
df4 = get_permissions("test_malware", 1, 1, "test_malware")
df5 = get_permissions("test_manipulated", 1, 2, "test_manipulated")
df1 = df1.combine_first(df2.combine_first(df3.combine_first(df4.combine_first(df5))))
df1 = df1.fillna(0)
df1.index.name = "name"
df1.to_csv("all.csv")
