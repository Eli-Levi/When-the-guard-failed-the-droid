from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import os
from pathlib import Path
import re
import pandas as pd
import numpy as np


def get_permissions():
    os.chdir("niv_avi_files/samples")
    print("running apktool")
    perm_list = []
    name_list = []
    files = []
    for apk in Path('.').glob("*.apk"):
        print("working on", apk)
        files.append(str(apk)[0:len(str(apk))-4])
        os.system("apktool d -f "+str(apk))
        name_list.append(str(apk))

    print(files)
    new_f = open("perm.txt", "a")
    df = pd.DataFrame()
    first = 0
    for fileManifest in files:
        print("working of manifest for "+fileManifest)
        f = open(fileManifest+"/AndroidManifest.xml", "r")
        count = 0
        perm_list = []
        for line in f:
            x = re.search('<uses-permission android:name="*"', line)
            if x:
                count = count + 1
                apk_perms = pd.Series(line)
                apk_perms = apk_perms.apply(lambda perm: "perm:" + perm[19:])
                perm_list.append(' '.join(list(apk_perms)))

        list_of_names = [fileManifest+".apk"]*count
        tfidf = CountVectorizer(lowercase=False)
        temp_list = tfidf.fit_transform(perm_list)
        dfapp = pd.DataFrame.sparse.from_spmatrix(
            temp_list, index=list_of_names, columns="perm: " + tfidf.get_feature_names_out())
        dfapp.drop("perm: perm", axis=1, inplace=True)
        dfapp.drop("perm: name", axis=1, inplace=True)
        dfapp.drop("perm: android", axis=1, inplace=True)
        # dfapp.to_csv(str(fileManifest)+"our_data.csv")
        dfapp = dfapp.fillna(0)
        if first == 0:
            first = 1
            df = dfapp
        else:
            df = df.combine_first(dfapp)
            df = df.fillna(0)
    df = df.assign(type=None)
    df = df.assign(group_num=None)
    df = df.assign(group_mani=None)
    df = df.assign(category=None)
    df = df.assign(perm_rate=None)
    df = df.assign(call__=None)
    df = df.fillna(0)
    df.to_csv("data.csv")


get_permissions()

