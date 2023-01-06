from itertools import combinations, permutations
from multiprocessing import Process
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import pickle
from scipy import sparse
import threading

# load the data from a pickle file
with open("../data/train_data.pickle", "rb") as f:
    train_data = pickle.load(f)
with open("../data/test_data.pickle", "rb") as f:
    test_data = pickle.load(f)

# combine the data from train and test
# ensure one hot encoding from a universal set of values
data = pd.concat([train_data, test_data], axis=0, sort=False)

# cast the data to the correct type
data['KC(Default)'] = data['KC(Default)'].astype(str)
data['Opportunity(Default)'] = data['Opportunity(Default)'].astype(str)

# split unit and section out

# split by comma
data.insert(3, "Problem Unit", data.apply(lambda row: row["Problem Hierarchy"].split(',')[0].strip(), axis=1))
data.insert(4, "Problem Section", data.apply(lambda row: row["Problem Hierarchy"].split(',')[1].strip(), axis=1))

with open("../data/data.pickle", "wb") as f:
    pickle.dump(data, f)

# one hot encode the data
def one_hot_encode(data, column_name,categories=None):
    one_hot = pd.get_dummies(data[column_name], prefix=categories, sparse=True)
    # data = data.drop(column_name, axis=1)
    # data = data.join(one_hot)
    return one_hot

# two dimensional array of the features to prepare numerically represent the data
def join_columns(data, columns):
    return ",".join([str(data[column]) for column in columns])
def join_name(columns):
    return ",".join(columns)

def use_one_hot_encoding_on_each_piar(data, cols):
    data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
    return one_hot_encode(data, join_name(cols))

relation_list = ["Problem Unit", "Problem Section", "Problem Name", "Step Name", "Anon Student Id"]
total = []
for l in range(1,1+len(relation_list)):
    for i in combinations(relation_list, l):
        total.append(i)
        # total.append(list(combinations(relation_list, l)))
# print("ok")
features = []
for i in total:
    # print("o")
    features.append(use_one_hot_encoding_on_each_piar(data, i))
    # print("k")

with open("../features/features.pickle", "wb") as f:
    pickle.dump(features, f)
