# %matplotlib inline
from itertools import permutations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import pickle
from scipy import sparse


# load the data from a pickle file
with open("data/train_data.pickle", "rb") as f:
    train_data = pickle.load(f)
with open("data/test_data.pickle", "rb") as f:
    test_data = pickle.load(f)

# # Print the first 5 rows of the data
# print(train_data.head())

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

with open("data/data.pickle", "wb") as f:
    pickle.dump(data, f)

# print(data.head())

# one hot encode the data
def one_hot_encode(data, column_name,categories=None):
    one_hot = pd.get_dummies(data[column_name], prefix=categories, sparse=True)
    # data = data.drop(column_name, axis=1)
    # data = data.join(one_hot)
    return one_hot

# two dimensional array of the features
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
    for i in permutations(relation_list, l):
        total.append(i)
features = []
for i in total:
    features.append(use_one_hot_encoding_on_each_piar(data, i))
with open("data/features.pickle", "wb") as f:
    pickle.dump(features, f)
# https://blog.csdn.net/u014281392/article/details/89525026  about spark random forest
# # Problem Unit
# pu_features = one_hot_encode(data, "Problem Unit")
# # Problem Section
# ps_features = one_hot_encode(data, "Problem Section")
# # Problem Name
# pn_features = one_hot_encode(data, "Problem Name")
# # Step Name
# sn_features = one_hot_encode(data, "Step Name")
# # Student ID
# sid_features = one_hot_encode(data, "Anon Student Id")


# # two dimension of Student ID and Problem Unit
# cols = ["Anon Student Id", "Problem Unit"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# sid_pu_features = one_hot_encode(data, join_name(cols))
# # two dimension of Problem Unit and Problem Section
# cols = ["Problem Unit", "Problem Section"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# pu_ps_features = one_hot_encode(data, join_name(cols))
# # two dimension of Problem Section and Problem Name
# cols = ["Problem Section", "Problem Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# ps_pn_features = one_hot_encode(data, join_name(cols))
# # Problem Name and Step Name
# cols = ["Problem Name", "Step Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# pn_sn_features = one_hot_encode(data, join_name(cols))
# # Student ID and Problem Section
# cols = ["Anon Student Id", "Problem Section"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# sid_ps_features = one_hot_encode(data, join_name(cols))
# # Student ID and Problem Name
# cols = ["Anon Student Id", "Problem Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# sid_pn_features = one_hot_encode(data, join_name(cols))
# # Student ID and Step Name
# cols = ["Anon Student Id", "Step Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# sid_sn_features = one_hot_encode(data, join_name(cols))

# # high dimensional array of the features
# # Student ID and Problem Name and Section Name
# cols = ["Anon Student Id", "Problem Name", "Problem Section"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# sid_pn_ps_features = one_hot_encode(data, join_name(cols))
# # Problem Unit and Problem Section and Problem Name
# cols = ["Problem Unit", "Problem Section", "Problem Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# pu_ps_pn_features = one_hot_encode(data, join_name(cols))
# # Problem Section and Problem Name and Step Name
# cols = ["Problem Section", "Problem Name", "Step Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# ps_pn_sn_features = one_hot_encode(data, join_name(cols))
# # Student ID and Problem Section and Problem Name and Step Name
# cols = ["Anon Student Id", "Problem Unit", "Problem Section", "Problem Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# sid_pu_ps_pn_features = one_hot_encode(data, join_name(cols))
# # Problem Unit and Problem Section and Problem Name and Step Name
# cols = ["Problem Unit", "Problem Section", "Problem Name", "Step Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# pu_ps_pn_sn_features = one_hot_encode(data, join_name(cols))
# # Student ID and Problem Unit and Problem Section and Problem Name and Step Name
# cols = ["Anon Student Id", "Problem Unit", "Problem Section", "Problem Name", "Step Name"]
# data[join_name(cols)] = data.apply(join_columns, axis=1, args=(cols,))
# sid_pu_ps_pn_sn_features = one_hot_encode(data, join_name(cols))

# features = [pu_features, ps_features, sn_features, sid_features
#     , sid_pu_features, pu_ps_features, ps_pn_features, pn_sn_features
#     , sid_pn_ps_features, pu_ps_pn_features, ps_pn_sn_features, sid_pu_ps_pn_features
#     , pu_ps_pn_sn_features
#     , sid_ps_features, sid_pn_features
#     , sid_sn_features, sid_pu_ps_pn_sn_features]

# dict_features = {"pu_features": pu_features, "ps_features": ps_features, "sn_features": sn_features, "sid_features": sid_features, "sid_pu_features": sid_pu_features, "pu_ps_features": pu_ps_features, "ps_pn_features": ps_pn_features, "pn_sn_features": pn_sn_features, "sid_pn_ps_features": sid_pn_ps_features, "pu_ps_pn_features": pu_ps_pn_features, "ps_pn_sn_features": ps_pn_sn_features, "sid_pu_ps_pn_features": sid_pu_ps_pn_features, "pu_ps_pn_sn_features": pu_ps_pn_sn_features, "sid_ps_features": sid_ps_features, "sid_pn_features": sid_pn_features, "sid_sn_features": sid_sn_features, "sid_pu_ps_pn_sn_features": sid_pu_ps_pn_sn_features}
# for feature in dict_features:
#     filename = "features/" + feature + ".pickle"
#     print("Saving " + filename)
#     with open(filename, "wb") as f:
#         pickle.dump(dict_features[feature], f)
