# %matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import pickle
from scipy import sparse

with open("data/train_data.pickle", "rb") as f:
    train_data = pickle.load(f)
with open("data/test_data.pickle", "rb") as f:
    test_data = pickle.load(f)
with open("data/data.pickle", "rb") as f:
    data = pickle.load(f)
# KC##**##
def get_kc_dummies(row, headers, col):
    tokens = row[col].split("~~")
    opps = np.asarray([int(s) if s.lower() != "nan" else 0 for s in row["Opportunity(Default)"].split("~~")])
    opps = np.log(opps + 1) # log (1 + x)
    sr = pd.Series(np.zeros(len(headers)), index = headers)
    sr[tokens] = opps
    # return sr.to_frame().to_sparse(fill_value=0)
    return sr.astype(pd.SparseDtype(int, fill_value=0))
compound_KCs = set(data["KC(Default)"])
print("KC(Default) has %d compound values." % len(compound_KCs))
KCs = set()
for s in compound_KCs:
    KCs = KCs.union(set(s.split("~~")))

# print("There are %d atomic KCs, so we will be adding as many columns to this dataframe" % len(KCs))

# Cast and split columns
kc_features = data.apply(get_kc_dummies, axis=1, result_type="expand", args=(KCs, "KC(Default)"))

# Numerical features
# Problem View
def norm_problem_view(row):
    return math.log(int(row["Problem View"]) + 1)

pv_features = data.apply(norm_problem_view, axis=1)
# Opportunity
def norm_opportunity(row):
    opps = np.asarray([int(s) if s.lower() != "nan" else 0 for s in row["Opportunity(Default)"].split("~~")])
    return math.log(np.min(opps) + 1)

opp_features = data.apply(norm_opportunity, axis=1)

numerical_features = pd.concat((pv_features, opp_features), axis=1)
features = []

# features_name_list = [ "pu_features", "ps_features", "sn_features", "sid_features", "sid_pu_features", "pu_ps_features", "ps_pn_features", "pn_sn_features", "sid_pn_ps_features", "pu_ps_pn_features", "ps_pn_sn_features", "sid_pu_ps_pn_features", "pu_ps_pn_sn_features", "sid_ps_features", "sid_pn_features", "sid_sn_features", "sid_pu_ps_pn_sn_features"]
# for feature_name in features_name_list:
#     filename = "features/" + feature_name + ".pickle"
with open("features/features.pickle", "rb") as f:
    features = pickle.load(f)
print("ok")
features.append(kc_features)
features.append(numerical_features)

aggdf = sparse.hstack([f.astype(pd.SparseDtype(int, fill_value=0)).sparse.to_coo().tocsr() for f in features], format="csr", dtype=float)
with open("data/aggdf.pickle", "wb") as f:
        pickle.dump(aggdf, f)