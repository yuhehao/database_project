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
# with open("data/data.pickle", "rb") as f:
#     data = pickle.load(f)

with open("data/aggdf.pickle", "rb") as f:
    aggdf = pickle.load(f)
print("ok!")

# Split training set and testing set
X = aggdf[0:len(train_data)]
Y = train_data["Correct First Attempt"]
print("Table X:", X.shape)
print("Table Y:", Y.shape)
X_ = aggdf[len(train_data):]
Y_ = test_data["Correct First Attempt"]
print("Table X_:", X_.shape)
print("Table Y_:", Y_.shape)


print("data has been ready!")

# ML
# logistic regression
# from sklearn.linear_model import LogisticRegression
# lr_model = LogisticRegression(solver="liblinear", n_jobs=-16, max_iter=2000) # Enable all CPUs
# lr_model.fit(X, Y)
# print(type(lr_model))

# # random forest
# from sklearn import ensemble
# est_count = 1000
# rf_model = ensemble.RandomForestClassifier(n_estimators = est_count, criterion="entropy", max_depth=30, n_jobs=-16) # Enable all CPUs
# rf_model.fit(X, Y)



# using pyspark
# from pyspark.ml.classification import LogisticRegression
# lr = LogisticRegression(labelCol="Correct First Attempt", maxIter=10).fit(X)
# lr_pred = lr.transform(Y)
# print(lr_pred.show())
# print("len:", len(lr_pred))
# from pyspark.sql import SparkSession
# from pyspark import SparkContext
# from pyspark.sql import SQLContext
# spark=SparkSession.builder.appName('data_processing').getOrCreate()
# sc = SparkContext()
# sqlContest = SQLContext(sc)
# df = sqlContest.createDataFrame(X)
# print(type(df))
# from pyspark.ml.classification import RandomForestClassifier,LogisticRegression, DecisionTreeClassifier
# rf = RandomForestClassifier(labelCol="Correct First Attempt", numTrees=50, maxDepth=6).fit(X)
# rf_pred = rf.transform(Y)
# print(rf_pred)



# MSE
# Root Mean Squared Error
# Here, we consider using numpy as a powerful
# utility to solve the RMSE
def RMSE(P, Y):
    P = P[~np.isnan(Y)]
    Y = Y[~np.isnan(Y)]
    return np.sqrt(np.sum(np.square(P - Y)) / len(Y))
# Logistic Classification

# P = rf.predict_proba(X)[:, 1]
# P_ = rf.predict_proba(X_)[:, 1]

# P = lr_model.predict_proba(X)[:, 1]
# P_ = lr_model.predict_proba(X_)[:, 1]

# print("logistic regression Train Error:", RMSE(P, Y))
# print("logistic regression Test Error:", RMSE(P_, Y_))

# # Generate submission
# RES = P_[np.isnan(Y_)]
# test_data.loc[np.isnan(test_data["Correct First Attempt"]), "Correct First Attempt"] = RES
# test_data.to_csv("data/predict_result.csv", sep='\t', index=False)

# # Save
# pd.Series(P).to_csv("data/sparse_train.csv", sep='\t', header=["sparse_res"])
# pd.Series(P_).to_csv("data/sparse_test.csv", sep='\t', header=["sparse_res"])

# Random Forest
#format: var,var : 
est_list = [i for i in range(1,201,10)]
m_list = [i for i in range(5,101,4)]
randomforest = []
# random forest
from sklearn import ensemble
def rf_learn(est_count,m):
    rf_model = ensemble.RandomForestClassifier(n_estimators = est_count, criterion="entropy", max_depth=m, n_jobs=-12) # Enable all CPUs
    rf_model.fit(X, Y)

    P = rf_model.predict_proba(X)[:, 1]
    P_ = rf_model.predict_proba(X_)[:, 1]

    Train_error = RMSE(P, Y)
    Test_error = RMSE(P_, Y_)
    print("Random Forest Train Error:", Train_error)
    print("Random Forest Test Error:", Test_error)
    return [Train_error,Test_error]
for i in est_list:
    for j in m_list:
        randomforest.append(rf_learn(i,j))
        
with open("data/rf_list.pickle", "wb") as f:
    pickle.dump(randomforest, f)
# # Generate submission
# RES = P_[np.isnan(Y_)]
# test_data.loc[np.isnan(test_data["Correct First Attempt"]), "Correct First Attempt"] = RES
# test_data.to_csv("data/predict_result.csv", sep='\t', index=False)
# # Save
# pd.Series(P).to_csv("data/sparse_train.csv", sep='\t', header=["sparse_res"])
# pd.Series(P_).to_csv("data/sparse_test.csv", sep='\t', header=["sparse_res"])
