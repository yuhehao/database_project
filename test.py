import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import pickle
from scipy import sparse


# # load the data from a pickle file
# with open("data/aggdf.pickle", "rb") as f:
#     df = pickle.load(f)
# with open("data/train_data.pickle", "rb") as f:
#     train_data = pickle.load(f)
# with open("data/test_data.pickle", "rb") as f:
#     test_data = pickle.load(f)
# X = df[0:len(train_data)]
# Y = train_data["Correct First Attempt"]
# print("Table X:", X.shape)
# print("Table Y:", Y)

from pyspark.sql import SparkSession
 
spark = SparkSession.builder.appName(
    'Read CSV File into DataFrame').getOrCreate()
 
df = spark.read.csv('data/test.csv', sep='\t')
df = df.withColumnRenamed("_c0", "Row")
df = df.withColumnRenamed("_c1", "Anon Student Id")
df = df.withColumnRenamed("_c2", "Problem Hierarchy")
df = df.withColumnRenamed("_c3", "Problem Name")
df = df.withColumnRenamed("_c4", "Problem View")
df = df.withColumnRenamed("_c5", "Step Name")
df = df.withColumnRenamed("_c6", "Step Start Time")
df = df.withColumnRenamed("_c7", "First Transaction Time")
df = df.withColumnRenamed("_c8", "Correct Transaction Time")
df = df.withColumnRenamed("_c9", "Step End Time")
df = df.withColumnRenamed("_c10", "Step Duration (sec)")
df = df.withColumnRenamed("_c11", "Correct Step Duration (sec)")
df = df.withColumnRenamed("_c12", "Error Step Duration (sec)")
df = df.withColumnRenamed("_c13", "Correct First Attempt")
df = df.withColumnRenamed("_c14", "Incorrects")
df = df.withColumnRenamed("_c15", "Hints")
df = df.withColumnRenamed("_c16", "Corrects")
df = df.withColumnRenamed("_c17", "KC(Default)")
df = df.withColumnRenamed("_c18", "Opportunity(Default)")

a = df.toPandas()
a = a.drop(index=[0])
print(a.head())

test_data = pd.read_csv('data/test.csv', sep="\t")
print(test_data.head())
