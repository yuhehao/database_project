import pandas as pd
import pickle

import pyspark
from pyspark.sql import SparkSession

# Read the data
train_file_path = "data/train.csv"
test_file_path = "data/test.csv"
train_data = pd.read_csv(train_file_path, sep="\t")
test_data = pd.read_csv(test_file_path, sep="\t")
# write the data to a pickle file
with open("data/train_data.pickle", "wb") as f:
    pickle.dump(train_data, f)
with open("data/test_data.pickle", "wb") as f:
    pickle.dump(test_data, f)

# if u want to use spark to load this data
# use following code replace the above code 
# spark = SparkSession.builder.appName('Read CSV File into DataFrame').getOrCreate()

# _test_data_spark = spark.read.csv('data/test.csv', sep='\t')
# _test_data_spark = _test_data_spark.withColumnRenamed("_c0", "Row")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c1", "Anon Student Id")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c2", "Problem Hierarchy")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c3", "Problem Name")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c4", "Problem View")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c5", "Step Name")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c6", "Step Start Time")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c7", "First Transaction Time")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c8", "Correct Transaction Time")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c9", "Step End Time")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c10", "Step Duration (sec)")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c11", "Correct Step Duration (sec)")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c12", "Error Step Duration (sec)")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c13", "Correct First Attempt")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c14", "Incorrects")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c15", "Hints")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c16", "Corrects")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c17", "KC(Default)")
# _test_data_spark = _test_data_spark.withColumnRenamed("_c18", "Opportunity(Default)")

# _train_data_spark = spark.read.csv('data/train.csv', sep='\t')
# _train_data_spark = _train_data_spark.withColumnRenamed("_c0", "Row")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c1", "Anon Student Id")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c2", "Problem Hierarchy")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c3", "Problem Name")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c4", "Problem View")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c5", "Step Name")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c6", "Step Start Time")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c7", "First Transaction Time")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c8", "Correct Transaction Time")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c9", "Step End Time")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c10", "Step Duration (sec)")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c11", "Correct Step Duration (sec)")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c12", "Error Step Duration (sec)")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c13", "Correct First Attempt")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c14", "Incorrects")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c15", "Hints")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c16", "Corrects")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c17", "KC(Default)")
# _train_data_spark = _train_data_spark.withColumnRenamed("_c18", "Opportunity(Default)")

# train_data = _train_data_spark.toPandas()
# train_data = train_data.drop(index=[0])
# test_data = _test_data_spark.toPandas()
# test_data = test_data.drop(index=[0])

# # write the data to a pickle file
# with open("data/train_data.pickle", "wb") as f:
#     pickle.dump(train_data, f)
# with open("data/test_data.pickle", "wb") as f:
#     pickle.dump(test_data, f)
