import pandas as pd
import pickle

import pyspark
from pyspark.sql import SparkSession

# # Read the data
# train_file_path = "data/train.csv"
# test_file_path = "data/test.csv"
# train_data = pd.read_csv(train_file_path, sep="\t")
# test_data = pd.read_csv(test_file_path, sep="\t")
# # write the data to a pickle file
# with open("data/train_data.pickle", "wb") as f:
#     pickle.dump(train_data, f)
# with open("data/test_data.pickle", "wb") as f:
#     pickle.dump(test_data, f)

# using spark
spark = SparkSession.builder.enableHiveSupport().getOrCreate()

train_data = spark.read.options(header='true', inferSchema='true', delimiter='\t').csv('data/train.csv')
train_data = train_data.select("Problem View","Problem Name", "Problem Hierarchy", "Step Name", "Anon Student Id","KC(Default)", "Opportunity(Default)")

train_data = train_data.toPandas()

test_data = spark.read.options(header='true', inferSchema='true', delimiter='\t').csv('data/test.csv')
test_data = test_data.select("Problem View","Problem Name", "Problem Hierarchy", "Step Name", "Anon Student Id","KC(Default)", "Opportunity(Default)")

test_data = test_data.toPandas()

# write the data to a pickle file
with open("data/train_data.pickle", "wb") as f:
    pickle.dump(train_data, f)
with open("data/test_data.pickle", "wb") as f:
    pickle.dump(test_data, f)