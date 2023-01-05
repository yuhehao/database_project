import pandas as pd
import pickle

import pyspark

# Read the data
train_file_path = "data/train.csv"
test_file_path = "data/test.csv"
# train_data = pd.read_csv(train_file_path, sep="\t")
# test_data = pd.read_csv(test_file_path, sep="\t")

train_df_spark =  spark.read.csv("train_file_path")
test_df_spark = spark.read.csv("test_file_path")
# write the data to a pickle file
# with open("data/train_data.pickle", "wb") as f:
#     pickle.dump(train_data, f)
# with open("data/test_data.pickle", "wb") as f:
#     pickle.dump(test_data, f)