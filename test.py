import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
import pickle
from scipy import sparse


# load the data from a pickle file
with open("data/aggdf.pickle", "rb") as f:
    df = pickle.load(f)
with open("data/train_data.pickle", "rb") as f:
    train_data = pickle.load(f)
with open("data/test_data.pickle", "rb") as f:
    test_data = pickle.load(f)
X = df[0:len(train_data)]
Y = train_data["Correct First Attempt"]
print("Table X:", X.shape)
print("Table Y:", Y)