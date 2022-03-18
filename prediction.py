import random
import numpy as np
import statistics
import pandas as pd
from sklearn.preprossing import LabelEncoder

# read the dataset

df = pd.read_csv("train.csv")

df = df.fillna(df.mode().iloc[0])

print(df)


