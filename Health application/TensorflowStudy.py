#preprocessing
import pandas as pd

raw_dataset = pd.read_csv("Dermatology.csv", na_values = '?')

#if there is some NaN value, removes the row
dataset = raw_dataset.dropna()

#implementation
import tensorflow as tf

train_dataset = dataset.sample(frac=0.75,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('erythema')
test_labels = test_dataset.pop('erythema')

