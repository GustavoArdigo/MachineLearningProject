import pandas as pd
import __future__ #for future features in newer versions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# load database
test = pd.read_csv("dadosFiltrado.csv")

# choose train and test dataset
train_dataset = test.sample(frac=0.75,random_state=1) #the random_state gives the seed for the randomization
test_dataset = test.drop(train_dataset.index)

# answers for train and test dataset
train_labels = train_dataset.pop('label')
test_labels = test_dataset.pop('label')

def build_model():
    model = keras.Sequential([
      layers.Input(len(train_dataset.keys())), #input_shape = 3
      layers.Dense(2, activation = 'relu'),
      layers.Dense(1, activation='sigmoid'),
  ])
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

model = build_model()
print(model.summary())

predicted_labels_ANN = model.fit(train_dataset, train_labels, epochs=4)
print(predicted_labels_ANN)

test_loss, test_acc = model.evaluate(test_dataset,  test_labels)

print('\nTest accuracy:', test_acc)


