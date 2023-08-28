import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

def make_model(input_shape, x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        layers.Dense(64, activation='relu', input_shape=[input_shape]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=0, callbacks=[early_stop])
    hist = pd.DataFrame(history.history)
    print(hist.tail())

train_data = pd.read_csv('./home_data/train.csv')
test_data = pd.read_csv('./home_data/test.csv')

train_data = train_data.drop(['Id'], axis=1)
test_data = test_data.drop(['Id'], axis=1)
numeric_cols = train_data.select_dtypes(include=[np.number]).columns
train_data = train_data[numeric_cols]
train_data = train_data.astype('float32')
train_data = train_data.fillna(train_data.mean())

x_train, x_test, y_train, y_test = train_test_split(train_data, train_data['SalePrice'], train_size=.8, test_size=0.2, random_state=0)
make_model(len(numeric_cols), x_train, y_train, x_test, y_test)
print(train_data.head(), x_train.head(), y_train.head())