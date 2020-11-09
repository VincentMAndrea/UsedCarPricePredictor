import numpy as np
import keras
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

df = pd.read_csv('vehicles.csv', usecols=['Year', 'Make', 'Model', 'Condition', 'Title', 'Transmission', 'Odometer',
                                          'Price'])
print(df.shape)
print(df.columns)

print(df.nunique(axis=0))
print(df.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))))

top_200 = [x for x in df.Model.value_counts().sort_values(ascending=False).head(200).index]

df_200 = df[df.Model.isin(top_200)]

print(df_200.nunique(axis=0))
print(df_200.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))))

df_final = pd.get_dummies(df_200, drop_first=True)
print(df_final.shape)

test_data = pd.DataFrame(data=[[2020, 'buick', 'rendezvous', 'fair', 'salvage', 'automatic', 79778]])

test_data = pd.get_dummies(test_data)

missing_cols = set(df_final.columns) - set(test_data.columns)
for c in missing_cols:
    test_data[c] = 0

test_data = test_data[df_final.columns]
test_data = test_data.loc[:, test_data.columns != 'Price']

X = df_final.loc[:, df_final.columns != 'Price']
y = df_final['Price']
X = StandardScaler().fit_transform(X)

model = keras.Sequential()
model.add(keras.layers.Dense(7, activation='relu', input_shape=(243, )))
model.add(keras.layers.Dense(7, activation='relu'))
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, validation_split=0.33, epochs=30, callbacks=[keras.callbacks.EarlyStopping(patience=5)])

print(model.predict(test_data, batch_size=1))
model.save('vehicle_value_model.h5')