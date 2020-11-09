import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.utils import shuffle

df = pd.read_csv('vehicles.csv', usecols=['Year', 'Make', 'Model', 'Condition', 'Title', 'Transmission', 'Odometer', 'Price'])
print(df.shape)
print(df.columns)

top_1000 = [x for x in df.Model.value_counts().sort_values(ascending=False).head(1000).index]

df_1000 = df[df.Model.isin(top_1000)]

print(df_1000.nunique(axis=0))
print(df_1000.describe().apply(lambda s: s.apply(lambda x: format(x, 'f'))))

df_final = pd.get_dummies(df_1000)

X = df_final.loc[:, df_final.columns != 'Price']
y = df_final['Price']

df_final = shuffle(df_final)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=0)
model = RandomForestRegressor(random_state=1, n_estimators=10)
model.fit(X_train, y_train)
pred = model.predict(X_test)

joblib.dump(model, 'vehicle_value_model_alt_est10.pkl')

print(mae(y_test, pred))
print(df_final['Price'].mean())
print(model.score(X_test, y_test))
