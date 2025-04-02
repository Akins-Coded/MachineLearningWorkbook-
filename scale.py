import pandas
# from sklearn import linear_model
# from sklearn.preprocessing import StandardScaler

# scale = StandardScaler()
# df = pandas.read_csv('Heart.csv')
# X = df[['Age', 'Sex']]

# scaledX = scale.fit_transform(X)
# print(scaledX)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv('home_data.csv')
X = df[['floors', 'bedroom', 'bathrooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
