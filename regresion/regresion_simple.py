import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/Advertising.csv")
data.head()

data = data.iloc[:, 1:]
data.head()

data.info()

data.describe()

data.columns

cols = ['Radio', 'Newspaper', 'TV']

for col in cols:
  plt.plot(data[col], data['Sales'], 'ro')
  plt.title("Venta respecto a la publicidad en %s" % col)
  plt.show()


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

X = data['TV'].values.reshape(-1, 1)
Y = data['Sales'].values

x_train, x_test, y_train, t_test = train_test_split(X,Y, test_size=0.2, random_state=42)
print(x_test.shape)
print(x_train.shape)

# %%
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

predict = lin_reg.predict(x_test)
predict

print("Predicciones: {} Reales: {}".format(predict[:4], t_test[:4]))

# RMSE => is good is nearly to min Independent variable
rmse = root_mean_squared_error(predict, t_test)
print("rmse: ", rmse)
r2 = r2_score(predict, t_test)
print("r2: ", r2)

plt.plot(x_test, t_test, 'bo')
plt.plot(x_test, predict)
plt.show()

def modelos_simple(independient):
  X = data[independient].values.reshape(-1, 1)
  Y = data['Sales'].values
  
  x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

  lin_reg = LinearRegression()
  lin_reg.fit(x_train, y_train)
  predict = lin_reg.predict(x_test)

  rmse = root_mean_squared_error(predict, t_test)
  print("RMSE: ", rmse)
  r2 = r2_score(predict, t_test)
  print("R2: ", r2)
  
  plt.plot(x_test, t_test, 'bo')
  plt.plot(x_test, predict)
  plt.show()

modelos_simple("Radio")

modelos_simple("Newspaper")