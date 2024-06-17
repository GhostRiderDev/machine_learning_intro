import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv("data/Advertising.csv")

def regression_multiple(data, toRemove):
  x = data.drop(toRemove, axis=1).values
  y = data['Sales'].values
  
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

  lin_reg = LinearRegression()
  lin_reg.fit(x_train, y_train)

  y_pred = lin_reg.predict(x_test)

  print("Predicciones: {} Reales: {}".format(y_pred[:4], y_test[:4]))

  rmse = root_mean_squared_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)

  print("RMSE: ", rmse)
  print("R2: ", r2)

 
  sns.regplot(x= y_test, y=y_pred)
  plt.show()

regression_multiple(data,["Radio", "Sales"])

regression_multiple(data,["Newspaper", "Sales"])



