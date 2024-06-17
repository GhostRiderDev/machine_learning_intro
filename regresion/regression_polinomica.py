# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
pos = [x for x in range(1, 11)]
post = ["Pasante de Desarrollo",
 "Desarrollador Junior",
 "Desarrollador Intermedio",
 "Desarrollador Senior",
 "Líder de Proyecto",
 "Gerente de Proyecto",
 "Arquitecto de Software",
 "Director de Desarrollo",
 "Director de Tecnología",
 "Director Ejecutivo (CEO)"]
salary = [1200.0, 2500.0, 4000.0, 4800.0, 6500.0, 9000.0, 12820.0, 15000.0, 25000.0, 50000.0]

# %%
data = {
  "years": pos,
  "positions": post,
  "salary": salary
}

data = pd.DataFrame(data)
data.head()

# %%
plt.scatter(x=data["years"], y=data["salary"])
plt.show()

# %%
x = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, -1].values

# %%
from sklearn.linear_model import LinearRegression

regression = LinearRegression()

regression.fit(x, y)

# %%
predition =  regression.predict(x)

plt.scatter(data["years"], data["salary"])
plt.plot(x, predition, color="red")
plt.show()

# %%
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3)
x_poly = poly.fit_transform(x)
x_poly

# %%
regression_2  = LinearRegression()
regression_2.fit(x_poly, y)

# %%
plt.scatter(data["years"], data["salary"])
plt.plot(x, regression_2.predict(x_poly), color="green")
plt.show()

# %%
predict = poly.fit_transform([[2]])
regression_2.predict(predict)

# %%
from sklearn.metrics import r2_score
y_predict = regression_2.predict(x_poly)
r2 = r2_score(y, y_predict)
r2


