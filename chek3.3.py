import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


data = pd.read_csv("titanic-passengers.csv")


features = data[["Age", "Fare"]]
target = data["Survived"]


features.fillna(features.mean(), inplace=True) 

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


model_linear = LinearRegression()
model_linear.fit(X_train, y_train)


plt.scatter(X_train["Age"], y_train, label='Données d\'entraînement', color='blue')
plt.plot(X_train["Age"], model_linear.predict(X_train), label='Régression linéaire', color='red')
plt.xlabel('Âge')
plt.ylabel('Survie')
plt.legend()
plt.title('Régression Linéaire')
plt.show()


y_pred_linear = model_linear.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"Erreur quadratique moyenne (MSE) : {mse_linear}")
print(f"Coefficient de détermination (R²) : {r2_linear}")

model_multiple = LinearRegression()
model_multiple.fit(X_train, y_train)


degree = 2
model_poly = make_pipeline(PolynomialFeatures(degree), LinearRegression())
model_poly.fit(X_train, y_train)


y_pred_multiple = model_multiple.predict(X_test)
y_pred_poly = model_poly.predict(X_test)

mse_multiple = mean_squared_error(y_test, y_pred_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print(f"Performance de la régression linéaire multiple - MSE : {mse_multiple}, R² : {r2_multiple}")
print(f"Performance de la régression polynomiale (degré {degree}) - MSE : {mse_poly}, R² : {r2_poly}")
