import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Wczytanie zestawu danych
california_housing = fetch_california_housing(as_frame=True)
X = california_housing.data
y = california_housing.target

# Podział zestawu na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Wykonanie regresji liniowej na zbiorze uczącym
model = LinearRegression()
model.fit(X_train, y_train)

# Obliczenie błędów dla zbioru uczącego i testowego
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print(f"MAE dla uczących: {mae_train:.2f}")
print(f"MAE dla testowych: {mae_test:.2f}")
print(f"MSE dla uczących: {mse_train:.2f}")
print(f"MSE dla testowych: {mse_test:.2f}")

# Rysowanie wykresów dla każdej zmiennej niezależnej
for i, col in enumerate(X.columns):
    # Wykonanie regresji liniowej z wykorzystaniem jednej zmiennej
    model_i = LinearRegression()
    model_i.fit(X_train[[col]], y_train)
    y_train_pred_i = model_i.predict(X_train[[col]])
    y_test_pred_i = model_i.predict(X_test[[col]])

    # Obliczenie błędów dla zmiennej niezależnej
    mae_train_i = mean_absolute_error(y_train, y_train_pred_i)
    mae_test_i = mean_absolute_error(y_test, y_test_pred_i)
    mse_train_i = mean_squared_error(y_train, y_train_pred_i)
    mse_test_i = mean_squared_error(y_test, y_test_pred_i)

    # Rysowanie wykresu dla zmiennej niezależnej
    plt.subplot(2, 4, i+1)
    plt.scatter(X_train[col], y_train, s=2)
    plt.scatter(X_test[col], y_test, s=2, alpha=0.5)
    plt.plot(X_train[col], y_train_pred_i, color='red')
    plt.plot(X_test[col], y_test_pred_i, color='orange')
    plt.xlabel(col)
    plt.ylabel("y")
    plt.title(f"MAE(train)={mae_train_i:.2f}\nMAE(test)={mae_test_i:.2f}")
plt.tight_layout()
plt.show()