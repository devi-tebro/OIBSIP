import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df= pd.read_csv(r'C:\Users\devis\Documents\Osasis\car data.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
#print(df.isnull().sum())

#right skewed indicating large proportion of low to mid priced cars and small proportion of high priced cars
sns.histplot(df["Selling_Price"], kde=True, color="teal")
plt.title("Selling Price Distribution")
plt.xlabel("Selling Price (Lakhs)")
plt.ylabel("Frequency/Density")
plt.show()

# Presence of high-value outliers and right skewness in selling prices
plt.boxplot(df["Selling_Price"])
plt.ylabel("Selling Price")
plt.title("Box Plot of Selling Price")
plt.show()

# Newer cars generally have higher selling prices, showing a negative relationship between age and price
plt.scatter(df["Year"], df["Selling_Price"])
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.title("Year vs Selling Price")
plt.show()

# Selling price decreases as kilometers driven increases, indicating an inverse relationship
plt.scatter(df["Driven_kms"], df["Selling_Price"])
plt.xlabel("Kms Driven")
plt.ylabel("Selling Price")
plt.title("Kms Driven vs Selling Price")
plt.show()

# Selling price varies across fuel types, with diesel vehicles showing higher median prices
sns.boxplot(x="Fuel_Type", y="Selling_Price", data=df)
plt.title("Fuel Type vs Selling Price")
plt.show()

# Automatic cars tend to have higher selling prices compared to manual cars
sns.boxplot(x="Transmission", y="Selling_Price", data=df)
plt.title("Transmission vs Selling Price")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("Selling_Price", axis=1)
y = df_encoded["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Linear Regression Performance")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(
    max_depth=6,
    random_state=42
)

dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mean_squared_error(y_test, y_pred_dt))
r2_dt = r2_score(y_test, y_pred_dt)

print("Decision Tree Performance")
print("MAE:", mae_dt)
print("RMSE:", rmse_dt)
print("R2 Score:", r2_dt)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Performance")
print("MAE:", mae_rf)
print("RMSE:", rmse_rf)
print("R2 Score:", r2_rf)

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gbr.fit(X_train, y_train)

y_pred_gbr = gbr.predict(X_test)
mae_gbr = mean_absolute_error(y_test, y_pred_gbr)
rmse_gbr = np.sqrt(mean_squared_error(y_test, y_pred_gbr))
r2_gbr = r2_score(y_test, y_pred_gbr)

print("Gradient Boosting Performance")
print("MAE:", mae_gbr)
print("RMSE:", rmse_gbr)
print("R2 Score:", r2_gbr)

results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
    "MAE": [mae, mae_dt, mae_rf, mae_gbr],
    "RMSE": [rmse, rmse_dt, rmse_rf, rmse_gbr],
    "R2 Score": [r2, r2_dt, r2_rf, r2_gbr]
})

print(results)

sample_car = X_test.iloc[0:1] 
actual_price = y_test.iloc[0]

# Use the Random Forest model to predict
predicted_price = rf.predict(sample_car)

print(f"Actual Price in Dataset: {actual_price} Lakhs")
print(f"Model's Predicted Price: {predicted_price[0]:.2f} Lakhs")