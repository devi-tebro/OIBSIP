import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the dataset
df = pd.read_csv(r'C:\Users\devis\Documents\Osasis\Advertising.csv')

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
print("Dataset Preview:")
print(df.head())
print(df.isnull().sum())

plt.subplot(1, 3, 1)
sns.regplot(x='TV', y='Sales', data=df, line_kws={"color": "red"})
plt.title('TV vs Sales')

plt.subplot(1, 3, 2)
sns.regplot(x='Radio', y='Sales', data=df, line_kws={"color": "red"})
plt.title('Radio vs Sales')

plt.subplot(1, 3, 3)
sns.regplot(x='Newspaper', y='Sales', data=df, line_kws={"color": "red"})
plt.title('Newspaper vs Sales')

plt.tight_layout()
plt.show()

X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.4f}")

print(f"Intercept (Base Sales): {model.intercept_:.2f}")#even without advertising they earn
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("Model Coefficients")
print(coefficients)

new_data = pd.DataFrame([[75,150,30]], columns=['TV', 'Radio', 'Newspaper'])
prediction = model.predict(new_data)
print(f"\nPredicted Sales for the given spend: {prediction[0]:.2f}")