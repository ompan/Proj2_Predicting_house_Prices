import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("house_prices.csv")

# Features and Target
X = data[['bedrooms', 'square_footage']]
y = data['price']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'Coefficients: {model.coef_}')
print(f'Intercept: {model.intercept_}')

# Plot the results
plt.figure(figsize=(10, 5))

# Plot Bedrooms vs Price
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test['bedrooms'], y=y_test, color='blue', label='Actual')
sns.scatterplot(x=X_test['bedrooms'], y=y_pred, color='red', label='Predicted')
plt.title('Bedrooms vs Price')

# Plot Square Footage vs Price
plt.subplot(1, 2, 2)
sns.scatterplot(x=X_test['square_footage'], y=y_test, color='blue', label='Actual')
sns.scatterplot(x=X_test['square_footage'], y=y_pred, color='red', label='Predicted')
plt.title('Square Footage vs Price')

plt.tight_layout()
plt.show()