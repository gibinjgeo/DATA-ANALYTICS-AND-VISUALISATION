import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
df = pd.read_csv("data.csv")

# Fix column names: strip whitespace
df.columns = df.columns.str.strip()

# Define features and target
features = ['RAINFALL(MM)',
            'Humidity (%)',
            'Poultry Count (Millions)',
            'TEMPERATURE(DEGREE CELCIUS)']
target = 'TOTAL NUMBER OF CASES'

# Drop missing values
df = df.dropna(subset=features + [target])

# --- Step 1: Remove outliers using IQR on the target ---
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1

# Define acceptable range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter data
df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]

# --- Step 2: Regression Modeling ---
X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Compute mean of values in (y) training set
y_base = np.mean(y_train)

# Replicate the mean values as many times as there are values in the test set
y_pred_base = [y_base] * len(y_test)


# Optional: Show the predicted values of (y) next to the actual values of (y)
df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
print(df_base_pred)

# Evaluate
intercept = model.intercept_
coefficients = list(zip(features, model.coef_))
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal', label='Data points')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect prediction')

plt.title("Multiple Linear Regression", fontsize=14)
plt.xlabel("Actual Total Number of Cases")
plt.ylabel("Predicted Total Number of Cases")
plt.legend()
plt.grid(True)

# Add performance metrics
textstr = f"Intercept: {intercept:.2f}\n"
textstr += '\n'.join([f"{f}: {coef:.4f}" for f, coef in coefficients])
textstr += f"\n\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.4f}"

plt.gcf().text(0.65, 0.2, textstr, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
plt.tight_layout()
plt.show()

# Prepare future data (2026) for prediction
future_data = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "RAINFALL(MM)": [90, 85, 70, 60, 65, 75, 80, 78, 85, 88, 90, 92],
    "Humidity (%)": [85, 84, 82, 80, 78, 76, 75, 74, 76, 78, 80, 82],
    "Poultry Count (Millions)": [80, 82, 83, 85, 84, 86, 88, 87, 85, 84, 82, 81],
    "TEMPERATURE(DEGREE CELCIUS)": [4.5, 5.2, 8.0, 11.5, 14.0, 18.5, 21.0, 20.5, 17.0, 12.0, 8.5, 5.0]
})

# Predict using trained model
X_future = future_data[features]
future_data["Predicted Cases"] = model.predict(X_future)

# Add average predicted value to the plot
average_predicted = df['TOTAL NUMBER OF CASES'].mean()

# Plot with average line
plt.figure(figsize=(10, 6))
plt.plot(future_data["Month"], future_data["Predicted Cases"], marker='o', linestyle='-', label='Predicted Cases')
plt.axhline(y=average_predicted, color='red', linestyle='--', label=f'Average: {average_predicted:.2f}')
plt.title("Forecasted Avian Disease Cases in 2026")
plt.xlabel("Month")
plt.ylabel("Predicted Total Cases")
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

