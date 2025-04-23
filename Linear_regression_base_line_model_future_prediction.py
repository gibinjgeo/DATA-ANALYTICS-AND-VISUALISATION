import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load dataset
df = pd.read_csv("data.csv")

# Fix column names
df.columns = df.columns.str.strip()

# Define features and target
# Define features and target
features = ['RAINFALL(MM)',
            'Humidity (%)',
            'Poultry Count (Millions)',
            'TEMPERATURE(DEGREE CELCIUS)']
target = 'TOTAL NUMBER OF CASES'

# Drop missing values
df = df.dropna(subset=features + [target])

# Prepare data
X = df[features]
y = df[target]

# Train-test split (60% train, 40% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model parameters
intercept = model.intercept_
coefficients = list(zip(features, model.coef_))

# Evaluate Linear Regression Model
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
rmse_norm = rmse / (y.max() - y.min())
r2 = metrics.r2_score(y_test, y_pred)

# Dummy model (mean-based)
y_base = np.mean(y_train)
y_pred_base = [y_base] * len(y_test)

# Evaluate Dummy Model
mae_base = metrics.mean_absolute_error(y_test, y_pred_base)
mse_base = metrics.mean_squared_error(y_test, y_pred_base)
rmse_base = math.sqrt(mse_base)
rmse_norm_base = rmse_base / (y.max() - y.min())
r2_base = metrics.r2_score(y_test, y_pred_base)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='teal', label='Data points')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect prediction')
plt.title("Linear Regression: Actual vs Predicted", fontsize=14)
plt.xlabel("Actual Total Cases")
plt.ylabel("Predicted Total Cases")
plt.legend()
plt.grid(True)
textstr = f"Intercept: {intercept:.2f}\n"
textstr += '\n'.join([f"{f}: {coef:.4f}" for f, coef in coefficients])
textstr += f"\n\nMAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nRMSE Norm: {rmse_norm:.4f}\nR²: {r2:.4f}"
plt.gcf().text(0.65, 0.2, textstr, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
plt.tight_layout()
plt.show()

# --- Visualize Dummy Model ---
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_base, alpha=0.6, color='orange', label='Baseline')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect prediction')
plt.title("Baseline Model (Mean Prediction)", fontsize=14)
plt.xlabel("Actual Total Cases")
plt.ylabel("Predicted Total Cases")
plt.legend()
plt.grid(True)
textstr_base = f"Baseline MAE: {mae_base:.2f}\nMSE: {mse_base:.2f}\nRMSE: {rmse_base:.2f}\nRMSE Norm: {rmse_norm_base:.4f}\nR²: {r2_base:.4f}"
plt.gcf().text(0.65, 0.2, textstr_base, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5'))
plt.tight_layout()
plt.show()

# --- Forecast for 2026 ---
future_data = pd.DataFrame({
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "RAINFALL(MM)": [90, 85, 70, 60, 65, 75, 80, 78, 85, 88, 90, 92],
    "Humidity (%)": [85, 84, 82, 80, 78, 76, 75, 74, 76, 78, 80, 82],
    "Poultry Count (Millions)": [80, 82, 83, 85, 84, 86, 88, 87, 85, 84, 82, 81],
    "TEMPERATURE(DEGREE CELCIUS)": [4.5, 5.2, 8.0, 11.5, 14.0, 18.5, 21.0, 20.5, 17.0, 12.0, 8.5, 5.0]
})

X_future = future_data[features]
future_data["Predicted Cases"] = model.predict(X_future)

# Plot future predictions
plt.figure(figsize=(10, 6))
plt.plot(future_data["Month"], future_data["Predicted Cases"], marker='o', linestyle='-')
plt.title("Forecasted Avian Disease Cases in 2026")
plt.xlabel("Month")
plt.ylabel("Predicted Total Cases")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# import ace_tools as tools; tools.display_dataframe_to_user(name="2026 Predicted Cases", dataframe=future_data[["Month", "Predicted Cases"]])
