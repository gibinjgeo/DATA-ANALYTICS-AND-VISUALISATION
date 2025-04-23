import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# Define features and target
features = ['RAINFALL(MM)', 'Humidity (%)', 'Poultry Count (Millions)', 'TEMPERATURE(DEGREE CELCIUS)']
target = 'TOTAL NUMBER OF CASES'

# Drop rows with missing values
df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# # Load dataset
# df = pd.read_csv("data.csv")
#
# # Fix column names: strip whitespace
# df.columns = df.columns.str.strip()
#
# # Define features and target
# features = ['RAINFALL(MM)',
#             'Humidity (%)',
#             'Poultry Count (Millions)',
#             'TEMPERATURE(DEGREE CELCIUS)']
# target = 'TOTAL NUMBER OF CASES'
#
# # Drop missing values
# df = df.dropna(subset=features + [target])
#
# # --- Step 1: Remove outliers using IQR on the target ---
# Q1 = df[target].quantile(0.25)
# Q3 = df[target].quantile(0.75)
# IQR = Q3 - Q1
#
# # Define acceptable range
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
#
# # Filter data
# df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]
#
# # --- Step 2: Regression Modeling ---
# X = df[features]
# y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- PCA with 2 Components --------------------
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca_2, y, test_size=0.4, random_state=42)

# Train regression model
model_2 = LinearRegression()
model_2.fit(X_train, y_train)
y_pred_2 = model_2.predict(X_test)

# Evaluate
mae_2 = metrics.mean_absolute_error(y_test, y_pred_2)
mse_2 = metrics.mean_squared_error(y_test, y_pred_2)
rmse_2 = np.sqrt(mse_2)
r2_2 = metrics.r2_score(y_test, y_pred_2)

# -------------------- PCA with 1 Component --------------------
X_pca_1 = X_pca_2[:, [0]]
X_train, X_test, y_train, y_test = train_test_split(X_pca_1, y, test_size=0.4, random_state=42)

model_1 = LinearRegression()
model_1.fit(X_train, y_train)
y_pred_1 = model_1.predict(X_test)

# Evaluate
mae_1 = metrics.mean_absolute_error(y_test, y_pred_1)
mse_1 = metrics.mean_squared_error(y_test, y_pred_1)
rmse_1 = np.sqrt(mse_1)
r2_1 = metrics.r2_score(y_test, y_pred_1)

# -------------------- Print Results --------------------
print("\n📊 Model with 2 PCA Components:")
print(f"R²:   {r2_2:.4f}")
print(f"MAE:  {mae_2:.2f}")
print(f"RMSE: {rmse_2:.2f}")

print("\n📊 Model with 1 PCA Component:")
print(f"R²:   {r2_1:.4f}")
print(f"MAE:  {mae_1:.2f}")
print(f"RMSE: {rmse_1:.2f}")

# -------------------- Plot Comparisons --------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_1, alpha=0.7, color='orange', label='1 PCA Component')
plt.scatter(y_test, y_pred_2[:len(y_test)], alpha=0.7, color='green', label='2 PCA Components')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual Total Cases")
plt.ylabel("Predicted Total Cases")
plt.title("Actual vs Predicted (PCA Models)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
