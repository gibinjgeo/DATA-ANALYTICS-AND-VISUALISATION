import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# Define target and possible features
target = 'TOTAL NUMBER OF CASES'
possible_features = ['RAINFALL(MM)',
    '0-7', 'ADULT', 'IMMATURE', 'UNKNOWN',
     'Scotland', 'South East'
]

# Keep only features that exist in the dataset
features = [col for col in possible_features if col in df.columns]

# Drop rows with missing values
df = df.dropna(subset=features + [target])

# --- Remove outliers from the target using IQR ---
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]

# Prepare data
X = df[features]
y_log = np.log1p(df[target])  # log(1 + y) transformation

# Train-test split
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Train Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
model.fit(X_train, y_train_log)

# Predict and back-transform
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test = np.expm1(y_test_log)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display metrics
print("📊 Gradient Boosting Model Performance (Outliers Removed):")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")

# --- Plot Actual vs Predicted ---
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel("Actual Total Cases")
plt.ylabel("Predicted Total Cases")
plt.title("Actual vs Predicted (Gradient Boosting with Outlier Removal)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Feature Importances ---
feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(np.array(features)[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance from Gradient Boosting")
plt.tight_layout()
plt.show()
