import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load and clean dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# Define regression-related features
features = ['RAINFALL(MM)', 'Humidity (%)', 'Poultry Count (Millions)', 'TEMPERATURE(DEGREE CELCIUS)']
target = 'TOTAL NUMBER OF CASES'

# Filter the dataset to include only relevant columns and drop missing values
regression_df = df[features + [target]].dropna()

# --- Correlation Heatmap (including target) ---
correlation_matrix = regression_df.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Regression Features and Target")
plt.tight_layout()
plt.show()

# --- Variance Inflation Factor (VIF) Calculation ---
X = add_constant(regression_df[features])  # Add intercept term
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# --- Correlation Heatmap (features only) ---
plt.figure(figsize=(8, 6))
sns.heatmap(regression_df[features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap: Features Only")
plt.tight_layout()
plt.show()

