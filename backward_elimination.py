import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load and clean dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# Define features and target
features = ['RAINFALL(MM)', 'Humidity (%)', 'Poultry Count (Millions)', 'TEMPERATURE(DEGREE CELCIUS)']
target = 'TOTAL NUMBER OF CASES'

# Drop missing values
df = df.dropna(subset=features + [target])

# --- Remove outliers from target using IQR method ---
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter the dataset
df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]

# --- Backward Elimination ---
X = df[features]
y = df[target]

# Add constant term for intercept
X_be = sm.add_constant(X)

# Initialize loop variables
current_features = X_be.columns.tolist()
removed_features = []
step = 1

# Backward elimination loop
while True:
    X_train, X_test, y_train, y_test = train_test_split(X_be[current_features], y, test_size=0.4, random_state=0)
    model = sm.OLS(y_train, X_train).fit()
    p_values = model.pvalues.drop("const", errors='ignore')

    max_p_feature = p_values.idxmax()
    max_p_value = p_values[max_p_feature]

    # Display current step and p-values
    print(f"\nStep {step} - Model with features: {current_features}")
    print(p_values)

    if max_p_value < 0.05:
        print("\n All remaining features are statistically significant (p < 0.05).")
        break

    print(f"\n Eliminating feature: '{max_p_feature}' with p-value = {max_p_value:.4f} (not significant)")
    current_features.remove(max_p_feature)
    removed_features.append((max_p_feature, max_p_value))
    step += 1

# Final model summary
print("\nFinal model includes features:", current_features)
final_model = sm.OLS(y_train, X_train[current_features]).fit()
print("\n📋 Final Model Summary:\n")
print(final_model.summary())

# Print removed features
print("\n🗑️ Removed Features (in order):")
for f, p in removed_features:
    print(f"- {f}: p = {p:.4f}")
