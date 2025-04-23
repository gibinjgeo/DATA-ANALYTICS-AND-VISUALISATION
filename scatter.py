# Re-run after kernel reset
import pandas as pd
import matplotlib.pyplot as plt

# Reload dataset
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# Define features and target
features = ['RAINFALL(MM)', 'Humidity (%)', 'Poultry Count (Millions)', 'TEMPERATURE(DEGREE CELCIUS)']
target = 'TOTAL NUMBER OF CASES'

# Drop missing values
df = df.dropna(subset=features + [target])

# Remove outliers using IQR method on target
Q1 = df[target].quantile(0.25)
Q3 = df[target].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_clean = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]

# Create 4 scatter plots in a single figure with adjusted layout
fig, axs = plt.subplots(2, 2, figsize=(12, 9))
axs = axs.ravel()

for i, feature in enumerate(features):
    axs[i].scatter(df_clean[feature], df_clean[target], alpha=0.7)
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel(target)
    axs[i].set_title(f'{feature} vs {target}')
    axs[i].grid(True)

plt.suptitle("Scatter Plots (Outliers Removed): Features vs Total Cases", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit the title
plt.show()
