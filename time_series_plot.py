import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")


# Drop rows where year (Column 25) or month (Column 24) is missing
df_clean = df.dropna(subset=['Column 25', 'Column 24'])

# Create the datetime column
df_clean['Date'] = pd.to_datetime(df_clean['Column 25'].astype(int).astype(str) + '-' + df_clean['Column 24'] + '-01', format='%Y-%b-%d')

# Identify the total cases column
case_col = [col for col in df_clean.columns if 'TOTAL' in col.upper() and 'CASE' in col.upper()][0]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df_clean['Date'], df_clean[case_col], marker='o')

plt.title('Total Number of Cases Over Time')
plt.xlabel('Year and Month')
plt.ylabel('Total Number of Cases')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

