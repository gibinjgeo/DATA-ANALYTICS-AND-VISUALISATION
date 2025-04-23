import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data.csv")

# List of region columns (update this list if needed)
region_columns = [
    'East Midlands', 'East of England', 'London', 'North East', 'North West',
    'Scotland', 'South East', 'South West', 'Wales', 'West Midlands',
    'Yorkshire and The Humber '
]

# Sum the total cases for each region
region_totals = df[region_columns].sum()

# Plot as a bar graph
plt.figure(figsize=(12, 6))
plt.bar(region_totals.index, region_totals.values)

plt.title('Total Number of Cases by Region')
plt.xlabel('Region')
plt.ylabel('Total Number of Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
