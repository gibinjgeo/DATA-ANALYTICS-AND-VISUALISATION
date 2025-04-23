import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("data.csv")

# Define age category columns
age_columns = ['0-7', 'ADULT', 'IMMATURE', 'UNKNOWN']

# Sum the total cases for each age category
age_totals = df[age_columns].sum()

# Plot as a bar graph
plt.figure(figsize=(8, 5))
plt.bar(age_totals.index, age_totals.values)

plt.title('Total Cases by Age Category')
plt.xlabel('Age Category')
plt.ylabel('Total Number of Cases')
plt.tight_layout()
plt.show()
