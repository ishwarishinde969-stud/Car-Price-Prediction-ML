import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("graphs", exist_ok=True)

df = pd.read_csv("data/car data.csv")

# Selling Price Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Selling_Price'], kde=True)
plt.savefig("graphs/selling_price_distribution.png")
plt.close()

# Fuel Type vs Selling Price
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fuel_Type'], y=df['Selling_Price'])
plt.savefig("graphs/fuel_type_vs_price.png")
plt.close()

# Car Age vs Selling Price
df['Car_Age'] = 2024 - df['Year']

plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Car_Age'], y=df['Selling_Price'])
plt.savefig("graphs/car_age_vs_price.png")
plt.close()

print("EDA graphs saved")