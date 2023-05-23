# -*- coding: utf-8 -*-
"""
Created on Mon May 22 11:43:53 2023

@author: User
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = os.getcwd()
df = pd.read_csv(path + "\clean_data\cleaned_data_global_mental_health.csv")
df

# Filter the DataFrame for "yes" and "no" values in the self_experience column
yes_df = df[df["Self_experience"] == "Yes"]
no_df = df[df["Self_experience"] == "No"]

# Count the occurrences of each country_wealth value in the filtered DataFrames
yes_wealth_counts = yes_df["Country_wealth"].value_counts()
no_wealth_counts = no_df["Country_wealth"].value_counts()

# Rename the bars
yes_wealth_counts = yes_wealth_counts.rename({
    "Lower-middle income": "Low-middle",
    "High income": "High",
    "Low income": "Low",
    "Upper-middle income": "Upper-middle"
})
no_wealth_counts = no_wealth_counts.rename({
    "Lower-middle income": "Low-middle",
    "High income": "High",
    "Low income": "Low",
    "Upper-middle income": "Upper-middle"
})

# Define bar colors
bar_colors = {
    "High": "darkgreen",
    "Upper-middle": "limegreen",
    "Low": "darkred",
    "Low-middle": "indianred"
}

# Plot the bar plots
plt.figure(figsize=(13, 4))

plt.subplot(1, 2, 1)
plt.bar(yes_wealth_counts.index, yes_wealth_counts.values, color=[bar_colors.get(x, 'gray') for x in yes_wealth_counts.index])
plt.xlabel("Country Wealth")
plt.ylabel("Count")
plt.title("Count of 'Yes' in Self Experience by Country Wealth")

plt.subplot(1, 2, 2)
plt.bar(no_wealth_counts.index, no_wealth_counts.values, color=[bar_colors.get(x, 'gray') for x in no_wealth_counts.index])
plt.xlabel("Country Wealth")
plt.ylabel("Count")
plt.title("Count of 'No' in Self Experience by Country Wealth")

plt.tight_layout()
plt.show()

df = pd.read_csv(path + "\clean_data\cleaned_data_global_mental_health.csv")
importance_counts = df["Importance"].value_counts(normalize=True) * 100

# Create a new DataFrame to display the results
importance_table = pd.DataFrame({
    "Importance": importance_counts.index,
    "Percentage": importance_counts.values
})

# Sort the table by the "Importance" column
importance_table = importance_table.sort_values("Percentage")

# Display the table
print(importance_table)