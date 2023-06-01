# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:26:16 2023

@author: User
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib_venn import venn3


def main():
    path = os.getcwd()
    df = pd.read_csv(path + "\clean_data\cleaned_data_global_mental_health.csv")
    df2 = pd.read_csv(path + "\clean_data\malaysia_clean_mental_health.csv")

    st.title("Importance Table")
    st.write("The table below shows the percentage of importance values:")
    st.table(get_importance_table(df))

    plt.tight_layout()
    st.title("Bar Plots")
    st.write("The bar plots below show the count of 'Yes' and 'No' in Self Experience by Country Wealth:")
    st.pyplot(get_bar_plots(df))

    st.title("Venn Diagram")
    st.write("The Venn diagram below shows the overlap of mental illnesses in Malaysia:")
    st.pyplot(get_venn_diagram(df2))

def get_importance_table(df):
    importance_counts = df["Importance"].value_counts(normalize=True) * 100

    # Get unique regions from the DataFrame and sort them alphabetically
    regions = sorted(df["Region"].unique())

    # Create a new DataFrame to display the results
    importance_table = pd.DataFrame({"Region": regions})

    # Initialize columns with zeros in the desired sequence
    columns = ["Region", "More important", "As important", "Less important"]
    for column in columns[1:]:
        importance_table[column] = 0

    # Fill the table with percentage values per region
    for region in regions:
        region_df = df[df["Region"] == region]
        region_counts = region_df["Importance"].value_counts(normalize=True) * 100
        for index, count in region_counts.items():
            importance_table.loc[importance_table["Region"] == region, index] = count

    # Sort the table by the "Region" column in alphabetical order
    importance_table = importance_table.sort_values("Region")

    # Add summary line
    importance_table.loc[len(importance_table)] = [
        "Summary",
        importance_table["More important"].mean(),
        importance_table["As important"].mean(),
        importance_table["Less important"].mean()
    ]

    return importance_table


def get_bar_plots(df):
    yes_df = df[df["Self_experience"] == "Yes"]
    no_df = df[df["Self_experience"] == "No"]
    yes_wealth_counts = yes_df["Country_wealth"].value_counts()
    no_wealth_counts = no_df["Country_wealth"].value_counts()

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

    bar_colors = {
        "High": "darkgreen",
        "Upper-middle": "limegreen",
        "Low": "darkred",
        "Low-middle": "indianred"
    }

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
    
    return plt


def get_venn_diagram(df):
    count = df["Depression"].value_counts()["Yes"]

    depression_yes = set(df[df["Depression"] == "Yes"].index)
    anxiety_yes = set(df[df["Anxiety"] == "Yes"].index)
    panic_attacks_yes = set(df[df["Panic_attacks"] == "Yes"].index)

    no_df = df[(df["Depression"] == "No") & (df["Anxiety"] == "No") & (df["Panic_attacks"] == "No")]
    no_count = len(no_df)

    venn_labels = {
        "100": len(depression_yes - anxiety_yes - panic_attacks_yes),
        "010": len(anxiety_yes - depression_yes - panic_attacks_yes),
        "001": len(panic_attacks_yes - depression_yes - anxiety_yes),
        "110": len(depression_yes & anxiety_yes - panic_attacks_yes),
        "101": len(depression_yes & panic_attacks_yes - anxiety_yes),
        "011": len(anxiety_yes & panic_attacks_yes - depression_yes),
        "111": len(depression_yes & anxiety_yes & panic_attacks_yes),
    }

    plt.figure(figsize=(8, 8))
    venn = venn3(subsets=venn_labels, set_labels=("Depression", "Anxiety", "Panic Attacks"))

    # Access the circles and modify their properties
    for circle in venn.patches:
        circle.set_edgecolor('black')
        circle.set_linewidth(1.5)

    # Add the count from no_df in a corner with label
    plt.text(0.45 , -0.5, f"No Mental Illness:\n{no_count}", horizontalalignment='left',
             verticalalignment='bottom', fontsize=12)

    plt.title("Venn Diagram of Mental Illnesses")
    plt.show()

    return plt

if __name__ == "__main__":
    main()
