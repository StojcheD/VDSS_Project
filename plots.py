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
import numpy as np


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
    
    st.title("Bar Plot 2")
    st.write("The barplot below shows the amount of men and women there are and how many suffer of a mental illness.")
    st.pyplot(get_gender_barplot(df2))
    
    st.title("Barplot percentage")
    st.write("Below is a barplot, that shows the percentage of how many men and women have a menatl illness.")
    st.pyplot(get_gender_percentage_barplot(df2))

    st.title("Stacked barplot")
    st.write("Wich major has what illness.")
    st.pyplot(get_stacked_barplot_by_major_all(df2))

    st.title("Stacked barplot 2")
    st.write("Wich major has what illness, without those, that have none.")
    st.pyplot(get_stacked_barplot_by_major_with(df2))


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

def get_gender_barplot(df):
    men_count = len(df[df["Gender"] == "M"])
    men_mental_illness_count = len(df[(df["Gender"] == "M") & ((df["Depression"] == "Yes") | (df["Anxiety"] == "Yes") | (df["Panic_attacks"] == "Yes"))])
    women_count = len(df[df["Gender"] == "F"])
    women_mental_illness_count = len(df[(df["Gender"] == "F") & ((df["Depression"] == "Yes") | (df["Anxiety"] == "Yes") | (df["Panic_attacks"] == "Yes"))])

    labels = ["Men", "Men with Mental Illness", "Women", "Women with Mental Illness"]
    counts = [men_count, men_mental_illness_count, women_count, women_mental_illness_count]

    colors = ["deepskyblue", "mediumblue", "hotpink", "purple"]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, counts, color=colors)
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.title("Count of Men and Women with Mental Illness")

    # Add count values on top of the bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 counts[i], ha='center', va='bottom', color='white')

    plt.show()

    return plt

def get_gender_percentage_barplot(df):
    total_men = len(df[df["Gender"] == "M"])
    total_women = len(df[df["Gender"] == "F"])

    men_mental_illness_count = len(df[(df["Gender"] == "M") & ((df["Depression"] == "Yes") | (df["Anxiety"] == "Yes") | (df["Panic_attacks"] == "Yes"))])
    women_mental_illness_count = len(df[(df["Gender"] == "F") & ((df["Depression"] == "Yes") | (df["Anxiety"] == "Yes") | (df["Panic_attacks"] == "Yes"))])

    men_mental_illness_percentage = (men_mental_illness_count / total_men) * 100
    women_mental_illness_percentage = (women_mental_illness_count / total_women) * 100

    labels = ["Men", "Women"]
    percentages = [men_mental_illness_percentage, women_mental_illness_percentage]

    colors = ["mediumblue", "purple"]

    plt.figure(figsize=(6, 6))
    bars = plt.bar(labels, percentages, color=colors)
    plt.xlabel("Gender")
    plt.ylabel("Percentage")
    plt.title("Percentage of Men and Women with Mental Illness")

    # Add percentage values on top of the bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{percentages[i]:.2f}%", ha='center', va='bottom', color='black')

    plt.ylim(0, 100)  # Set y-axis limit from 0 to 100

    plt.show()
    
    return plt


def get_stacked_barplot_by_major_all(df):
    majors = df["Major"].unique()
    mental_illnesses = ["Depression", "Anxiety", "Panic_attacks"]

    illness_counts = []
    for major in majors:
        major_df = df[df["Major"] == major]
        illness_count = []
        for illness in mental_illnesses:
            count = len(major_df[major_df[illness] == "Yes"])
            illness_count.append(count)
        illness_counts.append(illness_count)

    bar_width = 0.5
    bar_positions = np.arange(len(majors))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    # Calculate the sum of illness counts for each major
    illness_sums = np.sum(illness_counts, axis=1)

    # Sort majors based on the sum of illness counts
    sorted_indices = np.argsort(illness_sums)[::-1]  # Get sorted indices in descending order
    majors = majors[sorted_indices]
    illness_counts = np.array(illness_counts)[sorted_indices]

    plt.figure(figsize=(10, 6))
    bars = []
    bottom = np.zeros(len(majors))

    for i, illness_count in enumerate(illness_counts.T):
        bar = plt.bar(bar_positions, illness_count, bottom=bottom, width=bar_width, color=colors[i])
        bars.append(bar)
        bottom += illness_count

    plt.xlabel("Major")
    plt.ylabel("Count")
    plt.title("Distribution of Mental Illness by Major")
    plt.xticks(bar_positions, majors, rotation=85)
    plt.legend(bars, mental_illnesses)

    plt.tight_layout()
    plt.show()
    
    return plt

def get_stacked_barplot_by_major_with(df):
    majors = df["Major"].unique()
    mental_illnesses = ["Depression", "Anxiety", "Panic_attacks"]

    illness_counts = []
    for major in majors:
        major_df = df[df["Major"] == major]
        illness_count = []
        for illness in mental_illnesses:
            count = len(major_df[major_df[illness] == "Yes"])
            illness_count.append(count)
        if any(illness_count):  # Check if any mental illness count is greater than 0
            illness_counts.append(illness_count)

    bar_width = 0.5
    bar_positions = np.arange(len(illness_counts))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # Blue, Orange, Green

    # Calculate the sum of illness counts for each major
    illness_sums = np.sum(illness_counts, axis=1)

    # Sort majors based on the sum of illness counts
    sorted_indices = np.argsort(illness_sums)[::-1]  # Get sorted indices in descending order
    majors = majors[sorted_indices]
    illness_counts = np.array(illness_counts)[sorted_indices]

    plt.figure(figsize=(10, 6))
    bars = []
    bottom = np.zeros(len(illness_counts))

    for i, illness_count in enumerate(illness_counts.T):
        bar = plt.bar(bar_positions, illness_count, bottom=bottom, width=bar_width, color=colors[i])
        bars.append(bar)
        bottom += illness_count

    plt.xlabel("Major")
    plt.ylabel("Count")
    plt.title("Distribution of Mental Illness by Major")
    plt.xticks(bar_positions, majors, rotation=85)
    plt.legend(bars, mental_illnesses)

    plt.tight_layout()
    plt.show()

    return plt

if __name__ == "__main__":
    main()
