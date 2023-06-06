import altair
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import json
import branca
import folium
from streamlit_folium import st_folium
import altair as alt
import seaborn as sns
import os
from matplotlib_venn import venn3
import numpy as np

path = os.getcwd()

def get_data(path):
    # Read happiness data from CSV files for each year
    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    happiness_data = {}
    for year in years:
        csv_path = os.path.join(path, "..", "clean_data", f"{year}.csv")
        happiness_data[year] = pd.read_csv(csv_path)
    
    # Read geojson data for map visualization
    geojson_path = os.path.join(path, "..", "clean_data", "countries.geojson")
    with open(geojson_path) as response:
        geo = json.load(response)
    
    return happiness_data, geo


def visualize_happiness_map(happiness_data, geo, selected_year, labels, tooltips):
    df = happiness_data[selected_year]
    df_new = pd.DataFrame(df).set_index("Country")

    
    for feature in geo["features"]:
        if feature["properties"]["ADMIN"] in df_new.index:
            feature["properties"]["happinesstatus"] = f"Happiness score {selected_year}: {df_new.loc[feature['properties']['ADMIN']]['Happiness Score']}"
        else:
            feature["properties"]["happinesstatus"] = "No happiness Data"
    
    map = folium.Map(zoom_start=4, scrollWheelZoom=False, tiles='CartoDB positron')
    
    choropleth = folium.Choropleth(
        name="Income Status Map",
        geo_data=geo,
        data=df,
        columns=labels,
        key_on='feature.properties.ADMIN',
        line_opacity=0.8,
        fill_opacity=0.5,
        highlight=True,
        fill_color="YlGn",
        legend_name="Income Status"
    )
    
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(tooltips, labels=False)
    )
    
    choropleth.geojson.add_to(map)
    folium.LayerControl().add_to(map)
    
    st_map = st_folium(map, width=700, height=450)
    
    return st_map


def visualize_treemap(data, selected_year):
    df = data[selected_year]
    
    fig = go.Figure(
        go.Treemap(
            labels=['World', 'Region', 'Country'],
            parents=['', 'World', 'Region'],
            values=df['Happiness Score'],
            hovertemplate='<b>%{label}</b><br>Happiness Score: %{value}<br>Health (Life Expectancy): %{color}<extra></extra>',
            branchvalues='total',
            marker=dict(
                colors=df['Health (Life Expectancy)'],
                coloraxis='coloraxis'
            )
        )
    )
    
    fig.update_layout(
        margin=dict(t=5, l=2.5, r=2.5, b=2.5),
        coloraxis=dict(colorscale='RdBu')
    )
    
    return fig

def visualize_correlation_heatmap(df):
    # Calculate the correlation matrix
    corr = df.corr()

    # Create a heatmap using Seaborn
    fig, ax = plt.subplots()
    sns.heatmap(corr, cmap="Blues", annot=True, ax=ax)

    return fig

def visualize_box_plots(df, age_range, include_international=True, include_domestic=True):
    # Filter the dataframe based on age range
    df_filtered = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

    # Filter by student type
    if not include_international:
        df_filtered = df_filtered[df_filtered['inter_dom'] == 'Dom']
    if not include_domestic:
        df_filtered = df_filtered[df_filtered['inter_dom'] == 'Inter']

    # Create box plots using Plotly Express
    fig1 = px.box(df_filtered, x="Suicide", y="ToDep", color="Gender", points="all")
    fig1.update_traces(quartilemethod="exclusive")

    fig2 = px.box(df_filtered, x="Suicide", y="ToAS", color="Gender", points="all")
    fig2.update_traces(quartilemethod="exclusive")

    fig3 = px.box(df_filtered, x="Dep", y="ToDep", color="Gender", points="all")
    fig3.update_traces(quartilemethod="exclusive")

    fig4 = px.box(df_filtered, x="Dep", y="ToAS", color="Gender", points="all")
    fig4.update_traces(quartilemethod="exclusive")

    return fig1, fig2, fig3, fig4


################################################################################

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

#################################################

def main():
    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    selected_year = st.sidebar.selectbox('Select the year for Happiness data', years)

    # Call the get_data function to retrieve the necessary data
    data = get_data(path)

    # Visualize the happiness map
    st.header('Happy people around the world')
    st.write('Higher incomes lead to more happy faces. The countries with a higher economic score are more happy than the countries with low scores.'
             ' The freedom of choice and social support has a mixed effect on the happiness index depending on the country. The map shows the countries happiness score with economic, social support and freedom to make life choices score alongside the income for 7 years.')
    map_figure = visualize_happiness_map(data, selected_year, True, True, True)
    st_map = st_folium(map_figure, width=700, height=450)
    # Perform any additional visualization or interaction with the map as needed
    # ...

    # Visualize the treemap
    st.header('Treemap')
    st.subheader('Description about the graph')
    treemap_figure = visualize_treemap(data, selected_year)
    st.plotly_chart(treemap_figure, theme="streamlit", use_container_width=True)


    # Visualize the correlation heatmap
    st.header('Correlation between different Human Development Indices')
    correlation_heatmap = visualize_correlation_heatmap(data)
    st.pyplot(correlation_heatmap)


    # Visualize the box plots
    st.header('Box plot')
    box_plots = visualize_box_plots(data)


    ########################################

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

    st.title("Stacked barplot")
    st.write("Wich major has what illness.")
    st.pyplot(get_stacked_barplot_by_major_all(df2))

    st.title("Stacked barplot 2")
    st.write("Wich major has what illness without those that have none.")
    st.pyplot(get_stacked_barplot_by_major_with(df2))

    ##########################################
    
    
if __name__ == "__main__":
    main()
