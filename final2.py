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

path = os.getcwd

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
    df_new = df.set_index("Country")
    
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

def main():
    years = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
    selected_year = st.sidebar.selectbox('Select the year for Happiness data', years)

    # Call the get_data function to retrieve the necessary data
    data = get_data(path)

    # Visualize the happiness map
    st.header('Happy people around the world')
    st.write('Higher incomes lead to more happy faces. The countries with a higher economic score are more happy than the countries with low scores.'
             ' The freedom of choice and social support has a mixed effect on the happiness index depending on the country. The map shows the countries happiness score with economic, social support and freedom to make life choices score alongside the income for 7 years.')
    map_figure = visualize_happiness_map(data, selected_year, True, True, True, True)
    st_map = st_folium(map_figure, width=700, height=450)
    # Perform any additional visualization or interaction with the map as needed
    # ...

    # Visualize the treemap
    st.header('Treemap')
    st.subheader('Description about the graph')
    treemap_figure = visualize_treemap(data, selected_year)
    st.plotly_chart(treemap_figure, theme="streamlit", use_container_width=True)
    # Perform any additional visualization or interaction with the treemap as needed
    # ...

    # Visualize the correlation heatmap
    st.header('Correlation between different Human Development Indices')
    correlation_heatmap = visualize_correlation_heatmap(data)
    st.pyplot(correlation_heatmap)
    # Perform any additional visualization or interaction with the correlation heatmap as needed
    # ...

    # Visualize the box plots
    st.header('Box plot')
    box_plots = visualize_box_plots(data)
    # Perform any additional visualization or interaction with the box plots as needed
    # ...

if __name__ == "__main__":
    main()
