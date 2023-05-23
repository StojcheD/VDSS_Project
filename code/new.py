import altair
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import branca
import folium
from streamlit_folium import st_folium
import altair as alt


APP_TITLE = 'The Name OF THe PRoject'
APP_SUB_TITLE = 'Source: asdfg'
st.set_page_config(APP_TITLE)
st.title(APP_TITLE)

years = ["2015", "2016","2017"]
happines_data = {}
st.header('The name of the graph')
st.subheader('description about the graph')

def get_data():
    # happines_data = {}
    # for year in years:
    #     happines_data[year] = pd.read_csv(f"D:\Bachelors\Semester2\VDSS\project_folder\VDSS_Visualisierungsprojekt\clean_data\{year}.csv")
    #
    # # df = pd.read_csv(
    # #     'D:\Bachelors\Semester2\VDSS\project_folder\VDSS_Visualisierungsprojekt\clean_data\cleaned_data_global_mental_health.csv',
    # #     usecols=[0, 1, 9])
    # # df.drop_duplicates(subset=['Country'], keep='first', inplace=True)
    #     Countries = happines_data["2015"]["Country"].unique()
    # # mapping = {"High income": 4, "Upper-middle income": 3, "Lower-middle income": 2, "Low income": 1}
    # # df.replace(mapping , inplace=True)
    # # mapping_inv = {v: k for k, v in mapping.items()}
    #     df_indexed = happines_data[year].set_index("Country")
    with open(
            "D:\Bachelors\Semester2\VDSS\project_folder\VDSS_Visualisierungsprojekt\clean_data\countries.geojson") as response:
        geo = json.load(response)

    return geo ##, Countries, happines_data

geo = get_data()

labels = ["Country" ]
tooltips = ["ADMIN"]
with st.sidebar:
    # for c in countries:
    selectedYear = st.selectbox(
        'Select the year for Happines data',
        years)
    happines_data[selectedYear] = pd.read_csv(
        f"D:\Bachelors\Semester2\VDSS\project_folder\VDSS_Visualisierungsprojekt\clean_data\{selectedYear}.csv")

    countries = happines_data[selectedYear]["Country"].unique()
    df = happines_data[selectedYear]
    st.write(selectedYear)
    df_new = df.set_index("Country")


    happinesScore = st.checkbox("Happines Score", value = True)
    economyScore = st.checkbox("Economy Score", value = True)
    if happinesScore:
        labels.append("Happiness Score")
        tooltips.append("happinesstatus")
        for feature in geo["features"]:
            if feature["properties"]["ADMIN"] in countries:
                feature["properties"]["happinesstatus"] = f"Happines score {selectedYear}: {df_new.loc[feature['properties']['ADMIN']]['Happiness Score']}"
            else:
                feature["properties"]["happinesstatus"] = "No happiness Data"

    if economyScore:
        labels.append("Economy (GDP per Capita)")
        tooltips.append("economyScore")
        for feature in geo["features"]:
            if feature["properties"]["ADMIN"] in countries:
                feature["properties"]["economyScore"] = f"GDP score {selectedYear}: {df_new.loc[feature['properties']['ADMIN']]['Economy (GDP per Capita)']}"
            else:
                feature["properties"]["economyScore"] = "No happines Data"


map = folium.Map(zoom_start=4, scrollWheelZoom=False, tiles='CartoDB positron')

choropleth = folium.Choropleth(
        name = "Income Status Map",
        geo_data=geo,
        data=df,
        columns=labels,
        key_on='feature.properties.ADMIN',
        line_opacity=0.8,
        fill_opacity=0.5,
        highlight=True,
        fill_color="YlGn",
        legend_name="Unemployment Rate (%)"
    )
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(tooltips, labels=False)
)

choropleth.geojson.add_to(map)
folium.LayerControl().add_to(map)



st_map = st_folium(map, width=700, height=450)

if st_map['last_active_drawing']:
    country_name = st_map['last_active_drawing']['properties']['ADMIN']
else:
    country_name = ""

country_list = [''] + list(df['Country'].unique())
country_list.sort()
country_index = country_list.index(country_name) if country_name and country_name in country_list else 0
st.sidebar.selectbox('Country', country_list, country_index)
st.caption(APP_SUB_TITLE)





st.header('The name of the graph')
st.subheader('description about the graph')

fig = px.treemap(df,path=[px.Constant('World'),'Region','Country'], values='Happiness Score',color='Health (Life Expectancy)',hover_data=['Happiness Rank'],color_continuous_scale='RdBu',branchvalues = 'total')
fig.update_layout(margin = dict(t=5, l=2.5, r=2.5, b=2.5))
st.plotly_chart(fig)
st.caption('source of the graph maybe')