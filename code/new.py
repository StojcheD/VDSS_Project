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
import seaborn as sns


APP_TITLE = 'The Name OF THe PRoject'
APP_SUB_TITLE = 'Source: asdfg'
st.set_page_config(APP_TITLE)
st.title(APP_TITLE)

years = ["2015", "2016","2017","2018","2019","2020","2021"]
happines_data = {}
st.header('Happy people around the world')
st.write('Higher incomes lead to more happy faces. The countries with a higher economic score are more happy than the countries with low scores.'
         ' The freedom of choice and social support has a mixed effect on the happiness index depending on the country. The map shows the countries happiness score with economic, social support and freedom to make life choices score alongside the income for 7 years.')

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


selectedYear = st.selectbox('Select the year for Happines data',years)
st.write("The data is from the year:",selectedYear)
happines_data[selectedYear] = pd.read_csv(f"D:\Bachelors\Semester2\VDSS\project_folder\VDSS_Visualisierungsprojekt\clean_data\{selectedYear}.csv")

countries = happines_data[selectedYear]["Country"].unique()
df = happines_data[selectedYear]
df_new = df.set_index("Country")

col1, col2, col3, col4 = st.columns(4)
with col1:
    happinesScore = st.checkbox("Happines Score", value=True)
with col2:
    economyScore = st.checkbox("Economy Score", value=True)
with col3:
    socialSupport = st.checkbox("Social Support", value=True)
with col4:
    Freedom = st.checkbox("Freedom to make life choices", value=True)

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
            feature["properties"]["economyScore"] = "No economy Data"

if socialSupport:
    labels.append("Social support")
    tooltips.append("socialSupport")
    for feature in geo["features"]:
        if feature["properties"]["ADMIN"] in countries:
             feature["properties"]["socialSupport"] = f"Social support {selectedYear}: {df_new.loc[feature['properties']['ADMIN']]['Social support']}"
        else:
            feature["properties"]["socialSupport"] = "No Social Support Data"

if Freedom:
    labels.append("Freedom to make life choices")
    tooltips.append("Freedom")
    for feature in geo["features"]:
        if feature["properties"]["ADMIN"] in countries:
             feature["properties"]["Freedom"] = f"Freedom to make life choices {selectedYear}: {df_new.loc[feature['properties']['ADMIN']]['Freedom to make life choices']}"
        else:
            feature["properties"]["Freedom"] = "No Data"
#first graph
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
        legend_name="Income Status"
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

#country_list = [''] + list(df['Country'].unique())
#country_list.sort()
#country_index = country_list.index(country_name) if country_name and country_name in country_list else 0
#st.sidebar.selectbox('Country', country_list, country_index)
st.caption(APP_SUB_TITLE)
expander = st.expander("Conclusions")
expander.write("The top five positions in the happiness score are taken up by the Nordic countries through out the seven years with Switzerland a non-Nordic country taking up top places in these years as well."
               " However, the country GDP doesn't guarantee happiness for all as it is the case in India, where the GDP score is among the highest while it's happiness score is among the last 20, it is most probably due to the unequal distribution of wealth. Afghanistan  due to prolonged wars remained in the lowest ranks in the happiness index.")
#secound grapf
st.header('Treemap')
st.subheader('description about the graph')

data = {}
selectedYear = st.select_slider("Choose a Year", options=years)
data[selectedYear] = pd.read_csv(f"D:\Bachelors\Semester2\VDSS\project_folder\VDSS_Visualisierungsprojekt\clean_data\{selectedYear}.csv")
df = data[selectedYear]
st.write("The data is from the year:",selectedYear)



fig = px.treemap(df,path=[px.Constant('World'),'Region','Country'], values='Happiness Score',color='Health (Life Expectancy)',hover_data=['Happiness Rank'],color_continuous_scale='RdBu',branchvalues = 'total')
fig.update_layout(margin = dict(t=5, l=2.5, r=2.5, b=2.5))
st.plotly_chart(fig)
st.caption('source of the graph maybe')
expander = st.expander("See Conclusions")
expander.write("something")
#3rd graph
st.header('Correlation between different Human Development Indices')
st.write("Higher education leads to higher country wealth, however, interestingly, higher education doesn't provides understanding of mental health importance, on the contrary has a negative effect. Also, higher education leads to lower mental health self-experiences.")

df1 = pd.read_csv('D:\Bachelors\Semester2\VDSS\project_folder\VDSS_Visualisierungsprojekt\clean_data\cleaned_data_global_mental_health.csv',usecols=[2,3,7,8])
df1.drop(df1[df1['Education'] == '99'].index, inplace = True)
df1.drop(df1[df1['Self_experience'] == ' '].index, inplace = True)
mapping_dict = {'Elementary or less': 0,'Secondary': 1,'Tertiary': 2, 'High income': 3,'Upper-middle income': 2,'Lower-middle income': 1,'Low income':0,'Less important':0,'As important':1,'More important':2,'Yes':1,'No':0}
df1 = df1.replace(mapping_dict)
df_corr = df1.corr()
fig, ax = plt.subplots()
sns.heatmap(df_corr, ax=ax,cmap="Blues",annot=True)
st.write(fig)
st.caption('Source: aaaa')


expander = st.expander("See Conclusions")
expander.write("The mental health importance is correlated with people's self-experience. The more important the mental health is the more people are going to self-experience it. However, both these indices are negatively correlated with country wealth and education."
               " While country wealth and education both are correlated among themselves. ")

