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
    #     happines_data[year] = pd.read_csv(path + f"\..\clean_data\{year}.csv")
    #
    # # df = pd.read_csv(
    # #     path + '\..\clean_data\cleaned_data_global_mental_health.csv',
    # #     usecols=[0, 1, 9])
    # # df.drop_duplicates(subset=['Country'], keep='first', inplace=True)
    #     Countries = happines_data["2015"]["Country"].unique()
    # # mapping = {"High income": 4, "Upper-middle income": 3, "Lower-middle income": 2, "Low income": 1}
    # # df.replace(mapping , inplace=True)
    # # mapping_inv = {v: k for k, v in mapping.items()}
    #     df_indexed = happines_data[year].set_index("Country")
    with open(
            path + "\clean_data\countries.geojson") as response:
        geo = json.load(response)

    return geo ##, Countries, happines_data

geo = get_data()

labels = ["Country" ]
tooltips = ["ADMIN"]


selectedYear = st.selectbox('Select the year for Happines data',years)
st.write("The data is from the year:",selectedYear)
happines_data[selectedYear] = pd.read_csv(path + f"\clean_data\{selectedYear}.csv")

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
data[selectedYear] = pd.read_csv(path + f"\clean_data\{selectedYear}.csv")
df = data[selectedYear]
st.write("The data is from the year:",selectedYear)



fig = px.treemap(df,path=[px.Constant('World'),'Region','Country'], values='Happiness Score',color='Health (Life Expectancy)',hover_data=['Happiness Rank'],color_continuous_scale='RdBu',branchvalues = 'total')
fig.update_layout(margin = dict(t=5, l=2.5, r=2.5, b=2.5))
tab1, tab2 = st.tabs(["Streamlit theme (default)", "Plotly native theme"])
with tab1:
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
with tab2:
    st.plotly_chart(fig, theme=None, use_container_width=True)
st.caption('source of the graph maybe')
expander = st.expander("See Conclusions")
expander.write("something")
#3rd graph
st.header('Correlation between different Human Development Indices')
st.write("Higher education leads to higher country wealth, however, interestingly, higher education doesn't provides understanding of mental health importance, on the contrary has a negative effect. Also, higher education leads to lower mental health self-experiences.")

df1 = pd.read_csv(path + '\clean_data\cleaned_data_global_mental_health.csv',usecols=[2,3,7,8])
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





#graph4


st.header('Box plot')

df4 = pd.read_csv(path + '\clean_data\japan_student_mental_health.csv')

# assuming 'age' is ranging from 15 to 100
age_range = st.slider('Select age range', min_value=17, max_value=31, value=(17, 31))

# filter dataframe by selected age range
df4 = df4[(df4['Age'] >= age_range[0]) & (df4['Age'] <= age_range[1])]

col1, col2 = st.columns(2)
with col1:
    International = st.checkbox("International Students", value=True)
with col2:
    Domestic = st.checkbox("Domestic student ", value=True)

if International and Domestic:
    df4 = df4
elif Domestic:
    df4 = df4.query("inter_dom == 'Dom'")
elif International:
    df4 =df4.query("inter_dom == 'Inter'")
else:
    df4 = pd.DataFrame(columns = df4.columns)  # Create an empty DataFrame with the same columns as df4

fig1 = px.box(df4, x="Suicide", y="ToDep", color="Gender",points="all")
fig1.update_traces(quartilemethod="exclusive")

fig2 = px.box(df4, x="Suicide", y="ToAS", color="Gender",points="all")
fig2.update_traces(quartilemethod="exclusive")

fig3 = px.box(df4, x="Dep", y="ToDep", color="Gender",points="all")
fig3.update_traces(quartilemethod="exclusive")

fig4 = px.box(df4, x="Dep", y="ToAS", color="Gender",points="all")
fig4.update_traces(quartilemethod="exclusive")

tab1, tab2, tab3, tab4 = st.tabs(["Suicide vs ToDep", "Suicide vs ToAS","Depression vs ToDep","Depression vs ToAS"],)
with tab1:
    st.write(fig1)
with tab2:
    st.write(fig2)
with tab3:
    st.write(fig3)
with tab4:
  st.write(fig4)

expander = st.expander("See Conclusions")
expander.write("asdda")

#students = {}


def main():
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

if __name__ == "__main__":
        main()