import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import json
import folium
from streamlit_folium import st_folium
import seaborn as sns
import os
from matplotlib_venn import venn3
import numpy as np
import altair as alt


##############################################################################

path = os.getcwd()


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


df_japan_uni = pd.read_csv(path + "\clean_data\japan_student_mental_health.csv")
df_global_mental_health = pd.read_csv(path + "\clean_data\cleaned_data_global_mental_health.csv")
df_malaysia_uni = pd.read_csv(path + "\clean_data\malaysia_clean_mental_health.csv")


##############################################################################

APP_TITLE = 'The Name OF THe PRoject'


APP_SUB_TITLE = 'Source: asdfg'
st.set_page_config(APP_TITLE)
st.title(APP_TITLE)


##############################################################################


# world map


years = ["2015", "2016","2017","2018","2019","2020","2021"]
happines_data = {}
st.header('Happy people around the world')
st.write('Higher incomes lead to more happy faces. The countries with a higher economic score are more happy than the countries with low scores.'
         ' The freedom of choice and social support has a mixed effect on the happiness index depending on the country. The map shows the countries happiness score with economic, social support and freedom to make life choices score alongside the income for 7 years.')


labels = ["Country"]
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


##############################################################################


# Treemap

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


##############################################################################

# Importance Table

st.title("Importance Table")
st.write("The table below shows the percentage of importance values:")
importance_counts = df_global_mental_health["Importance"].value_counts(normalize=True) * 100
regions = sorted(df_global_mental_health["Region"].unique())
importance_table = pd.DataFrame({"Region": regions})
columns = ["Region", "More important", "As important", "Less important"]
for column in columns[1:]:
    importance_table[column] = 0

for region in regions:
    region_df = df_global_mental_health[df_global_mental_health["Region"] == region]
    region_counts = region_df["Importance"].value_counts(normalize=True) * 100
    for index, count in region_counts.items():
        importance_table.loc[importance_table["Region"] == region, index] = count

importance_table = importance_table.sort_values("Region")
importance_table.loc[len(importance_table)] = [
    "Summary",
    importance_table["More important"].mean(),
    importance_table["As important"].mean(),
    importance_table["Less important"].mean()
]
st.table(importance_table)
expander = st.expander("See Conclusions")
expander.write("ddd")






##########################################################################
# Correlation graph

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


##############################################################################


# Venn Diagram

st.title("Venn Diagram")
st.write("The Venn diagram below shows the overlap of mental illnesses in Malaysia:")
count = df_malaysia_uni["Depression"].value_counts()["Yes"]
depression_yes = set(df_malaysia_uni[df_malaysia_uni["Depression"] == "Yes"].index)
anxiety_yes = set(df_malaysia_uni[df_malaysia_uni["Anxiety"] == "Yes"].index)
panic_attacks_yes = set(df_malaysia_uni[df_malaysia_uni["Panic_attacks"] == "Yes"].index)
no_df = df_malaysia_uni[(df_malaysia_uni["Depression"] == "No") & (df_malaysia_uni["Anxiety"] == "No") & (df_malaysia_uni["Panic_attacks"] == "No")]
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
fig, ax = plt.subplots(figsize=(8, 8))
venn = venn3(subsets=venn_labels, set_labels=("Depression", "Anxiety", "Panic Attacks"))
for circle in venn.patches:
    circle.set_edgecolor('black')
    circle.set_linewidth(1.5)
plt.text(0.45, -0.5, f"No Mental Illness:\n{no_count}", horizontalalignment='left',
         verticalalignment='bottom', fontsize=12)
plt.title("Venn Diagram of Mental Illnesses")

# Pass the figure object 'fig' to st.pyplot()
st.pyplot(fig)
expander = st.expander("See Conclusions")
expander.write("assdaaasdsdaa")

##############################################################################

# Stacked Barplots

# Define the data and calculations (replace with your own data)
majors = df_malaysia_uni["Major"].unique()
mental_illnesses = ["Depression", "Anxiety", "Panic_attacks"]
illness_counts_all = []
illness_counts_with_illness = []

for major in majors:
    major_df = df_malaysia_uni[df_malaysia_uni["Major"] == major]
    illness_count_all = []
    illness_count_with_illness = []
    for illness in mental_illnesses:
        count_all = len(major_df[major_df[illness] == "Yes"])
        count_with_illness = count_all if count_all > 0 else 0
        illness_count_all.append(count_all)
        illness_count_with_illness.append(count_with_illness)
    illness_counts_all.append(illness_count_all)
    if any(illness_count_with_illness):
        illness_counts_with_illness.append(illness_count_with_illness)

illness_counts_all = np.array(illness_counts_all)
illness_counts_with_illness = np.array(illness_counts_with_illness)
illness_sums_all = np.sum(illness_counts_all, axis=1)
illness_sums_with_illness = np.sum(illness_counts_with_illness, axis=1)
sorted_indices_all = np.argsort(illness_sums_all)[::-1]
sorted_indices_with_illness = np.argsort(illness_sums_with_illness)[::-1]
majors_all = majors[sorted_indices_all]
majors_with_illness = majors[sorted_indices_with_illness]
illness_counts_all = illness_counts_all[sorted_indices_all]
illness_counts_with_illness = illness_counts_with_illness[sorted_indices_with_illness]

# Create the Streamlit app with tabs
st.title("Distribution of Mental Illness by Major")

# Tab 1: Every Major

st.write("Discribtion")

fig_all = plt.figure(figsize=(10, 6))
bars_all = []
bottom_all = np.zeros(len(majors_all))
for i, illness_count_all in enumerate(illness_counts_all.T):
    bar = plt.bar(np.arange(len(majors_all)), illness_count_all, bottom=bottom_all, width=0.5)
    bars_all.append(bar)
    bottom_all += illness_count_all

plt.xlabel("Major")
plt.ylabel("Count")
plt.title("Distribution of Mental Illness by Major (All Majors)")
plt.xticks(np.arange(len(majors_all)), majors_all, rotation=85)
plt.legend(bars_all, mental_illnesses)
plt.tight_layout()

# Tab 2: Only Majors with Mental Illness


fig_with_illness = plt.figure(figsize=(10, 6))
bars_with_illness = []
bottom_with_illness = np.zeros(len(majors_with_illness))
for i, illness_count_with_illness in enumerate(illness_counts_with_illness.T):
    bar = plt.bar(np.arange(len(majors_with_illness)), illness_count_with_illness, bottom=bottom_with_illness, width=0.5)
    bars_with_illness.append(bar)
    bottom_with_illness += illness_count_with_illness

plt.xlabel("Major")
plt.ylabel("Count")
plt.title("Distribution of Mental Illness by Major (Majors with at least one mental illness)")
plt.xticks(np.arange(len(majors_with_illness)), majors_with_illness, rotation=85)
plt.legend(bars_with_illness, mental_illnesses)
plt.tight_layout()


tab1, tab2 = st.tabs(["Every Major", "No Mental Illness"],)
with tab1:
    st.write(fig_all)
with tab2:
    st.write(fig_with_illness)

expander = st.expander("See Conclusions")
expander.write("ahhdha")
##############################################################################

# Boxplot

st.header('Understanding Mental Health in a University in Tokyo')

df_japan_uni = pd.read_csv(path + '\clean_data\japan_student_mental_health.csv')
df_japan_uni = df_japan_uni.query("Academic == 'Under'")

# assuming 'age' is ranging from 15 to 100
age_range = st.slider('Select age range', min_value=17, max_value=31, value=(17, 31))

# filter dataframe by selected age range
df_japan_uni = df_japan_uni[(df_japan_uni['Age'] >= age_range[0]) & (df_japan_uni['Age'] <= age_range[1])]

col1, col2 = st.columns(2)
with col1:
    International = st.checkbox("International Students", value=True)
with col2:
    Domestic = st.checkbox("Domestic student ", value=True)

if International and Domestic:
    df_japan_uni = df_japan_uni
elif Domestic:
    df_japan_uni = df_japan_uni.query("inter_dom == 'Dom'")
elif International:
    df_japan_uni =df_japan_uni.query("inter_dom == 'Inter'")
else:
    df_japan_uni = pd.DataFrame(columns = df_japan_uni.columns)  # Create an empty DataFrame with the same columns as df4

fig1 = px.box(df_japan_uni, x="Suicidal Ideation", y="Depression score", color="Gender")
fig1.update_traces(quartilemethod="exclusive")

fig2 = px.box(df_japan_uni, x="Suicidal Ideation", y="Acculturative Stress", color="Gender")
fig2.update_traces(quartilemethod="exclusive")

fig3 = px.box(df_japan_uni, x="Depression", y="Depression score", color="Gender")
fig3.update_traces(quartilemethod="exclusive")

fig4 = px.box(df_japan_uni, x="Depression", y="Acculturative Stress", color="Gender")
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



###############################################################################
# Barchart


df_japan_uni = df_japan_uni.query("Academic == 'Under'")
st.header("Comparison between two Universities with similar GDP and Happiness score")
range_of_age = st.slider('Select the age', min_value=17, max_value=29, value=(17, 29))
filtered_df1 = df_malaysia_uni[(df_malaysia_uni['Age'] >= range_of_age[0]) & (df_malaysia_uni['Age'] <= range_of_age[1])]
filtered_df2 = df_japan_uni[(df_japan_uni['Age'] >= range_of_age[0]) & (df_japan_uni['Age'] <= range_of_age[1])]

def create_bar_chart(df, x, y, color, title, y_axis_title):
    chart = alt.Chart(df).mark_bar().encode(
        x=x,
        y=alt.Y(y, axis=alt.Axis(title=y_axis_title)),
        color=color
    ).properties(
        width=250, height=250,
        title=title
    )
    return chart

bar_chart1 = create_bar_chart(filtered_df1, "Depression", "count(Depression)", alt.Color("Gender", scale=alt.Scale(domain=['Male', 'Female'], range=['#8ec7df', '#ff505a'])),"Japan University", "Count of Depression")
bar_chart2 = create_bar_chart(filtered_df2, "Depression", "count(Depression)", alt.Color("Gender", scale=alt.Scale(domain=['Male', 'Female'], range=['#8ec7df', '#ff505a'])),"Malaysia University", "Count of Depression")

combined_chart = alt.hconcat(bar_chart1, bar_chart2)
st.altair_chart(combined_chart, use_container_width=True)

expander = st.expander("See Conclusions")
expander.write("bbbb")

##############################################################################
