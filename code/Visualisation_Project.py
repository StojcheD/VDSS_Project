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

# Import the data

path = os.getcwd()
df_japan_uni = pd.read_csv(path + "\..\\clean_data\japan_student_mental_health.csv")
df_global_mental_health = pd.read_csv(path + "\..\\clean_data\cleaned_data_global_mental_health.csv")
df_malaysia_uni = pd.read_csv(path + "\..\\clean_data\malaysia_clean_mental_health.csv")


def get_data():
    with open(
            path + "\..\\clean_data\countries.geojson") as response:
        geo = json.load(response)

    return geo ##, Countries, happines_data


geo = get_data()


# function to create a world map with happiness data
# can be changed to see diffrent years, social score, economy score, etc.

def generate_happiness_map():
    years = ["2015", "2016","2017","2018","2019","2020","2021"]
    happines_data = {}
    st.header('Happiness around the world')
    st.write('Inspecting happiness or well-being correlating with country wealth.')

    labels = ["Country"]
    tooltips = ["ADMIN"]

    selectedYear = st.selectbox('Select the year for Happiness data', years)
    st.write("The data is from the year:", selectedYear)
    happines_data[selectedYear] = pd.read_csv(path + f"\..\\clean_data\{selectedYear}.csv")

    countries = happines_data[selectedYear]["Country"].unique()
    df = happines_data[selectedYear]
    df_new = df.set_index("Country")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        happinesScore = st.checkbox("Happiness Score", value=True)
    with col2:
        economyScore = st.checkbox("Economy Score", value=True)
    with col3:
        socialSupport = st.checkbox("Social Support", value=True)
    with col4:
        Freedom = st.checkbox("Freedom to make life choices", value=True)

    if happinesScore:
        labels.append("Happiness Score")
        tooltips.append("happinessstatus")
        for feature in geo["features"]:
            if feature["properties"]["ADMIN"] in countries:
                 feature["properties"]["happinessstatus"] = f"Happiness score {selectedYear}: {df_new.loc[feature['properties']['ADMIN']]['Happiness Score']}"
            else:
                feature["properties"]["happinessstatus"] = "No happiness Data"

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

    map = folium.Map(location=[45, 0], zoom_start=1.3, scrollWheelZoom=False, tiles='CartoDB positron')

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
        legend_name="Hapiness Score"
    )
    choropleth.add_to(map)

    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(tooltips, labels=False)
    )

    choropleth.geojson.add_to(map)

    st_map = st_folium(map, width=700, height=450)

    if st_map['last_active_drawing']:
        country_name = st_map['last_active_drawing']['properties']['ADMIN']
    else:
        country_name = ""

    st.caption("Source:")
    expander = st.expander("Conclusions")
    expander.write("The countries with a higher economic score are more happy than the countries with low scores. "
                   "The freedom of choice and social support has a mixed effect on the happiness index depending on the country."
                   " The map shows the countries happiness score with economic, social support and freedom to make life choices score alongside the income for 7 years."
                   "The top five positions in the happiness score are taken up by the Nordic countries throughout the seven years with Switzerland, a non-Nordic country, also taking top places in these years."
                   " However, the country's GDP doesn't guarantee happiness for all, as is the case in India, where the GDP score is among the highest while its happiness score is among the last 20."
                   " This is most probably due to the unequal distribution of wealth. Afghanistan, due to prolonged wars, remained in the lowest ranks in the happiness index.")


# function to create a treemap

def generate_treemap():
    st.header('Mental Health and Its Impact on Life Expectancy')
    st.write('Investigating the potential correlation between mental health conditions and longevity across various global populations.')



    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    data = {}

    # create a slider to select the year that will be shown
    selectedYear = st.select_slider("Choose a Year", options=years)
    data[selectedYear] = pd.read_csv(path + f"\..\/clean_data/{selectedYear}.csv")
    df = data[selectedYear]
    st.write("The data is from the year:", selectedYear)

    fig = px.treemap(df, path=[px.Constant('World'), 'Region', 'Country'], values='Happiness Score',
                     color='Health (Life Expectancy)', hover_data=['Happiness Rank'],
                     color_continuous_scale='RdBu', branchvalues='total')
    fig.update_layout(margin=dict(t=5, l=2.5, r=2.5, b=2.5))

    st.plotly_chart(fig, theme=None, use_container_width=True)

    st.caption('Source of the graph')

    expander = st.expander("See Conclusions")
    expander.write("The happiness score and the life expectancy is corelated for most countries."
                   " Western Europe has the highest happiness score while at the same time it also has the highest life expectancy."
                   " Sub-Saharan Africa excluding Mauritius has lowest life expectancy with one of the lowest happiness score. "
                   "There is little to no change in life expectancy for the considered period of time.Western Europe and Eastern Asia has the highest life expectancy in the world."
                   "Both regions are technologically advance thus have similar life conditions. Middle East and Northern Africa have some countries with high life expectancy. "
                   "Iraq, Yemen, Haiti,"
                   "  shows a negative trend in life expectancy from 2016 to 2021.  However, Latin America countries shows a positive trend in life expectancy from 2016 to 2021.")


# function to create a table about the importance of menatl health in the world

def create_importance_table(df):
    st.title("Regional Analysis of Importance Ratings")
    st.write("The following table provides a detailed percentage breakdown of 'importance' ratings across different regions. Each rating represents how respondents have ranked a particular factor in their region.")
    importance_counts = df["Importance"].value_counts(normalize=True) * 100
    regions = sorted(df["Region"].unique())
    importance_table = pd.DataFrame({"Region": regions})
    columns = ["Region", "More important", "As important", "Less important"]
    for column in columns[1:]:
        importance_table[column] = 0

    for region in regions:
        region_df = df[df["Region"] == region]
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
    expander.write("The regions that have seen war in the last century or in the current takes mental health as more important than the more peaceful regions. "
                   "However, most of the countries does take it as important even if its not more important."
                   " South Asia, has the highest score regarding low importance of mental health, followed by Sub-Saharan Africa. "
                   "American continent has the lowest score in considering the mental health less important,"
                   " thus making it two regions with people considering mental health as either more important or as important. ")


# function for the heatmap 

def plot_correlation_heatmap():
    st.header("Exploring the Correlation Among Education, Income and Perception of Mental Health")
    st.write("This heatmap depicts the correlation between various factors such as level of education, income category and self-experience of mental health.")

    df1 = pd.read_csv(path + '\..\\clean_data\cleaned_data_global_mental_health.csv', usecols=[2, 3, 7, 8])
    df1.drop(df1[df1['Education'] == '99'].index, inplace=True)
    df1.drop(df1[df1['Self_experience'] == ' '].index, inplace=True)
    mapping_dict = {'Elementary or less': 0, 'Secondary': 1, 'Tertiary': 2, 'High income': 3, 'Upper-middle income': 2,
                    'Lower-middle income': 1, 'Low income': 0, 'Less important': 0, 'As important': 1,
                    'More important': 2, 'Yes': 1, 'No': 0}
    df1 = df1.replace(mapping_dict)
    df_corr = df1.corr()

    fig, ax = plt.subplots()
    sns.heatmap(df_corr, ax=ax, cmap="Blues", annot=True)
    st.write(fig)

    st.caption('Source: https://wellcome.org/reports/wellcome-global-monitor-covid-19/2020')

    expander = st.expander("See Conclusions")
    expander.write("Higher education leads to higher country wealth, however, interestingly, higher education doesn't provide understanding of mental health importance. "
                   "On the contrary, it has a negative effect. Also, higher education leads to lower mental health self-experiences."
                   "The mental health importance is correlated with people's self-experience. The more important the mental health is,"
                   "the more people are going to self-experience it."
                   "However, both these indices are negatively correlated with country wealth and education."
                   "While country wealth and education are both correlated among themselves.")


# function for a vann-diagramm to show how many have which mental illness 
# in the malaysia dataset

def plot_venn_diagram(df_malaysia_uni):
    st.title("Intersection of Mental Health Conditions in Malaysia's University Students")
    st.write("The Venn diagram provides a graphical representation of the overlap between depression, anxiety, and panic attacks among Malaysian university students. Each circle in the diagram represents one of the conditions. The areas where the circles intersect illustrate the students dealing with multiple conditions simultaneously.")

    count = df_malaysia_uni["Depression"].value_counts()["Yes"]
    depression_yes = set(df_malaysia_uni[df_malaysia_uni["Depression"] == "Yes"].index)
    anxiety_yes = set(df_malaysia_uni[df_malaysia_uni["Anxiety"] == "Yes"].index)
    panic_attacks_yes = set(df_malaysia_uni[df_malaysia_uni["Panic_attacks"] == "Yes"].index)
    no_df = df_malaysia_uni[(df_malaysia_uni["Depression"] == "No") & (df_malaysia_uni["Anxiety"] == "No") & (
                df_malaysia_uni["Panic_attacks"] == "No")]
    no_count = len(no_df)

    # create the diffrent parts of the diagramm and fills it with the
    # respective number from the dataset
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

    st.pyplot(fig)

    expander = st.expander("See Conclusions")
    expander.write("The Venn diagram sheds light on the status of depression, anxiety and panic attacks and how they are related. "
                   "Depression is related to both anxiety and panic attacks while the relationship between anxiety and panic attacks is comparably not stronger. "
                   "However, the combined relationship among all the three factors is a stronger. "
                   "That means a good number of respondents had anxiety leading to depression and panic attack.")


# function to create two stacked barplots, one with all majors and one without
# those who have no menatl illnesses, from the malaysian dataset

def plot_stacked_barplots(df_malaysia_uni):
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

    st.title("Prevalence of Mental Illnesses across Different University Majors")

    st.write("This section provides two stacked bar plots indicating the distribution of depression, anxiety, and panic attacks among students pursuing different majors."
             "The first plot covers all majors, whereas the second focuses only on those majors where at least one instance of mental illness has been reported."
             "This breakdown can offer insights into the relationship between study disciplines and mental health conditions.")

    # creating the one with all majors
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

    # creating the one only with those majors with menatl illnesses
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
        st.pyplot(fig_all)
    with tab2:
        st.pyplot(fig_with_illness)

    expander = st.expander("See Conclusions")
    expander.write("The distribution shows the mental illness by major of the respondents of the survey.  "
                   "Students with major in STEM courses have a higher rate of mental illness as compared to the other majors."
                   " BIT, Engineering, and Computer science are the three leading majors with higher rate of mental illness."
                   " It is common to have panic attacks as a student due to the exam pressure or stress however, "
                   "the STEM students face depression and anxiety alongside panic attacks. ")




def visualize_mental_health_data():
    st.header('Understanding Mental Health among University Students in Tokyo')
    st.write("In this section, we are analyzing mental health data from a university in Tokyo, focusing on variables such as age,"
             "student type (international or domestic), and gender. Users can manipulate the age range slider to narrow down the"
             "demographic. The series of box plots offer a detailed comparison of depression scores and acculturative stress levels"
             "against the presence of suicidal ideation and depression, segregated by gender.")

    df_japan_uni = pd.read_csv(path + '\..\/clean_data/japan_student_mental_health.csv')
    df_japan_uni = df_japan_uni.query("Academic == 'Under'")

    # create a slider to select the range of the shown age
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
        df_japan_uni = df_japan_uni.query("inter_dom == 'Inter'")
    else:
        df_japan_uni = pd.DataFrame(columns=df_japan_uni.columns)

    # create the diffrent figures to later show in diffrent tabs
    fig1 = px.box(df_japan_uni, x="Suicidal Ideation", y="Depression score", color="Gender")
    fig1.update_traces(quartilemethod="exclusive")

    fig2 = px.box(df_japan_uni, x="Suicidal Ideation", y="Acculturative Stress", color="Gender")
    fig2.update_traces(quartilemethod="exclusive")

    fig3 = px.box(df_japan_uni, x="Depression", y="Depression score", color="Gender")
    fig3.update_traces(quartilemethod="exclusive")

    fig4 = px.box(df_japan_uni, x="Depression", y="Acculturative Stress", color="Gender")
    fig4.update_traces(quartilemethod="exclusive")

    # create the diffrent tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Suicide vs ToDep", "Suicide vs ToAS", "Depression vs ToDep", "Depression vs ToAS"],)
    with tab1:
        st.write(fig1)
    with tab2:
        st.write(fig2)
    with tab3:
        st.write(fig3)
    with tab4:
        st.write(fig4)

    expander = st.expander("See Conclusions")
    expander.write("The boxplot shows the suicide and depression against total depression score and total acculturative stress. "
                   "For the complete age range i.e., from 17 to 31, in international students females have less suicidal thoughts even with a higher median score of depression as compare to male students,"
                   " the same is true for domestic students as well.For suicide vs total acculturative stress, the median for the complete age range for male and females with no suicidal thoughts, the median lie very close to each other, however, the median for female students with suicidal thoughts is a bit high in acculturative stress."
                   " However, for the domestic students the score is highly variable for both suicidal and no suicidal thoughts for both genders.For depression vs total acculturative stress (ToAS) international females students feel depressed with higher median for acculturative stress while for male students with a postive response to depression question, the median is a bit low."
                   " However, the variability in both genders remain there. On the other hand students with no depression have a comparatively lower median of ToAS for both male and female students, only the female students data shows higher variability. "
                   "The domestic students with positive response for depression shows highly variable response from male students while for the female students the variability is lower. However, for the negative responders the median for both male and female students are close to each other with little variation in the responders score.")


# create a barplot to compare the japanese data to the malaysian dataset

def compare_universities():
    df_japan_uni = pd.read_csv(path + '\..\/clean_data/japan_student_mental_health.csv')
    df_japan_uni = df_japan_uni.query("Academic == 'Under'")

    st.header("Comparison between two Universities with similar GDP and Happiness score")
    st.write("This section provides a comparative study of depression rates between two universities located in Japan and Malaysia,"
             "countries having similar GDP and Happiness Index. By adjusting the age range, the user can explore the distribution of"
             "depression instances in the chosen demographic across both universities. Separate bar charts for each university,"
             "color-coded by gender, are presented side-by-side for a direct comparison.")
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
    expander.write("The bar plot shows the data from two countries with much the same happiness score and GDP. The students in the university in Japan shows the higher number of females with no depression and a significant number of male students as well."
                   " However, for respondents with answer yes for the depression question. the number of female students remain high as well while the number of male students is very low. Same is the case for Malaysia except the respondents with answer yes,"
                   " the number of male students is higher than the students in Japan.However for age range 17 to 22 the statistics follow the general trend however for the students age 22 to 29 (most probably post-graduate students)"
                   " more females students responded yes to the question of depression, while the number of male students with a positive response to the same question grew in Malaysia, meaning more male students were having the signs of depression.  ")


# main function to put all graphs together and name the project
def main():
    geo = get_data()


    APP_TITLE = 'Thoughts around the world'
    #st.set_page_config(APP_TITLE)
    st.set_page_config(
        page_title="Visualisation Project",
        page_icon=":bar_chart:",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "This is a comprehensive visual exploration project, focusing on global mental health trends. Using a multitude of visualizations and interactive features, the project aims to provide insights into the mental health landscape across various demographic groups and geographical regions."
        }
    )
    st.markdown('<h1 style="color: black;">Thoughts around the world</h1>', unsafe_allow_html=True)
    #st.title(APP_TITLE)
    st.subheader("A visual exploration of global mental health")
    generate_happiness_map()
    generate_treemap()
    create_importance_table(df_global_mental_health)
    plot_correlation_heatmap()
    plot_venn_diagram(df_malaysia_uni)
    plot_stacked_barplots(df_malaysia_uni)
    visualize_mental_health_data()
    compare_universities()


if __name__ == '__main__':
    main()