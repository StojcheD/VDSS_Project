# Markdown zu VDSS
### Im folgenden wird der Code beschrieben, mit welchem die Streamlit-App gebaut wurde. Dieser kann wie folgt ausgeführt werden:

Zuerst muss das Working Directory  geändert werden, um zum entsperchenden Ordner zu gelangen:

"cd C:\...\VDSS_Visualisierungsprojekt\code"

Dann kann ganz normal die App gestertet werden mit:

streamlit run Visualisation_Project.py

Importieren der python librarys
'''
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
'''
