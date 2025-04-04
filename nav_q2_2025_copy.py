# =================================== IMPORTS ================================= #
import csv, sqlite3
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import plotly.figure_factory as ff
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from folium.plugins import MousePosition
import plotly.express as px
from datetime import datetime, timedelta
import folium
import os
import sys
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.development.base_component import Component
# 'data/~$bmhc_data_2024_cleaned.xlsx'
# print('System Version:', sys.version)
# -------------------------------------- DATA ------------------------------------------- #

current_dir = os.getcwd()
current_file = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = 'data/Navigation_Responses.xlsx'
file_path = os.path.join(script_dir, data_path)
data = pd.read_excel(file_path)
df = data.copy()

df.columns = df.columns.str.strip()

# Create new database named 'df_q1' that only includes columns where "Date of Activity:" is between 202-10-01 and 2024-12-31:
df['Date of activity:'] = pd.to_datetime(df['Date of activity:'], format='%m-%d-%Y', errors='coerce')

df = df[(df['Date of activity:'] >= '2024-10-01') & (df['Date of activity:'] <= '2024-12-31')]

# Define a discrete color sequence
color_sequence = px.colors.qualitative.Plotly

# print(df.head())
# print('Total entries: ', len(df))
# print('Column Names: \n', df_q1.columns)
# print('DF Shape:', df.shape)
# print('Dtypes: \n', df.dtypes)
# print('Info:', df.info())
# print("Amount of duplicate rows:", df.duplicated().sum())

# print('Current Directory:', current_dir)
# print('Script Directory:', script_dir)
# print('Path to data:',file_path)

# ================================= Columns Navigation ================================= #

# Column Names: 
#  Index([
#     'Timestamp', 
#     'Date of activity:', 
#     'Person filling out this form:',
#     'Activity duration (minutes):', 
#     'Location Encountered:',
#     'Individual's Name:', 
#     'Individual's Date of Birth:',
#     'Individual's Insurance Status:', 
#     'Individual's street address:',
#     'City:', 
#     'ZIP Code:', 
#     'County:', 
#     'Type of support given:',
#     'Provide brief support description:', 
#     'Individual's Status:',
#     'HMIS SPID Number:', 
#     'MAP Card Number', 
#     'Gender:', 
#     'Race/Ethnicity:'
# ],
# dtype='object')

# ================================= Columns FH ================================= #

# Column Names: 
#  Index([
#         'screener_id', 
#         'site',
#         'program', 
#         'submitted_by', 
#         'submitted_for',
#         'created_at', 
#         'seeker_name',
#         'Status', 
#         'referral_id',
#         'Are you Hispanic Latino/a or of Spanish origin?',
#         'Are you a veteran of the US Armed Forces?', 
#         'Birthdate', 
#         'Client age',
#         'Do you utilize any of these benefits or social services? Check all that       apply:',
#         'Email Address', 
#         'First Name',
#         'Gender',
#         'How can BMHC support you today?',
#         'Last Name', 'Phone Number',
#         'Preferred Language', 
#         'Race/Ethnicity',
#         'What is your living situation?',
#         'ZIP Code'],
#       dtype='object')

# ================================= Missing Values ================================= #

# missing = df_q1.isnull().sum()
# print('Columns with missing values before fillna: \n', missing[missing > 0])

# Missing Values:

# Individual's Insurance Status:      1
# Individual's street address:        2
# City:                               1
# ZIP Code:                           2
# County:                             3
# HMIS SPID Number:                 154
# MAP Card Number                   158                                26

# ============================== Checking Duplicates ========================== #

# Check for duplicate columns
# duplicate_columns = df.columns[df.columns.duplicated()].tolist()
# print(f"Duplicate columns found: {duplicate_columns}")
# if duplicate_columns:
#     print(f"Duplicate columns found: {duplicate_columns}")

# Check for duplicate rows
# duplicate_rows = df[df.duplicated(subset=['First Name'], keep=False)]
# print(f"Duplicate rows found: \n {duplicate_rows}")

# Get the count of duplicate 'First Name' values
# duplicate_count = df['First Name'].duplicated(keep=False).sum()

# Print the count of duplicate 'First Name' values
# print(f"Count of duplicate 'First Name' values: {duplicate_count}")

# Remove duplicate rows based on 'First Name'
# df = df.drop_duplicates(subset=['First Name'], keep='first')

# Verify removal of duplicates
# duplicate_count_after_removal = df['First Name'].duplicated(keep=False).sum()
# print(f"Count of duplicate 'First Name' values after removal: {duplicate_count_after_removal}")

# # Merge the "Zip Code" columns if they exist
# if 'Zip Code' in df.columns:
#     zip_code_columns = [col for col in df.columns if col == 'Zip Code']
#     if len(zip_code_columns) > 1:
#         df['Zip Code'] = df[zip_code_columns[0]].combine_first(df[zip_code_columns[1]])
#         # Drop the duplicate "Zip Code" columns, keeping only the merged one
#         df = df.loc[:, ~df.columns.duplicated()]

# ============================== Data Preprocessing ========================== #

# Check for duplicate columns
# duplicate_columns = df.columns[df.columns.duplicated()].tolist()
# print(f"Duplicate columns found: {duplicate_columns}")
# if duplicate_columns:
#     print(f"Duplicate columns found: {duplicate_columns}")

# Fill Missing Values for df
# List of columns to fill with 'Unknown'
columns_to_fill = [
    "Individual's Name:",
    "Individual's Insurance Status:",
    "City:",
    "County:",
    "Gender:",
    "Race/Ethnicity:"
]

# Fill missing values for categorical columns with 'Unknown'
for column in columns_to_fill:
    if column in df.columns:
        df[column] = df[column].fillna('Unknown')

# Fill missing values for numerical columns with a specific value (e.g., -1)
df['HMIS SPID Number:'] = df['HMIS SPID Number:'].fillna(-1)
df['MAP Card Number'] = df['MAP Card Number'].fillna(-1)

# fill missing values for "City:" with the most frequent value
df['City:'] = df['City:'].fillna(df['City:'].mode()[0])
df['County:'] = df['County:'].fillna(df['County:'].mode()[0])
df["Individual's street address:"] = df["Individual's street address:"].fillna('Unknown')

# -------------------------- Clients Served DF ------------------------- #

clients_served = len(df)
# print('Clients Served This Month:', patients_served)

# -------------------------- Activity Duration DF ------------------------- #

df_duration = df['Activity duration (minutes):'].sum()/60
# print('Activity Duration:', df_duration/60, 'hours')

# -------------------- Age DF ------------------- #

# Fill missing values for 'Birthdate' with random dates within a specified range
def random_date(start, end):
    return start + timedelta(days=np.random.randint(0, (end - start).days))

start_date = datetime(1950, 1, 1) # Example: start date, e.g., 1950-01-01
end_date = datetime(2000, 12, 31)

# Convert 'Individual's Date of Birth:' to datetime, coercing errors to NaT
df['Individual\'s Date of Birth:'] = pd.to_datetime(df['Individual\'s Date of Birth:'], errors='coerce')

# Fill missing values in 'Individual's Date of Birth:' with random dates
df['Individual\'s Date of Birth:'] = df['Individual\'s Date of Birth:'].apply(
    lambda x: random_date(start_date, end_date) if pd.isna(x) else x
)

# Calculate 'Client Age' by subtracting the birth year from the current year
df['Client Age'] = pd.to_datetime('today').year - df['Individual\'s Date of Birth:'].dt.year

# Handle NaT values in 'Client Age' if necessary (e.g., fill with a default value or drop rows)
df['Client Age'] = df['Client Age'].apply(lambda x: "N/A" if x < 0 else x)

# Define a function to categorize ages into age groups
def categorize_age(age):
    if age == "N/A":
        return "N/A"
    elif 10 <= age <= 19:
        return '10-19'
    elif 20 <= age <= 29:
        return '20-29'
    elif 30 <= age <= 39:
        return '30-39'
    elif 40 <= age <= 49:
        return '40-49'
    elif 50 <= age <= 59:
        return '50-59'
    elif 60 <= age <= 69:
        return '60-69'
    elif 70 <= age <= 79:
        return '70-79'
    else:
        return '80+'

# Apply the function to create the 'Age_Group' column
df['Age_Group'] = df['Client Age'].apply(categorize_age)

# Group by 'Age_Group' and count the number of patient visits
df_decades = df.groupby('Age_Group').size().reset_index(name='Patient_Visits')

# Sort the result by the minimum age in each group
age_order = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df_decades['Age_Group'] = pd.Categorical(df_decades['Age_Group'], categories=age_order, ordered=True)
df_decades = df_decades.sort_values('Age_Group')
# print(df_decades.value_counts())

# -------------------------- Insurance Status DF ------------------------- #

# Group by 'Individual's Insurance Status:' and count the number of occurrences
df["Individual's Insurance Status:"] = df["Individual's Insurance Status:"].str.strip()
df["Individual's Insurance Status:"] = df["Individual's Insurance Status:"].replace('MAP 000', 'MAP 100')
df["Individual's Insurance Status:"] = df["Individual's Insurance Status:"].replace('Did not disclose.', 'NONE')
df_insurance = df.groupby("Individual's Insurance Status:").size().reset_index(name='Count')
# print(df_q1["Individual's Insurance Status:"].value_counts())

# ------------------------ Location Encountered DF ----------------------- #

# "Location Encountered:" dataframe:
df['Location Encountered:'] = df['Location Encountered:'].str.strip()
df['Location Encountered:'] = df['Location Encountered:'].replace('Southbridge', 'SouthBridge')
df['Location Encountered:'] = df['Location Encountered:'].replace('The Bumgalows', 'The Bungalows')
df['Location Encountered:'] = df['Location Encountered:'].replace('DACC', 'Downtown Austin Community Court')
df['Location Encountered:'] = df['Location Encountered:'].replace('SNHC', 'Sunrise Navigation Homeless Center')
df['Location Encountered:'] = df['Location Encountered:'].replace('HATC', 'Housing Authority of Travis County')
df['Location Encountered:'] = df['Location Encountered:'].replace('CFV', 'Community First Village')
df['Location Encountered:'] = df['Location Encountered:'].replace('BMHC', 'Black Men\'s Health Clinic')
df_location = df['Location Encountered:'].value_counts().reset_index(name='Count')
# print(df_q1['Location Encountered:'].value_counts())

# ------------------------ Type of Support Given DF --------------------------- #

# DataFrame for columns "Type of support given:" and "Date of activity:"
df_support = df[['Type of support given:', 'Date of activity:']]

# # 'How can BMHC support you today?'
# df_support = df_support['Type of support given:'].value_counts().reset_index(name='Count')

# # Extract the month from the 'Date of activity:' column
df_support['Month'] = df_support['Date of activity:'].dt.month_name()

# # Filter data for October, November, and December
df_support_q = df_support[df_support['Month'].isin(['October', 'November', 'December'])]

# Group the data by 'Month' and 'Type of support given:' to count occurrences
df_support_counts = (
    df_support_q.groupby(['Month', 'Type of support given:'])
    .size()
    .reset_index(name='Count')
)

# Sort months in the desired order
month_order = ['October', 'November', 'December']

df_support_counts['Month'] = pd.Categorical(
    df_support_counts['Month'], 
    categories = month_order, 
    ordered=True) # pd.Categorical is to specify the order of the categories for sorting purposes and to avoid alphabetical sorting

# Print the value counts
# print(df_service_counts)

# Filter data for October, November, and December
# df_october = df_support[df_support['Month'] == 'October']
# df_november = df_support[df_support['Month'] == 'November']
# df_december = df_support[df_support['Month'] == 'December']

# # Combine the data into a single DataFrame
# df_quarterly = pd.concat([df_october, df_november, df_december])

# # Get value counts for 'Type of support given:'
# df_support_counts = df_quarterly['Type of support given:'].value_counts().reset_index()
# df_support_counts.columns = ['Type of support given:', 'Count']

# Make a copy of the bar chart below:
support_fig = px.bar(
    df_support_counts,
    x='Month',
    y='Count',
    color='Type of support given:',
    barmode='group',
    text='Count',
    title='Type of Service Given',
    labels={
        'Count': 'Number of Services',
        'Month': 'Month',
        'Type of service given:': 'Type of Service'
    },
).update_layout(
    title_x=0.5,
    xaxis_title='Month',
    yaxis_title='Count',
    height=900,  # Adjust graph height
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    xaxis=dict(
        tickmode='array',
        tickvals=df_support_counts['Month'].unique(),
        tickangle=-35
    ),
    legend=dict(
        title='Type of Service Given',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left", # Anchor legend to the left
        y=1, # Position legend at the top
        yanchor="top" # Anchor legend at the top
).update_traces(
    textposition='auto',  # Position the text outside the bars
    hovertemplate=
    '<b>Month</b>: %{x}<br><b>Count</b>: %{y}<br><b>Type of Service</b>: %{color}<extra></extra>'
)
)


# ------------------------------- New/ Returning Stattus DF ----------------------------- #

# "Individual's Status:" dataframe:
df_status = df['Individual\'s Status:'].value_counts().reset_index(name='Count')

# -------------------------- Person Filling Out This Form DF ------------------------- #

# Cleaning up the 'Person filling out this form:' column
df['Person filling out this form:'] = df['Person filling out this form:'].str.strip()
df['Person filling out this form:'] = df['Person filling out this form:'].replace('Dominique', 'Dominique Street')
df['Person filling out this form:'] = df['Person filling out this form:'].replace('Jaqueline Ovieod', 'Jaqueline Oviedo')
df['Person filling out this form:'] = df['Person filling out this form:'].replace('Sonya', 'Sonya Hosey')

# Groupby Person filling out this form:
person_filling = df['Person filling out this form:'].value_counts().reset_index(name='Count')

# -------------------------- Race/ Ethnicity DF ------------------------- #

# Groupby Race/Ethnicity:
# strip whitespace from 'Race/Ethnixity:' column
df['Race/Ethnicity:'] = df['Race/Ethnicity:'].str.strip()
df['Race/Ethnicity:'] = df['Race/Ethnicity:'].replace('Hispanic/Latino', 'Hispanic/ Latino')
df['Race/Ethnicity:'] = df['Race/Ethnicity:'].replace('White', 'White/ European Ancestry')
df_race = df['Race/Ethnicity:'].value_counts().reset_index(name='Count')

# ------------------------ ZIP2 DF ----------------------- #

# make a copy of 'ZIP Code:' column to 'ZIP2':
df['ZIP2'] = df['ZIP Code:']
df['ZIP2'] = df['ZIP2'].astype(str)
df['ZIP2'] = df['ZIP2'].fillna(df['ZIP2'].mode()[0])
df_z = df['ZIP2'].value_counts().reset_index(name='Count')
# print(df_z.value_counts())

# =============================== Distinct Values ========================== #

# Get the distinct values in column

# distinct_service = df['What service did/did not complete?'].unique()
# print('Distinct:\n', distinct_service)

# ==================================== Folium =================================== #

mode_value = df['ZIP Code:'].mode()[0]
df['ZIP Code:'] = df['ZIP Code:'].replace("UNHOUSED", mode_value)
# df_q1['ZIP Code:'].fillna(df_q1['ZIP Code:'].mode()[0], inplace=True)
df['ZIP Code:'].fillna(mode_value, inplace=True)
df['ZIP Code:'] = df['ZIP Code:'].astype('Int64')
df['ZIP Code:'] = df['ZIP Code:'].replace(-1, df['ZIP Code:'].mode()[0])

# Count of visitors by zip code
df_zip = df['ZIP Code:'].value_counts().reset_index(name='Residents')
df_zip['ZIP Code:'] = df_zip['ZIP Code:'].astype(int)
df_zip['Residents'] = df_zip['Residents'].astype(int)

# Create a folium map
m = folium.Map([30.2672, -97.7431], zoom_start=10)

# Add different tile sets
folium.TileLayer('OpenStreetMap', attr='© OpenStreetMap contributors').add_to(m)
folium.TileLayer('Stamen Terrain', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
folium.TileLayer('Stamen Toner', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
folium.TileLayer('Stamen Watercolor', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
folium.TileLayer('CartoDB positron', attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
folium.TileLayer('CartoDB dark_matter', attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)

# Available map styles
map_styles = {
    'OpenStreetMap': {
        'tiles': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    },
    'Stamen Terrain': {
        'tiles': 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
        'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
    },
    'Stamen Toner': {
        'tiles': 'https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
        'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
    },
    'Stamen Watercolor': {
        'tiles': 'https://stamen-tiles.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg',
        'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
    },
    'CartoDB positron': {
        'tiles': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    },
    'CartoDB dark_matter': {
        'tiles': 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
    },
    'ESRI Imagery': {
        'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        'attribution': 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
    }
}

# Add tile layers to the map
for style, info in map_styles.items():
    folium.TileLayer(tiles=info['tiles'], attr=info['attribution'], name=style).add_to(m)

# Select a style
# selected_style = 'OpenStreetMap'
# selected_style = 'Stamen Terrain'
# selected_style = 'Stamen Toner'
# selected_style = 'Stamen Watercolor'
selected_style = 'CartoDB positron'
# selected_style = 'CartoDB dark_matter'
# selected_style = 'ESRI Imagery'

# Apply the selected style
if selected_style in map_styles:
    style_info = map_styles[selected_style]
    # print(f"Selected style: {selected_style}")
    folium.TileLayer(
        tiles=style_info['tiles'],
        attr=style_info['attribution'],
        name=selected_style
    ).add_to(m)
else:
    print(f"Selected style '{selected_style}' is not in the map styles dictionary.")
     # Fallback to a default style
    folium.TileLayer('OpenStreetMap').add_to(m)

# Function to get coordinates from zip code
def get_coordinates(zip_code):
    geolocator = Nominatim(user_agent="response_q4_2024.py")
    location = geolocator.geocode({"postalcode": zip_code, "country": "USA"})
    if location:
        return location.latitude, location.longitude
    else:
        print(f"Could not find coordinates for zip code: {zip_code}")
        return None, None

# Apply function to dataframe to get coordinates
df_zip['Latitude'], df_zip['Longitude'] = zip(*df_zip['ZIP Code:'].apply(get_coordinates))

# Filter out rows with NaN coordinates
df_zip = df_zip.dropna(subset=['Latitude', 'Longitude'])
# print(df_zip.head())
# print(df_zip[['Zip Code', 'Latitude', 'Longitude']].head())
# print(df_zip.isnull().sum())

# instantiate a feature group for the incidents in the dataframe
incidents = folium.map.FeatureGroup()

for index, row in df_zip.iterrows():
    lat, lng = row['Latitude'], row['Longitude']

    if pd.notna(lat) and pd.notna(lng):  
        incidents.add_child(# Check if both latitude and longitude are not NaN
        folium.vector_layers.CircleMarker(
            location=[lat, lng],
            radius=row['Residents'] * 1.2,  # Adjust the multiplication factor to scale the circle size as needed,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.4
        ))

# add pop-up text to each marker on the map
latitudes = list(df_zip['Latitude'])
longitudes = list(df_zip['Longitude'])

# labels = list(df_zip[['Zip Code', 'Residents_In_Zip_Code']])
labels = df_zip.apply(lambda row: f"ZIP Code: {row['ZIP Code:']}, Patients: {row['Residents']}", axis=1)

for lat, lng, label in zip(latitudes, longitudes, labels):
    if pd.notna(lat) and pd.notna(lng):
        folium.Marker([lat, lng], popup=label).add_to(m)
 
formatter = "function(num) {return L.Util.formatNum(num, 5);};"
mouse_position = MousePosition(
    position='topright',
    separator=' Long: ',
    empty_string='NaN',
    lng_first=False,
    num_digits=20,
    prefix='Lat:',
    lat_formatter=formatter,
    lng_formatter=formatter,
)

m.add_child(mouse_position)

# add incidents to map
m.add_child(incidents)

map_path = 'zip_code_map.html'
map_file = os.path.join(script_dir, map_path)
m.save(map_file)
map_html = open(map_file, 'r').read()

# ========================== DataFrame Table ========================== #

df_table = go.Figure(data=[go.Table(
    # columnwidth=[50, 50, 50],  # Adjust the width of the columns
    header=dict(
        values=list(df.columns),
        fill_color='paleturquoise',
        align='left',
        height=30,  # Adjust the height of the header cells
        # line=dict(color='black', width=1),  # Add border to header cells
        font=dict(size=12)  # Adjust font size
    ),
    cells=dict(
        values=[df[col] for col in df.columns],
        fill_color='lavender',
        align='left',
        height=25,  # Adjust the height of the cells
        # line=dict(color='black', width=1),  # Add border to cells
        font=dict(size=12)  # Adjust font size
    )
)])

df_table.update_layout(
    margin=dict(l=50, r=50, t=30, b=40),  # Remove margins
    height=400,
    # width=1500,  # Set a smaller width to make columns thinner
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)'  # Transparent plot area
)

# print(df.head())

# ----------------------------------- DASHBOARD -----------------------------------

app = dash.Dash(__name__)
server= app.server

app.layout = html.Div(
    children=[ 
        html.Div(
            className='divv', 
            children=[ 
            html.H1(
                'BMHC Quarterly Report Q1 2025', 
                className='title'),
            html.H1(
                '10/01/2025 - 12/31/2025', 
                className='title2'),

            html.Div(
                className='btn-box', 
                children=[
                    html.A(
                        'Repo',
                        href='https://github.com/CxLos/BMHC_Q1_2025_Responses',
                        className='btn'),
                ]),
    ]),  

# Data Table
# html.Div(
#     className='row0',
#     children=[
#         html.Div(
#             className='table',
#             children=[
#                 html.H1(
#                     className='table-title',
#                     children='Data Table'
#                 )
#             ]
#         ),
#         html.Div(
#             className='table2', 
#             children=[
#                 dcc.Graph(
#                     className='data',
#                     figure=df_table
#                 )
#             ]
#         )
#     ]
# ),

# ROW 1
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph33',
            children=[
                html.Div(
                    className='high1',
                    children=['Clients Serviced:']
                ),
                html.Div(
                    className='circle',
                    children=[
                        html.Div(
                            className='hilite',
                            children=[
                                html.H1(
                                    className='high2',
                                    children=[clients_served]
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ]
),

# ROW 2
html.Div(
    className='row2',
    children=[
        html.Div(
            className='graph3',
            children=[
                # Race Distribution
                dcc.Graph(
                    id='race-graph',
                    figure=px.pie(
                        df_race,
                        names='Race/Ethnicity:',
                        values='Count'
                    ).update_layout(
                        title='Patient Visits by Race',
                        title_x=0.5,
                        font=dict(
                            family='Calibri',
                            size=17,
                            color='black'
                        )
                    ).update_traces(
                        textinfo='label+percent',
                        hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
                    )
                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                # "Individual's Insurance Status:" bar chart
                dcc.Graph(
                    id='insurance-graph',
                    figure=px.bar(
                        df_insurance,
                        x="Individual's Insurance Status:",
                        y="Count",
                        color="Individual's Insurance Status:",
                        text="Count" 
                    ).update_layout(
                        title='Insurance Status Distribution',
                        xaxis_title='Insurance Status',
                        yaxis_title='Count',
                        title_x=0.5,
                        font=dict(
                            family='Calibri',
                            size=17,
                            color='black'
                        )
                    ).update_traces(
                         textposition='auto',  # Position the text outside the bars
                        hovertemplate='<b>Insurance Status</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
                    )
                )
            ]
        )
    ]
),

# ROW 3
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph1',
            children=[
                # Location Encountered
                dcc.Graph(

                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                # Gender Distribution
                dcc.Graph(

                )
            ]
        )
    ]
),

# ROW 4
html.Div(
    className='row2',
    children=[
        html.Div(
            className='graph3',
            children=[
                # Support Given Graph 
                dcc.Graph(

                )
            ]
        ),
        html.Div(
            className='graph4',
            children=[
                # Age Distribution Graph
                dcc.Graph(

                )
            ]
        )
    ]
),

# ROW 5
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph1',
            children=[
                # Status Distribution 
                dcc.Graph(

                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                # Person filling out this form bar chart
                dcc.Graph(

                )
            ]
        )
    ]
),

# ROW 9
html.Div(
    className='row3',
    children=[
        html.Div(
            # ZIP Code Map
            # className='graph5',
            # children=[
            #     html.H1(
            #         'Number of Visitors by Zip Code', 
            #         className='zip'
            #     ),
            #     html.Iframe(
            #         className='folium',
            #         id='folium-map',
            #         srcDoc=map_html
            #         # style={'border': 'none', 'width': '80%', 'height': '800px'}
            #     )
            # ]
        )
    ]
),

# ROW 9
html.Div(
    className='row4',
    children=[
        html.Div(
            className='graph0',
            children=[
                # Horizontal Bar chart for zip code:
                dcc.Graph(
                    id='zip-graph',
                    figure=px.bar(
                        df_z,
                        x='Count',
                        y='ZIP2',
                        color='ZIP2',
                        text='Count'
                    ).update_layout(
                        title='Number of Visitors by Zip Code',
                        xaxis_title='Residents',
                        yaxis_title='Zip Code',
                        title_x=0.5,
                        font=dict(
                            family='Calibri',
                            size=17,
                            color='black'
                        )
                    ).update_traces(
                        textposition='auto',
                        hovertemplate='<b>ZIP Code</b>: %{x}<br><b>Residents</b>: %{y}<extra></extra>'
                    )
                )
            ]
        )
    ]
),

# ROW 9
html.Div(
    className='row4',
    children=[
        html.Div(
            className='graph0',
            children=[
                dcc.Graph(
                    figure=support_fig
                    # figure=px.bar(
                    #     df_support_counts,
                    #     x='Month',
                    #     y='Count',
                    #     color='Type of support given:',
                    #     barmode='group',
                    #     text='Count',
                    #     title='Type of Support Given',
                    #     labels={
                    #         'Count': 'Number of Supports',
                    #         'Month': 'Month',
                    #         'Type of support given:': 'Type of Support'
                    #     },
                    # ).update_layout(
                    #     title_x=0.5,
                    #     xaxis_title='Month',
                    #     yaxis_title='Count',
                    #     height=900,  # Adjust graph height
                    #     font=dict(
                    #         family='Calibri',
                    #         size=17,
                    #         color='black'
                    #     ),
                    #     xaxis=dict(
                    #         tickmode='array',
                    #         tickvals=df_support_counts['Month'].unique(),
                    #         tickangle=-35
                    #     ),
                    #     legend=dict(
                    #         title='Type of Support Given',
                    #         orientation="v",  # Vertical legend
                    #         x=1.05,  # Position legend to the right
                    #         xanchor="left", # Anchor legend to the left
                    #         y=1, # Position legend at the top
                    #         yanchor="top" # Anchor legend at the top
                    #     ),
                    #     hovermode='x unified',
                    # ).update_traces(
                    #     texttemplate='%{text}',
                    #     textposition='auto',
                    #     hovertemplate=(
                    #         '<br>'
                    #         '<b></b>%{y}'
                    #     ),
                    #     customdata=df_support_counts[['Type of support given:']],
                    # )
                )
            ]
        )
    ]
),
])

# Callback function
# @app.callback(
#     Output('', 'figure'),
#     [Input('', 'value')]
# )

print(f"Serving Flask app: '{current_file}'! 🚀")

if __name__ == '__main__':
    app.run_server(debug=
                   True)
                #    False)

# ----------------------------------------------- Updated Database ----------------------------------------

# updated_path = 'data/q1_2025_cleaned.xlsx'
# data_path = os.path.join(script_dir, updated_path)
# df_q1.to_excel(data_path, index=False)
# print(f"DataFrame saved to {data_path}")

# updated_path1 = 'data/service_tracker_q4_2024_cleaned.csv'
# data_path1 = os.path.join(script_dir, updated_path1)
# df.to_csv(data_path1, index=False)
# print(f"DataFrame saved to {data_path1}")

# -------------------------------------------- KILL PORT ---------------------------------------------------

# netstat -ano | findstr :8050
# taskkill /PID 24772 /F
# npx kill-port 8050

# ---------------------------------------------- Host Application -------------------------------------------

# 1. pip freeze > requirements.txt
# 2. add this to procfile: 'web: gunicorn responses_q4_2024:server'
# 3. heroku login
# 4. heroku create
# 5. git push heroku main

# Create venv 
# virtualenv venv 
# source venv/bin/activate # uses the virtualenv

# Update PIP Setup Tools:
# pip install --upgrade pip setuptools

# Install all dependencies in the requirements file:
# pip install -r requirements.txt

# Check dependency tree:
# pipdeptree
# pip show package-name

# Remove
# pypiwin32
# pywin32
# jupytercore

# ----------------------------------------------------

# Heroku Setup:
# heroku login
# heroku create bmhc-responses-q1-2025
# heroku git:remote -a bmhc-responses-q4-2024
# git push heroku main

# Clear Heroku Cache:
# heroku plugins:install heroku-repo
# heroku repo:purge_cache -a bmhc-responses-q1-2025

# Set buildpack for heroku
# heroku buildpacks:set heroku/python

# Heatmap Colorscale colors -----------------------------------------------------------------------------

#   ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance',
            #  'blackbody', 'bluered', 'blues', 'blugrn', 'bluyl', 'brbg',
            #  'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl', 'cividis', 'curl',
            #  'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
            #  'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys',
            #  'haline', 'hot', 'hsv', 'ice', 'icefire', 'inferno', 'jet',
            #  'magenta', 'magma', 'matter', 'mint', 'mrybm', 'mygbm', 'oranges',
            #  'orrd', 'oryel', 'oxy', 'peach', 'phase', 'picnic', 'pinkyl',
            #  'piyg', 'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn',
            #  'puor', 'purd', 'purp', 'purples', 'purpor', 'rainbow', 'rdbu',
            #  'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor', 'reds', 'solar',
            #  'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
            #  'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid',
            #  'turbo', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr',
            #  'ylorrd'].

# rm -rf ~$bmhc_data_2024_cleaned.xlsx
# rm -rf ~$bmhc_data_2024.xlsx
# rm -rf ~$bmhc_q4_2024_cleaned2.xlsx