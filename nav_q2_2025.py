
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

# Google Web Credentials
import json
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 'data/~$bmhc_data_2024_cleaned.xlsx'
# print('System Version:', sys.version)
# -------------------------------------- DATA ------------------------------------------- #

current_dir = os.getcwd()
current_file = os.path.basename(__file__)
script_dir = os.path.dirname(os.path.abspath(__file__))
# data_path = 'data/Navigation_Responses.xlsx'
# file_path = os.path.join(script_dir, data_path)
# data = pd.read_excel(file_path)
# df = data.copy()

# Define the Google Sheets URL
sheet_url = "https://docs.google.com/spreadsheets/d/1Vi5VQWt9AD8nKbO78FpQdm6TrfRmg0o7az77Hku2i7Y/edit#gid=78776635"

# Define the scope
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load credentials
encoded_key = os.getenv("GOOGLE_CREDENTIALS")

if encoded_key:
    json_key = json.loads(base64.b64decode(encoded_key).decode("utf-8"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
else:
    creds_path = r"C:\Users\CxLos\OneDrive\Documents\BMHC\Data\bmhc-timesheet-4808d1347240.json"
    if os.path.exists(creds_path):
        creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    else:
        raise FileNotFoundError("Service account JSON file not found and GOOGLE_CREDENTIALS is not set.")

# Authorize and load the sheet
client = gspread.authorize(creds)
sheet = client.open_by_url(sheet_url)
worksheet = sheet.get_worksheet(0)  # ✅ This grabs the first worksheet
data = pd.DataFrame(worksheet.get_all_records())
# data = pd.DataFrame(client.open_by_url(sheet_url).get_all_records())
df = data.copy()

# Get the reporting month:
current_month = datetime(2025, 3, 1).strftime("%B")

df.columns = df.columns.str.strip()

# Filtered df where 'Date of Activity:' is between 2025-01-01 and 2025-03-31
df['Date of Activity'] = pd.to_datetime(df['Date of Activity'], errors='coerce')
df = df[(df['Date of Activity'] >= '2025-1-01') & (df['Date of Activity'] <= '2025-3-31')]
df['Month'] = df['Date of Activity'].dt.month_name()

df_1 = df[df['Month'] == 'January']
df_2 = df[df['Month'] == 'February']
df_3 = df[df['Month'] == 'March']

# Extract the month name
# df['Month'] = df['Date of activity:'].dt.month_name()

# Define a discrete color sequence
color_sequence = px.colors.qualitative.Plotly

# print(df.head())
# print('Total entries: ', len(df))
# print('DF Shape:', df.shape)
# print('Dtypes: \n', df.dtypes)
# print('Info:', df.info())
# print("Amount of duplicate rows:", df.duplicated().sum())

# print('Current Directory:', current_dir)
# print('Script Directory:', script_dir)
# print('Path to data:',file_path)

# ================================= Columns Navigation ================================= #

# print('Column Names: \n', df.columns)

nav_columns = [
    'Timestamp',
    'Date of Activity', 
    'Person submitting this form:',
    'Activity Duration (minutes):',
    'Total travel time (minutes):',
    'Location Encountered:', 
    "Individual's First Name:",  # Fixed single quote
    "Individual's Date of Birth:",  # Fixed single quote
    "Individual's Insurance Status:",  # Fixed single quote
    "Individual's street address:",  # Fixed single quote
    'City:', 
    'ZIP Code:',
    'County:',
    'Type of support given:', 
    'Provide brief support description:',
    "Individual's Status:",  # Fixed single quote
    'HMIS SPID Number:',
    'MAP Card Number',
    'Gender:',
    'Race/Ethnicity:', 
    'Direct Client Assistance Amount:',
    'Column 21', 
    "Individual's Last Name:",  # Fixed single quote
    'Any recent or planned changes to BMHC lead services or programs?',
    'Month'
]

# ================================= Missing Values ================================= #

# missing = df_q1.isnull().sum()
# print('Columns with missing values before fillna: \n', missing[missing > 0])

# ============================== Checking Duplicates ========================== #

# ============================== Data Preprocessing ========================== #

# Check for duplicate columns
# duplicate_columns = df.columns[df.columns.duplicated()].tolist()
# print(f"Duplicate columns found: {duplicate_columns}")
# if duplicate_columns:
#     print(f"Duplicate columns found: {duplicate_columns}")

df.rename(
    columns={
        "Person submitting this form:": "Person",
        "Activity Duration (minutes):": "Minutes",
        "Total travel time (minutes):": "Travel Time",
        "Location Encountered:": "Location",
        "Individual's Insurance Status:": "Insurance",
        "Type of support given:": "Support",
        "Race/Ethnicity:" : "Ethnicity",
        # "Individual's Status:": "Status",
    }, 
inplace=True)

# Fill Missing Values for df
# List of columns to fill with 'Unknown'
columns_to_fill = [
    "Individual's Name:",
    "Individual's Insurance Status:",
    "City:",
    "County:",
    "Gender:",
    "Race/Ethnicity:",
    "Individual's street address:"
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

# Get the reporting quarter:
def get_custom_quarter(date_obj):
    month = date_obj.month
    if month in [10, 11, 12]:
        return "Q1"  # October–December
    elif month in [1, 2, 3]:
        return "Q2"  # January–March
    elif month in [4, 5, 6]:
        return "Q3"  # April–June
    elif month in [7, 8, 9]:
        return "Q4"  # July–September

# Reporting Quarter (use last month of the quarter)
report_date = datetime(2025, 3, 1)  # Example report date for Q2 (Jan–Mar)
month = report_date.month
report_year = report_date.year
current_quarter = get_custom_quarter(report_date)
# print(f"Reporting Quarter: {current_quarter}")

# Adjust the quarter calculation for custom quarters
if month in [10, 11, 12]:
    quarter = 1  # Q1: October–December
elif month in [1, 2, 3]:
    quarter = 2  # Q2: January–March
elif month in [4, 5, 6]:
    quarter = 3  # Q3: April–June
elif month in [7, 8, 9]:
    quarter = 4  # Q4: July–September

# Define a mapping for months to their corresponding quarter
quarter_months = {
    1: ['October', 'November', 'December'],  # Q1
    2: ['January', 'February', 'March'],    # Q2
    3: ['April', 'May', 'June'],            # Q3
    4: ['July', 'August', 'September']      # Q4
}

# Get the months for the current quarter
months_in_quarter = quarter_months[quarter]

# Calculate start and end month indices for the quarter
all_months = [
    'January', 'February', 'March', 
    'April', 'May', 'June',
    'July', 'August', 'September', 
    'October', 'November', 'December'
]
start_month_idx = (quarter - 1) * 3
month_order = all_months[start_month_idx:start_month_idx + 3]

# -------------------------- Activity Duration DF ------------------------- #

df_nav_hours = df[['Month', 'Minutes']]
nav_hours = df['Minutes'].sum()/60
nav_hours = round(nav_hours)  # Round to the nearest whole number
# print('Activity Duration:', df_duration/60, 'hours')

# -------------------------- Clients Served DF ------------------------- #

clients_served = len(df)
# print('Clients Served This Month:', clients_served)

df['Clients Served'] = len(df)

clients = []
for month in months_in_quarter:
    clients_in_month = df[df['Month'] == month].shape[0]  # Count the number of rows for each month
    clients.append(clients_in_month)
    # print(f'Clients Served in {month}:', clients_in_month)

# Create a DataFrame with the results for plotting
df_clients = pd.DataFrame(
    {
    'Month': months_in_quarter,
    'Clients Served': clients
    }
)

# print(df_clients)

client_fig = px.bar(
    df_clients, 
    x='Month', 
    y='Clients Served',
    labels={'Clients Served': 'Number of Clients'},
    color='Month',  # Color the bars by month
    text='Clients Served',  # Display the value on top of the bars
).update_layout(
    title_x=0.5,
    xaxis_title='Month',
    yaxis_title='Count',
    height=600,  # Adjust graph height
    title=dict(
        text= f'{current_quarter} Clients Served by Month',
        x=0.5, 
        font=dict(
            size=35,
            family='Calibri',
            color='black',
            )
    ),
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    xaxis=dict(
        title=dict(
            text=None,
            # text="Month",
            font=dict(size=20),  # Font size for the title
        ),
        tickmode='array',
        tickvals=df_clients['Month'].unique(),
        tickangle=0  # Rotate x-axis labels for better readability
    ),
    legend=dict(
        # title='Administrative Activity',
        title=None,
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend at the top
    ),
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=20),  # Increase text size in each bar
    textposition='auto',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<b>Name</b>: %{label}<br><b>Count</b>: %{y}<extra></extra>'  
    ),
)

client_pie = px.pie(
    df_clients,
    names='Month',
    values='Clients Served',
    color='Month',
    height=550
).update_layout(
    title=dict(
        x=0.5,
        text=f'{current_quarter} Ratio of Clients Served',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    ),  # Center-align the title
    margin=dict(
        l=0,  # Left margin
        r=0,  # Right margin
        t=100,  # Top margin
        b=0   # Bottom margin
    )  # Add margins around the chart
).update_traces(
    rotation=180,  # Rotate pie chart 90 degrees counterclockwise
    textfont=dict(size=19),  # Increase text size in each bar
    textinfo='value+percent',
    # texttemplate='<br>%{percent:.0%}',  # Format percentage as whole numbers
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# -------------------------- Travel Time ------------------------- #

# print("Travel Time Unique Before: \n", df['Travel Time'].unique().tolist())

travel_unique =  [
0, 60, 30, 'The Bumgalows', 45, '', 15, 240
 ]

# Clean travel time values
df['Travel Time'] = (
    df['Travel Time']
    .astype(str)
    .str.strip()
    .replace({
        "" : pd.NA,
        "The Bumgalows" : 0,
    })
)

df['Travel Time'] = pd.to_numeric(df['Travel Time'], errors='coerce')
df['Travel Time'] = df['Travel Time'].fillna(0)

# print("Travel Time Unique After: \n", df['Total travel time (minutes):'].unique().tolist())
# print(['Travel Time Value Counts: \n', df['Travel Time'].value_counts()])

total_travel_time = df['Travel Time'].sum()/60
total_travel_time = round(total_travel_time)
# print("Total travel time:",total_travel_time)

# Calculate total travel time per month
travel_hours = []
for month in months_in_quarter:
    hours_in_month = df[df['Month'] == month]['Travel Time'].sum() / 60
    hours_in_month = round(hours_in_month)
    travel_hours.append(hours_in_month)
    # print(f'Travel Time in {month}:', hours_in_month)

df_travel = pd.DataFrame({
    'Month': months_in_quarter,
    'Travel Time': travel_hours
})

# Bar chart
travel_fig = px.bar(
    df_travel,
    x='Month',
    y='Travel Time',
    color='Month',
    text='Travel Time',
    labels={
        'Travel Time': 'Travel Time (hours)',
        'Month': 'Month'
    }
).update_layout(
    title_x=0.5,
    xaxis_title='Month',
    yaxis_title='Travel Time (hours)',
    height=600,
    title=dict(
        text=f'{current_quarter} Travel Time by Month',
        x=0.5, 
        font=dict(
            size=35,
            family='Calibri',
            color='black',
            )
    ),
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    xaxis=dict(
        title=dict(
            text=None,
            font=dict(size=20),
        ),
        tickmode='array',
        tickvals=df_travel['Month'].unique(),
        tickangle=0
    ),
).update_traces(
    texttemplate='%{text}',
    textfont=dict(size=20),
    textposition='auto',
    textangle=0,
    hovertemplate='<b>Month</b>: %{label}<br><b>Travel Time</b>: %{y} hours<extra></extra>',
).add_annotation(
    x='January',  # Specify the x-axis value
    y=df_nav_hours.loc[df_nav_hours['Month'] == 'January', 'Minutes'].values[0] - 10,  # Position slightly above the bar
    text='No data',  # Annotation text
    showarrow=False,  # Hide the arrow
    font=dict(size=30, color='red'),  # Customize font size and color
    align='center',  # Center-align the text
)

# Pie chart
travel_pie = px.pie(
    df_travel,
    names='Month',
    values='Travel Time',
    color='Month',
    height=550
).update_layout(
    title=dict(
        x=0.5,
        text=f'{current_quarter} Travel Time Ratio',
        font=dict(
            size=35,
            family='Calibri',
            color='black'
        ),
    ),
    margin=dict(l=0, r=0, t=100, b=0)
).update_traces(
    rotation=180,
    textfont=dict(size=19),
    textinfo='value+percent',
    hovertemplate='<b>%{label}</b>: %{value} hours<extra></extra>'
)

# ----------------------- New/ Returning Stattus DF ------------------------- #

# Group by 'Individual\'s Status:' for all months and calculate percentages
df_status = df['Individual\'s Status:'].value_counts().reset_index(name='Count')
df_status['Percentage'] = (df_status['Count'] / df_status['Count'].sum()) * 100
df_status['Percentage'] = df_status['Percentage'].round(0)  # Round to nearest whole number

# Pie chart
status_fig = px.pie(
    df_status,
    names='Individual\'s Status:',
    values='Count'
).update_layout(
    title= f'{current_quarter} New vs. Returning',
    title_x=0.5,
    height=600,
    showlegend=True,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    textinfo='percent+value',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>',
    texttemplate='<br>%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_status['Percentage']  # Pass calculated percentage as custom data
)

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

# Apply categorization
df['Age_Group'] = df['Client Age'].apply(categorize_age)

# Extract the month name
df['Month'] = df['Date of Activity'].dt.month_name()

# Filter for October, November, December
df_q_age = df[df['Month'].isin(['January', 'February', 'March'])]

# Group data by Month and Age Group
df_age_counts = (
    df_q_age.groupby(['Month', 'Age_Group'], sort=False)
    .size()
    .reset_index(name='Patient_Visits')
)

# Sort months and age groups
month_order = ['January', 'February', 'March']

age_order = [
    '10-19', 
    '20-29', 
    '30-39', 
    '40-49',
    '50-59',
    '60-69', 
    '70-79',
    '80+'
    ]

df_age_counts['Month'] = pd.Categorical(df_age_counts['Month'], categories=months_in_quarter, ordered=True)
df_age_counts['Age_Group'] = pd.Categorical(df_age_counts['Age_Group'], categories=age_order, ordered=True)

# print(df_decades.value_counts())

# Create the grouped bar chart
age_fig = px.bar(
    df_age_counts,
    x='Month',
    y='Patient_Visits',
    color='Age_Group',
    barmode='group',
    text='Patient_Visits',
    labels={
        'Patient_Visits': 'Number of Visits',
        'Month': 'Month',
        'Age_Group': 'Age Group'
    },
).update_layout(
    height=850,  # Adjust graph height
    title=dict(
        text= f'{current_quarter} Age Comparison',  # Title text
        x=0.5,  # Center-align the title
        font=dict(
            size=35,
            family='Calibri',  # Optional: specify font family),  # Font size for the title
            color='black',  # Optional: specify font color
        )
    ),
    xaxis=dict(
        tickangle=-25,  # Rotate x-axis labels for better readability
        tickfont=dict(size=18),  # Adjust font size for the tick labels
        title=dict(
            text=None,
            font=dict(size=22),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Number of Visits',
            font=dict(size=22),  # Font size for the title
        ),
    ),
    legend=dict(
        title='',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend to the top
    ),
    hovermode='x unified', # Display only one hover label per trace
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
    margin=dict(l=0, r=0, t=60, b=0),
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=20),  # Increase text size in each bar
    textposition='outside',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<br>'
        '<b>Count: </b>%{y}<br>'  # Count
    ),
    customdata=df_age_counts[['Age_Group']].values.tolist(),  # Add custom data for hover
).add_vline(
    x=0.5,  # Adjust the position of the line
    line_dash="dash",
    line_color="gray",
    line_width=2
).add_vline(
    x=1.5,  # Position of the second line
    line_dash="dash",
    line_color="gray",
    line_width=2
)

# Group by 'Age_Group' and count the number of patient visits
df_decades = df.groupby('Age_Group', observed=True).size().reset_index(name='Patient_Visits')
df_decades['Age_Group'] = pd.Categorical(df_decades['Age_Group'], categories=age_order, ordered=True)
df_decades = df_decades.sort_values('Age_Group')

# Age_Group Pie chart:
age_pie = px.pie(
    df_decades,
    names='Age_Group',
    values='Patient_Visits',
    color='Age_Group',
    height=900
).update_layout(
    title=dict(
        x=0.5,
        text= f'{current_quarter} Age Distribution',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    ),  # Center-align the title
).update_traces(
    textfont=dict(size=19),  # Increase text size in each bar
    textinfo='value+percent',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ------------------------ Type of Support Given DF --------------------------- #

# print("Support Unique Before: \n", df['Support'].unique().tolist())

support_unique = [
    
]

df['Support'] = (
    df['Support']
    .astype(str)
    .str.strip()
    .replace({
        "" : pd.NA,
    })
    )

# Group the data by 'Month' and 'Type of support given:' to count occurrences
df_support_counts = (
    df.groupby(['Month', 'Support'],sort=False)
    .size()
    .reset_index(name='Count')
)

# print("Support Unique After: \n", df['Support'].unique().tolist())

df_support_counts['Month'] = pd.Categorical(
    df_support_counts['Month'], 
    categories = months_in_quarter, 
    ordered=True)

# print(df_support_counts)

# Support_fig grouped bar chart
support_fig = px.bar(
    df_support_counts,
    x='Month',
    y='Count',
    color='Support',
    barmode='group',
    text='Count',
    title='Support',
    labels={ 
        'Count': 'Number of Services',
        'Month': 'Month',
        'Type of support given:': 'Type of Support'
    },
).update_layout(
    title_x=0.5,
    xaxis_title='Month',
    yaxis_title='Count',
    height=900,  # Adjust graph height
        title=dict(
        text= f'{current_quarter} Type of Support',  # Title text
        x=0.5,  # Center-align the title
        font=dict(
            size=35,
            family='Calibri',  # Optional: specify font family),  # Font size for the title
            color='black',  # Optional: specify font color
        )
    ),
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    xaxis=dict(
        tickmode='array',
        # tickvals=df_support_counts['Month'].unique(),
        tickangle=-35  # Rotate x-axis labels for better readability
    ),
    legend=dict(
        title='',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend at the top
    ),
    hovermode='x unified',  # Display only one hover label per trace
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
).update_traces(
    textposition='outside', 
    textangle=0, # Automatically position text above bars
    textfont=dict(size=30),  # Increase text size in each bar
    hovertemplate=(  # Custom hover template
        '<br>'
        '<b>Count: </b>%{y}'
    ),
    customdata=df_support_counts[["Support"]].values.tolist(),  # Add custom data for hover
).add_vline(
    x=0.5,  # Adjust the position of the line
    line_dash="dash",
    line_color="gray",
    line_width=2
).add_vline(
    x=1.5,  # Position of the second line
    line_dash="dash",
    line_color="gray",
    line_width=2
).add_shape(
    type="rect",
    x0=-0.5,  # Start of the first group
    x1=0.5,   # End of the first group
    y0=0,     # Start of y-axis
    y1=1,     # End of y-axis (relative to the chart area)
    fillcolor="lightgray",
    opacity=0.1,
    layer="below"
)

df_support = df['Support'].value_counts().reset_index(name='Count')

#  Pie chart:
support_pie = px.pie(
    df_support,
    names='Support',
    values='Count',
    color='Support',
    height=800
).update_layout(
    title=dict(
        x=0.5,
        text= f'{current_quarter} Type of Support Given',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    ),
    margin=dict(l=0, r=0, t=50, b=0)
).update_traces(
    rotation = 90,
    textfont=dict(size=19),  # Increase text size in each bar
    textinfo='value+percent',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# -------------------------- Insurance Status DF ------------------------- #

# Define a custom color mapping for each month
custom_colors = {
    'January': 'Blues',
    'February': 'Greens',
    'March': 'Oranges',
    'April': 'Purples',
    'May': 'Reds',
    'June': 'Greys',
    'July': 'YlGn',
    'August': 'YlOrBr',
    'September': 'PuRd',
    'October': 'BuPu',
    'November': 'GnBu',
    'December': 'YlGnBu',
}

df['Insurance'] = (df['Insurance']
      .astype(str)
        .str.strip()
        .replace({
            'MAP 000' : "MAP 100",
            'Did not disclose.' : "NONE",
        })
      )

# Group data by Month and Insurance Status
df_insurance_counts = (
    df.groupby(['Month', 'Insurance'], sort=False)
    .size() # Count the number of occurrences
    .reset_index(name='Count') # Reset the index and rename the count column
)

# Sort months in the desired order
df_insurance_counts['Month'] = pd.Categorical(
    df_insurance_counts['Month'], 
    categories=months_in_quarter, 
    ordered=True
)

# Add a custom color column based on the month
df_insurance_counts['Color'] = df_insurance_counts['Month'].map(custom_colors)

df_insurance_counts['Color'] = df_insurance_counts['Month'].map(custom_colors)

# Create the grouped bar chart
insurance_fig = px.bar(
    df_insurance_counts,
    x='Month',
    y='Count',
    color="Insurance",
    barmode='group',
    text='Count',
    labels={
        "Insurance": "Insurance Status",
        'Count': 'Number of Individuals',
        'Month': 'Month'
    },
).update_layout(
    title_x=0.5,
    xaxis_title='Month',
    yaxis_title='Count',
    height=900,  # Adjust graph height
    title=dict(
        text= f'{current_quarter } Insurance Status by Month',
        x=0.5, 
        font=dict(
            size=35,
            family='Calibri',
            color='black',
            )
    ),
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    xaxis=dict(
        tickangle=-15  # Rotate x-axis labels for better readability
    ),
    legend=dict(
        title='',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend to the top
    ),
    hovermode='x unified', # Display only one hover label per trace
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
    margin=dict(l=0, r=0, t=50, b=0),
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=30),  # Increase text size in each bar
    textposition='outside',  # Automatically position text above bars
    hovertemplate=(  # Custom hover template
            '<br>'
        '<b>Count:</b> %{y}<br>'  # Count
    ),
    customdata=df_insurance_counts[["Insurance"]].values.tolist(),  # Add custom data for hover
).add_vline(
    x=0.5,  # Adjust the position of the line
    line_dash="dash",
    line_color="gray",
    line_width=2
).add_vline(
    x=1.5,  # Position of the second line
    line_dash="dash",
    line_color="gray",
    line_width=2
)

df_insurance = df[['Month', "Insurance"]]
df_insurance = df_insurance.groupby('Insurance').size().reset_index(name='Count')

# Insurance Status Pie Chart:
insurance_pie = px.pie(
    df_insurance,
    names="Insurance",
    values='Count',
    color="Insurance",
    height=900
).update_layout(
    title=dict(
        x=0.5,
        text= f'{current_quarter} Insurance Distribution',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    ),  
    margin=dict(l=0, r=0, t=50, b=0)
).update_traces(
    rotation=180,  
    textfont=dict(size=19), 
    insidetextorientation='horizontal', 
    textinfo='percent+value',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ------------------------ Location Encountered DF ----------------------- #

# print("Location Unique Before:", df['Location'].unique().tolist())

location_unique = [
    "Black Men's Health Clinic", 'Downtown Austin Community Court', 'Cross Creek Hospital', 'South Bridge', 'The Bumgalows', 'Office (remote) ', 'The Bungalows', 'Extended Stay America (Host Hotel) ', 'last known area was St. John. Connected with Hungry Hill to help me search for client ', 'PHONE CONTACT', 'PHONE', 'BMHC', 'GudLife', 'phone ', 'Home of resident', 'Community First Village', 'Cross Creek hospital', 'over phone', 'phone', 'phone call', 'over the phone', 'Hungry Hill/Austin Urban League', 'Cross creek hospital', 'Vivent Health', 'Clients home', 'Extended Stay America ', 'Outreach in the field ', 'Integral Care St. John Office ', 'Extended Stay America Hotel ', 'Outreach ', 'picked client up from encampment for SSA appointment'
]

df['Location'] = (
    df['Location']
    .astype(str)
    .str.strip()
    .replace({
        '' : pd.NA,
        'BMHC': "Black Men's Health Clinic",
        "Black Men's Health Clinic": "Black Men's Health Clinic",
        
        'Downtown Austin Community Court': 'Downtown Austin Community Court',
        
        'Cross Creek Hospital': 'Cross Creek Hospital',
        'Cross creek hospital': 'Cross Creek Hospital',
        'Cross Creek hospital': 'Cross Creek Hospital',
        
        'South Bridge': 'SouthBridge',
        'Southbridge': 'SouthBridge',
        
        'The Bumgalows': 'The Bungalows',
        'The Bungalows': 'The Bungalows',
        
        'Office (remote)': 'Remote Office',
        
        'Extended Stay America (Host Hotel)': 'Extended Stay America',
        'Extended Stay America Hotel': 'Extended Stay America',
        'Extended Stay America': 'Extended Stay America',
        
        'last known area was St. John. Connected with Hungry Hill to help me search for client': 'Field Outreach - St. John Area',
        'picked client up from encampment for SSA appointment': 'Field Outreach - SSA Pickup',
        'Outreach in the field': 'Field Outreach',
        'Outreach': 'Field Outreach',
        
        'PHONE CONTACT': 'Phone',
        'PHONE': 'Phone',
        'phone': 'Phone',
        'phone ': 'Phone',
        'phone call': 'Phone',
        'over phone': 'Phone',
        'over the phone': 'Phone',
        
        'Home of resident': 'Client Home',
        'Clients home': 'Client Home',
        
        'GudLife': 'GudLife',
        'Vivent Health': 'Vivent Health',
        
        'Community First Village': 'Community First Village',
        'CFV': 'Community First Village',
        
        'Hungry Hill/Austin Urban League': 'Hungry Hill / Austin Urban League',
        
        'Integral Care St. John Office': 'Integral Care - St. John',
    })
)


# print("Location Unique After:", df['Location'].unique().tolist())

# Group data by Month and Location Encountered
df_location_counts = (
    df.groupby(['Month', 'Location'], 
    sort=False) # Do not sort the groups
    .size() # Count the number of occurrences
    .reset_index(name='Count') # Reset the index and rename the count column
)

# print("Location Unique After:", df['Location'].unique().tolist())

df_location_counts['Month'] = pd.Categorical(
    df_location_counts['Month'], # Categorize the months for sorting purposes and to avoid alphabetical sorting.
    categories=months_in_quarter, # Specify the order of the categories
    ordered=True) # Ensure the categories are ordered

df_location_counts = df_location_counts.sort_values(['Month', 'Location'])

# Create the grouped bar chart
location_fig = px.bar(
    df_location_counts,
    x='Month',
    y='Count',
    color='Location',
    barmode='group',
    text='Count',
    labels={
        'Count': 'Number of Encounters',
        'Month': 'Month', # Specify the axis labels
        'Location': 'Location' # Specify the axis labels
    },
).update_layout(
    title_x=0.5,
    yaxis_title='Number of Encounters',
    height=900,  # Adjust graph height
    title=dict(
        text=f'{current_quarter} Location Encountered by Month',
        x=0.5, 
        font=dict(
            size=35,
            family='Calibri',
            color='black',
            )
    ),
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    xaxis=dict(
        title=None,
        tickangle=-15,  # Rotate x-axis labels for better readability
        tickfont=dict(size=25),  # Adjust font size for the month labels
    ),
    yaxis=dict(
        title=dict(
            text='Number of Encounters',
            font=dict(size=25),  # Font size for the title
        ),
    ),
    legend=dict(
        title=dict(
            text='Location',  # Legend title text
            font=dict(size=20),  # Font size for the legend title
            side='top',  # Position the title at the top of the legend box
        ),
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top",  # Anchor legend at the top
        valign='middle',  # Vertically align legend content to the middle
),
    hovermode='x unified', # Display only one hover label per trace
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
    margin=dict(l=0, r=0, t=50, b=0)
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=30),  # Increase text size in each bar
    textposition='outside',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<br>'
        '<b>Count: </b>%{y}<br>'  # Count
    ),
    customdata=df_location_counts[['Location']].values.tolist(),  # Add custom data for hover
).add_vline(
    x=0.5,  # Adjust the position of the line
    line_dash="dash",
    line_color="gray",
    line_width=2
).add_vline(
    x=1.5,  # Position of the second line
    line_dash="dash",
    line_color="gray",
    line_width=2
)

df_location = df['Location'].value_counts().reset_index(name='Count')

#  Pie chart:
location_pie = px.pie(
    df_location,
    names='Location',
    values='Count',
    color='Location',
    height=800
).update_layout(
    title=dict(
        x=0.5,
        text= f'{current_quarter} Ratio of Location Encounters',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    ),
    margin=dict(l=0, r=0, t=50, b=0)
).update_traces(
    rotation=0,
    textfont=dict(size=19),  # Increase text size in each bar
    textinfo='percent',
    # textinfo=None,
    # texttemplate='%{value}<br>%{percent:.1%}', 
    # insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# --------------------- Person Filling Out This Form DF -------------------- #

person_unique = [
    'Viviana Varela', 
    'Jaqueline Oviedo',
    'Michael Lambert', 
    'Michael Lambert ', 
    'Larry Wallace Jr',
    'Rishit Yokananth', 
    'Dominique Street', 
    'Dr Larry Wallace Jr',
    'Eric Roberts', 
    'The Bumgalows', 
    'Kimberly Holiday', 
    'Toya Craney',
    'Sonya Hosey',
    'Eric roberts', 
    'EricRoberts'
]

# print("Person Unique Before:", df["Person"].unique().tolist())

df['Person'] = (
    df['Person']
        .astype(str)
        .str.strip()
        .replace({
            '' : pd.NA,
            'Dominique': 'Dominique Street',
            'Jaqueline Ovieod': 'Jaqueline Oviedo',
            'Sonya': 'Sonya Hosey',
            'EricRoberts': 'Eric Roberts',
            'Eric roberts': 'Eric Roberts',
            'Larry Wallace Jr': 'Dr Larry Wallace Jr',
        })
)

# Group data by Month and Person submitting the form
df_person_counts = (
    df.groupby(['Month', 'Person'], sort=False)
    .size()
    .reset_index(name='Count')
)

df_person_counts['Month'] = pd.Categorical(
    df_person_counts['Month'], 
    categories=months_in_quarter, 
    ordered=True)

# Sort the dataframe by 'Month' and 'Person submitting this form:'
df_person_counts = df_person_counts.sort_values(['Month', 'Person'])

# Create the grouped bar chart
person_fig = px.bar(
    df_person_counts,
    x='Month',
    y='Count',
    color='Person',
    barmode='group', 
    text='Count',
    labels={
        'Count': 'Number of Forms',
        'Month': 'Month',
        'Person': 'Person'
    },
).update_layout(
    title_x=0.5,
    xaxis_title='Month',
    yaxis_title='Number of Forms',
    height=900,  
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    title=dict(
        text= f'{current_quarter} Person Submitting Forms By Month',  
        font=dict(size=40),  
        x=0.5  
    ),
    xaxis=dict(
        title=dict(
            text=None, 
            standoff=200
        ),
        automargin=True,
        tickfont=dict(size=25)
    ),
    yaxis=dict(
        title=dict(
            text='Number of Submissions',
            font=dict(size=25),
        ),
    ),
    legend=dict(
        title='Name',  # Legend title
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend at the top
    ),
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
    hovermode='x unified', # Display only one hover label per trace
    margin=dict(l=0, r=0, t=50, b=0)
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=30),  # Increase text size in each bar
    textposition='outside',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<br>'
        '<b>Count: </b>%{y}' 
    ),
    customdata=df_person_counts[['Person']].values.tolist(),  # Add custom data for hover
).add_vline(
    x=0.5,  # Adjust the position of the line
    line_dash="dash",
    line_color="gray",
    line_width=2
).add_vline(
    x=1.5,  # Position of the second line
    line_dash="dash",
    line_color="gray",
    line_width=2
).add_shape(
    type="rect",
    x0=-0.5,  # Start of the first group
    x1=0.5,   # End of the first group
    y0=0,     # Start of y-axis
    y1=1,     # End of y-axis (relative to the chart area)
    fillcolor="lightgray",
    opacity=0.1,
    layer="below"
)

df_pf = df['Person'].value_counts().reset_index(name='Count')

#  Pie chart:
pf_pie = px.pie(
    df_pf,
    names='Person',
    values='Count',
    color='Person',
    height=800
).update_layout(
    title=dict(
        x=0.5,
        text= f'{current_quarter} Person Submitting Forms',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    ),
    legend=dict(
        # title='Administrative Activity',
        title=None,
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend at the top
    ),
    margin=dict(l=0, r=0, t=50, b=0),
).update_traces(
    rotation=90,
    textfont=dict(size=19),  # Increase text size in each bar
    textinfo='value+percent',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# -------------------------- Race/ Ethnicity DF ------------------------- #

df['Ethnicity'] = (
    df['Ethnicity']
        .astype(str)
        .str.strip()
        .replace({
            "": pd.NA,
            'Hispanic/Latino' : 'Hispanic/ Latino',
            'White' : 'White/ European Ancestry',
        })
)

df_race = df['Ethnicity'].value_counts().reset_index(name='Count')
df_race['Percentage'] = (df_race['Count'] / df_race['Count'].sum()) * 100
df_race['Percentage'] = df_race['Percentage'].round(0)  # Round to nearest whole number
# print(df_race)

race_fig = px.pie(
    df_race,
    names='Ethnicity',
    values='Count'
).update_layout(
    title= f'{current_quarter} Client Visits by Race',
    title_x=0.5,
    height=550,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    legend=dict(
        # title='Administrative Activity',
        title=None,
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend at the top
    ),
    margin=dict(l=0, r=0, t=50, b=0),
).update_traces(
    textinfo='percent+value',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>',
    texttemplate='%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_race['Percentage']  # Pass calculated percentage as custom data
)

# ------------------------ ZIP2 DF ----------------------- #

df['ZIP2'] = df['ZIP Code:']

# print("ZIP2 Unique Before:", df['ZIP 2'].unique().tolist())

zip_unique =[
78744, 78640, 78723, '', 78741, 78704, 78753, 78750, 78621, 78758, 78724, 78754, 78721, 78664, 78613, 78759, 78731, 78653, 78702, 78617, 78728, 78660, 78745, 78752, 78748, 78747, 78725, 78661, 78719, 78612, 76513, 78415, 78656, 78618, 'N/A', 78705, 78659, 78756, 78714, 78662, 76537, 78729, 78751, 78245, 78644, 78735, 'Texas', 78610, 78757, 78634, 75223, 'Unhoused', 78717, 'NA', 78749, 78727, 78683, 'Unknown', 'Unknown '
]

zip2_mode = df['ZIP2'].mode()[0]

df['ZIP2'] = (
    df['ZIP2']
    .astype(str)
    .str.strip()
    .replace({
        '': pd.NA,
        'Texas': zip2_mode,
        'Unhoused': zip2_mode,
        'UNHOUSED': zip2_mode,
        'Unknown': zip2_mode,
        'Unknown ': zip2_mode,
        'NA': zip2_mode,
        'N/A': zip2_mode,
        'nan': zip2_mode,
    })
)

df['ZIP2'] = df['ZIP2'].fillna(zip2_mode)
df_z = df['ZIP2'].value_counts().reset_index(name='Count')

# print("ZIP2 Unique After:", df['ZIP Code:'].unique().tolist())

zip_fig =px.bar(
    df_z,
    x='Count',
    y='ZIP2',
    color='ZIP2',
    text='Count',
    orientation='h'  # Horizontal bar chart
).update_layout(
    title='Number of Visitors by Zip Code Q1',
    xaxis_title='Residents',
    yaxis_title='Zip Code',
    title_x=0.5,
    height=1100,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
        yaxis=dict(
        tickangle=0  # Keep y-axis labels horizontal for readability
    ),
        legend=dict(
        title='ZIP Code',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend at the top
    ),
).update_traces(
    textposition='auto',  # Place text labels inside the bars
    textfont=dict(size=30),  # Increase text size in each bar
    # insidetextanchor='middle',  # Center text within the bars
    textangle=0,            # Ensure text labels are horizontal
    hovertemplate='<b>ZIP Code</b>: %{y}<br><b>Count</b>: %{x}<extra></extra>'
)

# =============================== Distinct Values ========================== #

# Get the distinct values in column

# distinct_service = df['What service did/did not complete?'].unique()
# print('Distinct:\n', distinct_service)

# ==================================== Folium =================================== #

print("Zip Unique Before:", df['ZIP Code:'].unique().tolist())

df = df[df['ZIP Code:'].str.strip() != ""]

zip_unique =[
78744, 78640, 78723, '', 78741, 78704, 78753, 78750, 78621, 78758, 78724, 78754, 78721, 78664, 78613, 78759, 78731, 78653, 78702, 78617, 78728, 78660, 78745, 78752, 78748, 78747, 78725, 78661, 78719, 78612, 76513, 78415, 78656, 78618, 'N/A', 78705, 78659, 78756, 78714, 78662, 76537, 78729, 78751, 78245, 78644, 78735, 'Texas', 78610, 78757, 78634, 75223, 'Unhoused', 78717, 'NA', 78749, 78727, 78683, 'Unknown', 'Unknown '
]

mode_value = df['ZIP Code:'].mode()[0]
df['ZIP Code:'] = df['ZIP Code:'].fillna(mode_value)

df['ZIP2'] = (
    df['ZIP2']
    .astype(str)
    .str.strip()
    .replace({
        '': pd.NA,
        'Texas': zip2_mode,
        'Unhoused': zip2_mode,
        'UNHOUSED': zip2_mode,
        'Unknown': zip2_mode,
        'Unknown ': zip2_mode,
        'NA': zip2_mode,
        'N/A': zip2_mode,
        'nan': zip2_mode,
    })
)

print("Zip Unique After:", df['ZIP Code:'].unique().tolist())

# Count of visitors by zip code
df['ZIP Code:'] = df['ZIP Code:'].where(df['ZIP Code:'].str.isdigit(), mode_value)
df['ZIP Code:'] = df['ZIP Code:'].astype(int)

df_zip = df['ZIP Code:'].value_counts().reset_index(name='Residents')
# df_zip['ZIP Code:'] = df_zip['index'].astype(int)
df_zip['Residents'] = df_zip['Residents'].astype(int)
# df_zip.drop('index', axis=1, inplace=True)

# print("Zip Unique After:", df['ZIP Code:'].unique().tolist())

# print(df_zip.head())

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
                f'BMHC Client Navigation Report {current_quarter} 2025', 
                className='title'),
            html.H1(
                '01/01/2025 - 3/31/2025', 
                className='title2'),
            html.Div(
                className='btn-box', 
                children=[
                    html.A(
                        'Repo',
                        href=f'https://github.com/CxLos/BMHC_{current_quarter}_2025_Responses',
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
            className='graph11',
            children=[
                html.Div(
                    className='high1',
                    children=[f'{current_quarter} Clients Served']
                ),
                html.Div(
                    className='circle1',
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
        html.Div(
            className='graph22',
            children=[
                html.Div(
                    className='high1',
                    children=[f'{current_quarter} Navigation Hours']
                ),
                html.Div(
                    className='circle2',
                    children=[
                        html.Div(
                            className='hilite',
                            children=[
                                html.H1(
                                    className='high3',
                                    children=[nav_hours]
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ]
),

# ROW 1
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph11',
            children=[
                html.Div(
                    className='high1',
                    children=[f'{current_quarter} Travel Time']
                ),
                html.Div(
                    className='circle2',
                    children=[
                        html.Div(
                            className='hilite',
                            children=[
                                html.H1(
                                    className='high3',
                                    children=[total_travel_time]
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className='graph22',
            children=[
                html.Div(
                    className='high1',
                    children=[f'{current_quarter} Blank']
                ),
                html.Div(
                    className='circle2',
                    children=[
                        html.Div(
                            className='hilite',
                            children=[
                                html.H1(
                                    className='high3',
                                    # children=[total_travel_time]
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        ),
    ]
),

# ROW 5
html.Div(
    className='row1',
    children=[
        html.Div(
            className='graph1',
            children=[
                dcc.Graph(
                    figure=client_fig
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    figure=client_pie
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
                dcc.Graph(
                    figure=travel_fig
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    figure=travel_pie
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
                dcc.Graph(
                    figure=status_fig
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    figure=race_fig
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=age_fig
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=age_pie
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=support_fig
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=support_pie
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=insurance_fig
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=insurance_pie
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=location_fig
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=location_pie
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
            className='graph0',
            children=[
                dcc.Graph(
                    figure=person_fig
                )
            ]
        )
    ]
),

html.Div(
    className='row3',
    children=[
        html.Div(
            className='graph0',
            children=[
                dcc.Graph(
                    figure=pf_pie
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
                    figure=zip_fig
                )
            ]
        )
    ]
),

html.Div(
    className='row3',
    children=[
        html.Div(
            # ZIP Code Map
            className='graph5',
            children=[
                html.H1(
                    'Number of Visitors by Zip Code', 
                    className='zip'
                ),
                html.Iframe(
                    className='folium',
                    id='folium-map',
                    srcDoc=map_html
                    # style={'border': 'none', 'width': '80%', 'height': '800px'}
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