
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

df['Month'] = df['Date of activity:'].dt.month_name()

df_oct = df[df['Month'] == 'October']
df_nov = df[df['Month'] == 'November']
df_dec = df[df['Month'] == 'December']

# Extract the month name
# df['Month'] = df['Date of activity:'].dt.month_name()

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

df['Clients Served'] = len(df)

# Clients served in October:
clients_oct = df[df['Month'] == 'October']
clients_served_oct = len(clients_oct)

# Clients served in November:
clients_nov = df[df['Month'] == 'November']
clients_served_nov = len(clients_nov)

# Clients Served in December:
clients_dec = df[df['Month'] == 'December']
clients_served_dec = len(clients_dec)

# Create a DataFrame with the results for plotting
df_clients_q1 = pd.DataFrame(
    {
    'Month': ['October', 'November', 'December'],
    'Clients Served': [clients_served_oct, clients_served_nov, clients_served_dec]
    }
)

# print(df_clients_q1)

clients_served_fig = px.bar(
    df_clients_q1, 
    x='Month', 
    y='Clients Served',
    title='Total Clients Served Each Month (Q1)',
    labels={'Clients Served': 'Number of Clients'},
    color='Month',  # Color the bars by month
    text='Clients Served',  # Display the value on top of the bars
).update_layout(
    title_x=0.5,  # Center the title
    height=500,
    font=dict(
        family='Calibri',
        size=14,
        color='black'
    )
)

# -------------------------- Activity Duration DF ------------------------- #

df_duration = df['Activity duration (minutes):'].sum()/60
df_duration = round(df_duration)  # Round to the nearest whole number
# print('Activity Duration:', df_duration/60, 'hours')

# Duration in October:
df_duration_oct = df_oct['Activity duration (minutes):'].sum()/60
df_duration_oct = round(df_duration_oct)  # Round to the nearest whole number

# Duration in November:
df_duration_nov = df_nov['Activity duration (minutes):'].sum()/60
df_duration_nov = round(df_duration_nov)  # Round to the nearest whole number

# Duration in December:
df_duration_dec = df_dec['Activity duration (minutes):'].sum()/60
df_duration_dec = round(df_duration_dec)  # Round to the nearest whole number

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
df['Month'] = df['Date of activity:'].dt.month_name()

# Filter for October, November, December
df_q_age = df[df['Month'].isin(['October', 'November', 'December'])]

# Group data by Month and Age Group
df_age_counts = (
    df_q_age.groupby(['Month', 'Age_Group'], sort=False)
    .size()
    .reset_index(name='Patient_Visits')
)

# Sort months and age groups
month_order = ['October', 'November', 'December']

age_order = [
    # '10-19', 
    '20-29', 
    '30-39', 
    '40-49',
    '50-59',
    '60-69', 
    '70-79',
    # '80+'
    ]

df_age_counts['Month'] = pd.Categorical(df_age_counts['Month'], categories=month_order, ordered=True)
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
        text='Month by Month Age Group Comparison',  # Title text
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
        title='Age Group',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend to the top
    ),
    hovermode='x unified', # Display only one hover label per trace
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=20),  # Increase text size in each bar
    textposition='auto',  # Automatically position text above bars
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

# Age Group Totals:

# Group by 'Age_Group' and count the number of patient visits
df_decades = df.groupby('Age_Group', observed=True).size().reset_index(name='Patient_Visits')
df_decades['Age_Group'] = pd.Categorical(df_decades['Age_Group'], categories=age_order, ordered=True)
df_decades = df_decades.sort_values('Age_Group')

# Bar chart for Age Group Totals:
age_totals_fig = px.bar(
    df_decades,
    x='Age_Group',
    y='Patient_Visits',
    color='Age_Group',
    text='Patient_Visits',
    labels={
        'Patient_Visits': 'Number of Visits',
        'Age_Group': 'Age Group'
    },
).update_layout(
    height=850,  # Adjust graph height
    title=dict(
        x=0.5,
        text='Age Group Totals For Q1',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        )
    ),
    xaxis=dict(
        tickfont=dict(size=18),  # Adjust font size for the month labels
        tickangle=-45,  # Rotate x-axis labels for better readability
        title=dict(
            text='Age Group',
            font=dict(size=20),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Number of Visits',
            font=dict(size=22),  # Font size for the title
        ),
    ),
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=20),  # Increase text size in each bar
    textposition='auto',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<b>Age Group</b>: %{y}<br><b>Count</b>: %{x}<extra></extra>'
    ),
    customdata=df_decades[['Age_Group']].values.tolist(),  # Add custom data for hover
)

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
        text='Q1 Age Distribution',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    )  # Center-align the title
).update_traces(
    textfont=dict(size=19),  # Increase text size in each bar
    textinfo='value+percent',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ------------------------ Type of Support Given DF --------------------------- #

# DataFrame for columns "Type of support given:" and "Date of activity:"
df_support = df[['Type of support given:', 'Date of activity:']]

# # Extract the month from the 'Date of activity:' column
df_support['Month'] = df_support['Date of activity:'].dt.month_name()

# # Filter data for October, November, and December
df_support_q = df_support[df_support['Month'].isin(['October', 'November', 'December'])]

# Group the data by 'Month' and 'Type of support given:' to count occurrences
df_support_counts = (
    df_support_q.groupby(['Month', 'Type of support given:'],sort=False)
    .size()
    .reset_index(name='Count')
)

# Sort months in the desired order
month_order = ['October', 'November', 'December']

df_support_counts['Month'] = pd.Categorical(
    df_support_counts['Month'], 
    categories = month_order, 
    ordered=True) # pd.Categorical is to specify the order of the categories for sorting purposes and to avoid alphabetical sorting

# print(df_service_counts)

# Support_fig grouped bar chart
support_fig = px.bar(
    df_support_counts,
    x='Month',
    y='Count',
    color='Type of support given:',
    barmode='group',
    text='Count',
    title='Type of Support Given',
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
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    xaxis=dict(
        tickmode='array',
        tickvals=df_support_counts['Month'].unique(),
        tickangle=-35  # Rotate x-axis labels for better readability
    ),
    legend=dict(
        title='Type of Support',
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
    textposition='auto', 
    textangle=0, # Automatically position text above bars
    textfont=dict(size=30),  # Increase text size in each bar
    hovertemplate=(  # Custom hover template
        '<br>'
        '<b>Count: </b>%{y}'
    ),
    customdata=df_support_counts[["Type of support given:"]].values.tolist(),  # Add custom data for hover
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

df_support = df['Type of support given:'].value_counts().reset_index(name='Count')

# Type of support given Bar Chart Totals:
support_totals_fig = px.bar(
    df_support,
    x='Type of support given:',
    y='Count',
    color='Type of support given:',
    text='Count',
).update_layout(
    height=850,  # Adjust graph height
    title=dict(
        x=0.5,
        text='Type of Support Given Q1 Totals',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        )
    ),
    xaxis=dict(
        tickfont=dict(size=18),  # Adjust font size for the month labels
        tickangle=-25,  # Rotate x-axis labels for better readability
        title=dict(
            text='',
            font=dict(size=20),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=22),  # Font size for the title
        ),
    ),
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=20),  # Increase text size in each bar
    textposition='auto',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<b>Support</b>: %{label}<br><b>Count</b>: %{y}<extra></extra>'    
        ),
)

#  Pie chart:
support_pie = px.pie(
    df_support,
    names='Type of support given:',
    values='Count',
    color='Type of support given:',
    height=800
).update_layout(
    title=dict(
        x=0.5,
        text='Q1 Type of Support Given',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    )  # Center-align the title
).update_traces(
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

# Ensure the 'Date of activity:' column is in datetime format
df['Date of activity:'] = pd.to_datetime(df['Date of activity:'], errors='coerce')

# Clean and preprocess the 'Individual's Insurance Status:' column
df["Individual's Insurance Status:"] = df["Individual's Insurance Status:"].str.strip()
df["Individual's Insurance Status:"] = df["Individual's Insurance Status:"].replace('MAP 000', 'MAP 100')
df["Individual's Insurance Status:"] = df["Individual's Insurance Status:"].replace('Did not disclose.', 'NONE')

# Extract the month name
df['Month'] = df['Date of activity:'].dt.month_name()

# Filter data for October, November, and December
df_q_insurance = df[df['Month'].isin(['October', 'November', 'December'])]

# Group data by Month and Insurance Status
df_insurance_counts = (
    df_q_insurance.groupby(['Month', "Individual's Insurance Status:"], sort=False)
    .size() # Count the number of occurrences
    .reset_index(name='Count') # Reset the index and rename the count column
)

# Sort months in the desired order
month_order = ['October', 'November', 'December']
df_insurance_counts['Month'] = pd.Categorical(
    df_insurance_counts['Month'], 
    categories=month_order, 
    ordered=True
)

# Add a custom color column based on the month
df_insurance_counts['Color'] = df_insurance_counts['Month'].map(custom_colors)

# Create the grouped bar chart
insurance_fig = px.bar(
    df_insurance_counts,
    x='Month',
    y='Count',
    color="Individual's Insurance Status:",
    barmode='group',
    text='Count',
    title='Insurance Status Monthly Comparison',
    labels={
        "Individual's Insurance Status:": "Insurance Status",
        'Count': 'Number of Individuals',
        'Month': 'Month'
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
        tickangle=-15  # Rotate x-axis labels for better readability
    ),
    legend=dict(
        title='Insurance Status',
        orientation="v",  # Vertical legend
        x=1.05,  # Position legend to the right
        xanchor="left",  # Anchor legend to the left
        y=1,  # Position legend at the top
        yanchor="top"  # Anchor legend to the top
    ),
    hovermode='x unified', # Display only one hover label per trace
    bargap=0.08,  # Reduce the space between bars
    bargroupgap=0,  # Reduce space between individual bars in groups
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=30),  # Increase text size in each bar
    textposition='auto',  # Automatically position text above bars
    hovertemplate=(  # Custom hover template
            '<br>'
        '<b>Count:</b> %{y}<br>'  # Count
    ),
    customdata=df_insurance_counts[["Individual's Insurance Status:"]].values.tolist(),  # Add custom data for hover
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

df_insurance = df.groupby("Individual's Insurance Status:").size().reset_index(name='Count')

# Bar chart for Insurance Totals:
insurance_totals_fig = px.bar(
    df_insurance,
    x='Individual\'s Insurance Status:',
    y='Count',
    color='Individual\'s Insurance Status:',
    text='Count',
).update_layout(
    height=850,  # Adjust graph height
    title=dict(
        x=0.5,
        text='Insurance Status Q1 Totals',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        )
    ),
    xaxis=dict(
        tickfont=dict(size=18),  # Adjust font size for the month labels
        tickangle=-25,  # Rotate x-axis labels for better readability
        title=dict(
            text=None,
            font=dict(size=20),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=22),  # Font size for the title
        ),
    ),
    bargap=0.08,  # Reduce the space between bars
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=20),  # Increase text size in each bar
    textposition='auto',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<b>Insurance</b>: %{label}<br><b>Count</b>: %{y}<extra></extra>'  
    ),
)

# Heatmap for Insurance Status:
# Create a pivot table for the heatmap
df_insurance_pivot = df_insurance_counts.pivot(
    index='Month',
    columns="Individual's Insurance Status:",
    values='Count'
)

# Create the heatmap
insurance_heatmap = go.Figure(
    data=go.Heatmap(
        z=df_insurance_pivot.values,
        x=df_insurance_pivot.columns,
        y=df_insurance_pivot.index,
        colorscale='viridis'
    )

).update_layout(
    title='Insurance Status Heatmap',
    title_x=0.5,
    xaxis_title="Insurance Status",
    yaxis_title="Month",
    font=dict(
        family='Calibri',
        size=17,
    )
)

# Treemap for Insurance Status:
insurance_treemap = px.treemap(
    df_insurance_counts,
    path=['Month', "Individual's Insurance Status:"],
    values='Count',
    color='Color',
    # color_continuous_scale='Blues',  # Adjust the colorscale as needed
    title='Insurance Status Treemap',
    height=1000
).update_layout(
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
    )
).update_traces(
    textinfo='label+value',  # Show label, count, and percent of parent
    hovertemplate='<b>%{label}</b>: %{percent}<extra></extra>'
)

# Insurance Status Pie Chart:
insurance_pie = px.pie(
    df_insurance_counts,
    names="Individual's Insurance Status:",
    values='Count',
    color="Individual's Insurance Status:",
    title='Q1 Insurance Status',
    height=900
).update_layout(
    title_x=0.5,
    font=dict(
        family='Calibri',
        size=17,
    )
).update_traces(
    textfont=dict(size=19), 
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    textinfo='percent+value',
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# ------------------------ Location Encountered DF ----------------------- #

# Clean and standardize 'Location Encountered:' values
df['Location Encountered:'] = df['Location Encountered:'].str.strip()
df['Location Encountered:'] = df['Location Encountered:'].replace({
    'Southbridge': 'SouthBridge',
    'The Bumgalows': 'The Bungalows',
    'DACC': 'Downtown Austin Community Court',
    'SNHC': 'Sunrise Navigation Homeless Center',
    'HATC': 'Housing Authority of Travis County',
    'CFV': 'Community First Village',
    'BMHC': 'Black Men\'s Health Clinic'
})

# Convert 'Date of activity:' to datetime
df['Date of activity:'] = pd.to_datetime(df['Date of activity:'])

# Extract the month name
df['Month'] = df['Date of activity:'].dt.month_name()

# Filter for October, November, December
df_q_location = df[df['Month'].isin(['October', 'November', 'December'])]

# Group data by Month and Location Encountered
df_location_counts = (
    df_q_location.groupby(['Month', 'Location Encountered:'], 
    sort=False) # Do not sort the groups
    .size() # Count the number of occurrences
    .reset_index(name='Count') # Reset the index and rename the count column
)

# Sort months and locations
month_order = ['October', 'November', 'December'] # Define the order of months

df_location_counts['Month'] = pd.Categorical(df_location_counts['Month'], # Categorize the months for sorting purposes and to avoid alphabetical sorting.
    categories=month_order, # Specify the order of the categories
    ordered=True) # Ensure the categories are ordered

df_location_counts = df_location_counts.sort_values(['Month', 'Location Encountered:'])


# Create the grouped bar chart
location_fig = px.bar(
    df_location_counts,
    x='Month',
    y='Count',
    color='Location Encountered:',
    barmode='group',
    text='Count',
    labels={
        'Count': 'Number of Encounters',
        'Month': 'Month', # Specify the axis labels
        'Location Encountered:': 'Location' # Specify the axis labels
    },
).update_layout(
    title_x=0.5,
    yaxis_title='Number of Encounters',
    height=900,  # Adjust graph height
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    title=dict(
        text='Locations Encountered by Month',  # Title text
        font=dict(size=40),  # Font size for the title
        x=0.5  # Center-align the title
    ),
    xaxis=dict(
        title=None,
        tickangle=-15,  # Rotate x-axis labels for better readability
        tickfont=dict(size=25),  # Adjust font size for the month labels
        # title=dict(
        #     # text='Month',
        #     font=dict(size=25),  # Font size for the title
        # ),
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
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=30),  # Increase text size in each bar
    textposition='outside',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<br>'
        '<b>Count: </b>%{y}<br>'  # Count
    ),
    customdata=df_location_counts[['Location Encountered:']].values.tolist(),  # Add custom data for hover
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

df_location = df['Location Encountered:'].value_counts().reset_index(name='Count')

# Bar chart for  Totals:
location_totals_fig = px.bar(
    df_location,
    x='Location Encountered:',
    y='Count',
    color='Location Encountered:',
    text='Count',
).update_layout(
    height=850,  # Adjust graph height
    title=dict(
        x=0.5,
        text='Locations Encountered Q1 Totals',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        )
    ),
    xaxis=dict(
        tickfont=dict(size=18),  # Adjust font size for the month labels
        tickangle=-25,  # Rotate x-axis labels for better readability
        title=dict(
            text=None,
            font=dict(size=20),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Count',
            font=dict(size=22),  # Font size for the title
        ),
    ),
    bargap=0.08,  # Reduce the space between bars
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=20),  # Increase text size in each bar
    textposition='auto',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<b>Location</b>: %{label}<br><b>Count</b>: %{y}<extra></extra>'  
    ),
)

#  Pie chart:
location_pie = px.pie(
    df_location,
    names='Location Encountered:',
    values='Count',
    color='Location Encountered:',
    height=800
).update_layout(
    title=dict(
        x=0.5,
        text='Q1 Encounters',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    )  # Center-align the title
).update_traces(
    rotation=-90,
    textfont=dict(size=19),  # Increase text size in each bar
    textinfo='value+percent',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# --------------------- Person Filling Out This Form DF -------------------- #

# Clean and standardize 'Person filling out this form:' values
df['Person filling out this form:'] = df['Person filling out this form:'].str.strip()
df['Person filling out this form:'] = df['Person filling out this form:'].replace({
    'Dominique': 'Dominique Street',
    'Jaqueline Ovieod': 'Jaqueline Oviedo',
    'Sonya': 'Sonya Hosey'
})

# Convert 'Date of activity:' to datetime
df['Date of activity:'] = pd.to_datetime(df['Date of activity:'])

# Extract the month name
df['Month'] = df['Date of activity:'].dt.month_name()

# Filter for October, November, December
df_q_person = df[df['Month'].isin(['October', 'November', 'December'])]

# Group data by Month and Person filling out the form
df_person_counts = (
    df_q_person.groupby(['Month', 'Person filling out this form:'], sort=False)
    .size()
    .reset_index(name='Count')
)

# Sort months and person names
month_order = ['October', 'November', 'December']
df_person_counts['Month'] = pd.Categorical(df_person_counts['Month'], categories=month_order, ordered=True)

# Sort the dataframe by 'Month' and 'Person filling out this form:'
df_person_counts = df_person_counts.sort_values(['Month', 'Person filling out this form:'])

# Create the grouped bar chart
person_fig = px.bar(
    df_person_counts,
    x='Month',
    y='Count',
    color='Person filling out this form:',
    barmode='group', # Group bars by month
    text='Count',
    labels={ # Specify axis labels
        'Count': 'Number of Forms',
        'Month': 'Month',
        'Person filling out this form:': 'Person'
    },
).update_layout(
    title_x=0.5,
    xaxis_title='Month',
    yaxis_title='Number of Forms',
    height=900,  # Adjust graph height
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    title=dict(
        text='Person Filling Forms By Month',  # Title text
        font=dict(size=40),  # Font size for the title
        x=0.5  # Center-align the title
    ),
    xaxis=dict(
        title=dict(
            text=None,  # The title of the x-axis
            standoff=200  # Add padding between the axis title and the chart
        ),
        automargin=True,  # Ensure margins adjust automatically
        tickfont=dict(size=25)  # Font size for tick labels
    ),
    yaxis=dict(
        title=dict(
            text='Number of Submissions',
            font=dict(size=25),  # Font size for the title
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
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=30),  # Increase text size in each bar
    textposition='outside',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<br>'
        '<b>Count: </b>%{y}'
        # 'Forms Filled: %{y}<br>'  # Count
    ),
    customdata=df_person_counts[['Person filling out this form:']].values.tolist(),  # Add custom data for hover
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

df_pf = df['Person filling out this form:'].value_counts().reset_index(name='Count')

# Bar chart for  Totals:
pf_totals_fig = px.bar(
    df_pf,
    x='Person filling out this form:',
    y='Count',
    color='Person filling out this form:',
    text='Count',
).update_layout(
    height=850,  # Adjust graph height
    title=dict(
        x=0.5,
        text='Total Q1 Form Submission by Person',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        )
    ),
    xaxis=dict(
        tickfont=dict(size=18),  # Adjust font size for the month labels
        tickangle=-25,  # Rotate x-axis labels for better readability
        title=dict(
            text=None,
            font=dict(size=20),  # Font size for the title
        ),
    ),
    yaxis=dict(
        title=dict(
            text='Number of Submissions',
            font=dict(size=22),  # Font size for the title
        ),
    ),
    bargap=0.08,  # Reduce the space between bars
).update_traces(
    texttemplate='%{text}',  # Display the count value above bars
    textfont=dict(size=20),  # Increase text size in each bar
    textposition='auto',  # Automatically position text above bars
    textangle=0, # Ensure text labels are horizontal
    hovertemplate=(  # Custom hover template
        '<b>Name</b>: %{label}<br><b>Count</b>: %{y}<extra></extra>'  
    ),
)

#  Pie chart:
pf_pie = px.pie(
    df_pf,
    names='Person filling out this form:',
    values='Count',
    color='Person filling out this form:',
    height=800
).update_layout(
    title=dict(
        x=0.5,
        text='Person Submitting Forms Q1',  # Title text
        font=dict(
            size=35,  # Increase this value to make the title bigger
            family='Calibri',  # Optional: specify font family
            color='black'  # Optional: specify font color
        ),
    )  # Center-align the title
).update_traces(
    textfont=dict(size=19),  # Increase text size in each bar
    textinfo='value+percent',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>'
)

# -------------------------- Race/ Ethnicity DF ------------------------- #

# Groupby Race/Ethnicity:
df['Race/Ethnicity:'] = df['Race/Ethnicity:'].str.strip()
df['Race/Ethnicity:'] = df['Race/Ethnicity:'].replace('Hispanic/Latino', 'Hispanic/ Latino')
df['Race/Ethnicity:'] = df['Race/Ethnicity:'].replace('White', 'White/ European Ancestry')

df_race = df['Race/Ethnicity:'].value_counts().reset_index(name='Count')
df_race['Percentage'] = (df_race['Count'] / df_race['Count'].sum()) * 100
df_race['Percentage'] = df_race['Percentage'].round(0)  # Round to nearest whole number
# print(df_race)

# Group by Race/Ethnicity for October
df_race_october = df[df['Month'] == 'October']['Race/Ethnicity:'].value_counts().reset_index(name='Count')
df_race_october.rename(columns={'index': 'Race/Ethnicity'}, inplace=True)
df_race_october['Percentage'] = (df_race_october['Count'] / df_race_october['Count'].sum()) * 100
df_race_october['Percentage'] = df_race_october['Percentage'].round(0)  # Round to nearest whole number
# print(df_race_october)
# print(df.columns)

# Group by Race/Ethnicity for November
df_race_november = df[df['Month'] == 'November']['Race/Ethnicity:'].value_counts().reset_index(name='Count')
df_race_november.rename(columns={'index': 'Race/Ethnicity'}, inplace=True)
df_race_november['Percentage'] = (df_race_november['Count'] / df_race_november['Count'].sum()) * 100
df_race_november['Percentage'] = df_race_november['Percentage'].round(0)  # Round to nearest whole number
# print(df_race_november)

# Group by Race/Ethnicity for December
df_race_december = df[df['Month'] == 'December']['Race/Ethnicity:'].value_counts().reset_index(name='Count')
df_race_december.rename(columns={'index': 'Race/Ethnicity'}, inplace=True)
df_race_december['Percentage'] = (df_race_december['Count'] / df_race_december['Count'].sum()) * 100
df_race_december['Percentage'] = df_race_december['Percentage'].round(0)  # Round to nearest whole number

# print(df_race_december[['Race/Ethnicity:', 'Count', 'Percentage']])


race_fig = px.pie(
    df_race,
    names='Race/Ethnicity:',
    values='Count'
).update_layout(
    title='Q1 Client Visits by Race',
    title_x=0.5,
    height=700,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    # hide legend
    showlegend=True
).update_traces(
    textinfo='label+percent+value',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>',
    texttemplate='%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_race['Percentage']  # Pass calculated percentage as custom data
)

# Race fig for October
race_fig_oct = px.pie(
    df_race_october,
    names='Race/Ethnicity:',
    values='Count'
).update_layout(
    title='Client Visits by Race October',
    title_x=0.5,
    height=700,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    # hide legend
    showlegend=False
).update_traces(
    textinfo='label+percent+value',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>',
    texttemplate='%{label}:<br>%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_race_october['Percentage']  # Pass calculated percentage as custom data
)

# Race fig for October
race_fig_nov = px.pie(
    df_race_november,
    names='Race/Ethnicity:',
    values='Count'
).update_layout(
    title='Client Visits by Race November',
    title_x=0.5,
    height=700,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    # hide legend
    showlegend=False
).update_traces(
    textinfo='label+percent+value',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>',
    texttemplate='%{label}:<br>%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_race_november['Percentage']  # Pass calculated percentage as custom data
)

# Race fig for December
race_fig_dec = px.pie(
    df_race_december,
    names='Race/Ethnicity:',
    values='Count'
).update_layout(
    title='Client Visits by Race December',
    title_x=0.5,
    height=700,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    ),
    # hide legend
    showlegend=False
).update_traces(
    textinfo='label+percent+value',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>',
    texttemplate='%{label}:<br>%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_race_december['Percentage']  # Pass calculated percentage as custom data
)

# ----------------------- New/ Returning Stattus DF ------------------------- #

# Group by 'Individual\'s Status:' for all months and calculate percentages
df_status = df['Individual\'s Status:'].value_counts().reset_index(name='Count')
df_status['Percentage'] = (df_status['Count'] / df_status['Count'].sum()) * 100
df_status['Percentage'] = df_status['Percentage'].round(0)  # Round to nearest whole number

df_status_oct = df[df['Month'] == 'October']['Individual\'s Status:'].value_counts().reset_index(name='Count')
df_status_oct['Percentage'] = (df_status_oct['Count'] / df_status_oct['Count'].sum()) * 100
df_status_oct['Percentage'] = df_status_oct['Percentage'].round(0)  # Round to nearest whole number

df_status_nov = df[df['Month'] == 'November']['Individual\'s Status:'].value_counts().reset_index(name='Count')
df_status_nov['Percentage'] = (df_status_nov['Count'] / df_status_nov['Count'].sum()) * 100
df_status_nov['Percentage'] = df_status_nov['Percentage'].round(0)  # Round to nearest whole number

df_status_dec = df[df['Month'] == 'December']['Individual\'s Status:'].value_counts().reset_index(name='Count')
df_status_dec['Percentage'] = (df_status_dec['Count'] / df_status_dec['Count'].sum()) * 100
df_status_dec['Percentage'] = df_status_dec['Percentage'].round(0)  # Round to nearest whole number


# Pie chart for the total 'Individual\'s Status:'
status_fig = px.pie(
    df_status,
    names='Individual\'s Status:',
    values='Count'
).update_layout(
    title='Q1 New vs. Returning',
    title_x=0.5,
    height=700,
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

# Pie chart for October
status_fig_oct = px.pie(
    df_status_oct,
    names='Individual\'s Status:',
    values='Count'
).update_layout(
    title='New vs. Returning October',
    title_x=0.5,
    height=700,
    showlegend=False,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    textinfo='label+percent+value',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>',
    texttemplate='%{label}:<br>%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_status_oct['Percentage']  # Pass calculated percentage as custom data
)

# Pie chart for November
status_fig_nov = px.pie(
    df_status_nov,
    names='Individual\'s Status:',
    values='Count'
).update_layout(
    title='New vs. Returning November',
    title_x=0.5,
    height=700,
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
    texttemplate='%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_status_nov['Percentage']  # Pass calculated percentage as custom data
)

# Pie chart for December
status_fig_dec = px.pie(
    df_status_dec,
    names='Individual\'s Status:',
    values='Count'
).update_layout(
    title='New vs. Returning December',
    title_x=0.5,
    height=700,
    showlegend=False,
    font=dict(
        family='Calibri',
        size=17,
        color='black'
    )
).update_traces(
    textinfo='label+percent+value',
    insidetextorientation='horizontal',  # Force text labels to be horizontal
    hovertemplate='<b>%{label}</b>: %{value}<extra></extra>',
    texttemplate='%{label}:<br>%{value}<br>%{customdata:.0f}%',  # Using manually calculated percentage
    customdata=df_status_dec['Percentage']  # Pass calculated percentage as custom data
)

# ------------------------ ZIP2 DF ----------------------- #

# make a copy of 'ZIP Code:' column to 'ZIP2':
df['ZIP2'] = df['ZIP Code:']
df['ZIP2'] = df['ZIP2'].astype(str).str.strip()
df['ZIP2'] = df['ZIP2'].fillna(df['ZIP2'].mode()[0])
df['ZIP2'] = df['ZIP2'].replace('UNHOUSED', df['ZIP Code:'].mode()[0])
df['ZIP2'] = df['ZIP2'].replace('Texas', df['ZIP Code:'].mode()[0])
df['ZIP2'] = df['ZIP2'].replace('nan', df['ZIP Code:'].mode()[0])
df['ZIP2'] = df['ZIP2'].astype(str)
df_z = df['ZIP2'].value_counts().reset_index(name='Count')
# print(df_z.value_counts())


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

# mode_value = df['ZIP Code:'].mode()[0]
# df['ZIP Code:'].fillna(mode_value, inplace=True)
# df['ZIP Code:'] = df['ZIP Code:'].replace('nan', mode_value)
# df['ZIP Code:'] = df['ZIP Code:'].replace("UNHOUSED", mode_value)
# df['ZIP Code:'] = df['ZIP Code:'].astype('Int64')
# df['ZIP Code:'] = df['ZIP Code:'].replace(-1, mode_value)
# # df_q1['ZIP Code:'].fillna(df_q1['ZIP Code:'].mode()[0], inplace=True)
# # print(df['ZIP Code:'].value_counts())

# # Count of visitors by zip code
# df_zip = df['ZIP Code:'].value_counts().reset_index(name='Residents')
# df_zip['ZIP Code:'] = df_zip['ZIP Code:'].astype(int)
# df_zip['Residents'] = df_zip['Residents'].astype(int)

# # Create a folium map
# m = folium.Map([30.2672, -97.7431], zoom_start=10)

# # Add different tile sets
# folium.TileLayer('OpenStreetMap', attr='© OpenStreetMap contributors').add_to(m)
# folium.TileLayer('Stamen Terrain', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
# folium.TileLayer('Stamen Toner', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
# folium.TileLayer('Stamen Watercolor', attr='Map tiles by Stamen Design, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
# folium.TileLayer('CartoDB positron', attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)
# folium.TileLayer('CartoDB dark_matter', attr='Map tiles by CartoDB, under CC BY 3.0. Data by OpenStreetMap, under ODbL.').add_to(m)

# # Available map styles
# map_styles = {
#     'OpenStreetMap': {
#         'tiles': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
#         'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
#     },
#     'Stamen Terrain': {
#         'tiles': 'https://stamen-tiles.a.ssl.fastly.net/terrain/{z}/{x}/{y}.jpg',
#         'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
#     },
#     'Stamen Toner': {
#         'tiles': 'https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png',
#         'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
#     },
#     'Stamen Watercolor': {
#         'tiles': 'https://stamen-tiles.a.ssl.fastly.net/watercolor/{z}/{x}/{y}.jpg',
#         'attribution': 'Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under ODbL.'
#     },
#     'CartoDB positron': {
#         'tiles': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
#         'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
#     },
#     'CartoDB dark_matter': {
#         'tiles': 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
#         'attribution': '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
#     },
#     'ESRI Imagery': {
#         'tiles': 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
#         'attribution': 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
#     }
# }

# # Add tile layers to the map
# for style, info in map_styles.items():
#     folium.TileLayer(tiles=info['tiles'], attr=info['attribution'], name=style).add_to(m)

# # Select a style
# # selected_style = 'OpenStreetMap'
# # selected_style = 'Stamen Terrain'
# # selected_style = 'Stamen Toner'
# # selected_style = 'Stamen Watercolor'
# selected_style = 'CartoDB positron'
# # selected_style = 'CartoDB dark_matter'
# # selected_style = 'ESRI Imagery'

# # Apply the selected style
# if selected_style in map_styles:
#     style_info = map_styles[selected_style]
#     # print(f"Selected style: {selected_style}")
#     folium.TileLayer(
#         tiles=style_info['tiles'],
#         attr=style_info['attribution'],
#         name=selected_style
#     ).add_to(m)
# else:
#     print(f"Selected style '{selected_style}' is not in the map styles dictionary.")
#      # Fallback to a default style
#     folium.TileLayer('OpenStreetMap').add_to(m)

# # Function to get coordinates from zip code
# def get_coordinates(zip_code):
#     geolocator = Nominatim(user_agent="response_q4_2024.py")
#     location = geolocator.geocode({"postalcode": zip_code, "country": "USA"})
#     if location:
#         return location.latitude, location.longitude
#     else:
#         print(f"Could not find coordinates for zip code: {zip_code}")
#         return None, None

# # Apply function to dataframe to get coordinates
# df_zip['Latitude'], df_zip['Longitude'] = zip(*df_zip['ZIP Code:'].apply(get_coordinates))

# # Filter out rows with NaN coordinates
# df_zip = df_zip.dropna(subset=['Latitude', 'Longitude'])
# # print(df_zip.head())
# # print(df_zip[['Zip Code', 'Latitude', 'Longitude']].head())
# # print(df_zip.isnull().sum())

# # instantiate a feature group for the incidents in the dataframe
# incidents = folium.map.FeatureGroup()

# for index, row in df_zip.iterrows():
#     lat, lng = row['Latitude'], row['Longitude']

#     if pd.notna(lat) and pd.notna(lng):  
#         incidents.add_child(# Check if both latitude and longitude are not NaN
#         folium.vector_layers.CircleMarker(
#             location=[lat, lng],
#             radius=row['Residents'] * 1.2,  # Adjust the multiplication factor to scale the circle size as needed,
#             color='blue',
#             fill=True,
#             fill_color='blue',
#             fill_opacity=0.4
#         ))

# # add pop-up text to each marker on the map
# latitudes = list(df_zip['Latitude'])
# longitudes = list(df_zip['Longitude'])

# # labels = list(df_zip[['Zip Code', 'Residents_In_Zip_Code']])
# labels = df_zip.apply(lambda row: f"ZIP Code: {row['ZIP Code:']}, Patients: {row['Residents']}", axis=1)

# for lat, lng, label in zip(latitudes, longitudes, labels):
#     if pd.notna(lat) and pd.notna(lng):
#         folium.Marker([lat, lng], popup=label).add_to(m)
 
# formatter = "function(num) {return L.Util.formatNum(num, 5);};"
# mouse_position = MousePosition(
#     position='topright',
#     separator=' Long: ',
#     empty_string='NaN',
#     lng_first=False,
#     num_digits=20,
#     prefix='Lat:',
#     lat_formatter=formatter,
#     lng_formatter=formatter,
# )

# m.add_child(mouse_position)

# # add incidents to map
# m.add_child(incidents)

# map_path = 'zip_code_map.html'
# map_file = os.path.join(script_dir, map_path)
# m.save(map_file)
# map_html = open(map_file, 'r').read()

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
                'BMHC Client Navigation Report Q1 2025', 
                className='title'),
            html.H1(
                '10/01/2024 - 12/31/2024', 
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
            className='graph11',
            children=[
                html.Div(
                    className='high1',
                    children=['Total Clients Served Q1:']
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
            className='graph2',
            children=[
                dcc.Graph(
                    figure=clients_served_fig
                )
            ]
        ),
    ]
),

# html.Div(
#     className='row1',
#     children=[
#         html.Div(
#             className='graph11',
#             children=[
#                 html.Div(
#                     className='high1',
#                     children=['Clients Served November:']
#                 ),
#                 html.Div(
#                     className='circle1',
#                     children=[
#                         html.Div(
#                             className='hilite',
#                             children=[
#                                 html.H1(
#                                     className='high2',
#                                     children=[clients_served_nov]
#                                 ),
#                             ]
#                         ),
#                     ],
#                 ),
#             ],
#         ),
#         html.Div(
#             className='graph22',
#             children=[
#                 html.Div(
#                     className='high1',
#                     children=['Clients Served December:']
#                 ),
#                 html.Div(
#                     className='circle1',
#                     children=[
#                         html.Div(
#                             className='hilite',
#                             children=[
#                                 html.H1(
#                                     className='high2',
#                                     children=[clients_served_dec]
#                                 ),
#                             ]
#                         ),
#                     ],
#                 ),
#             ],
#         ),
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
                    children=['Total Navigation Hours Q1:']
                ),
                html.Div(
                    className='circle2',
                    children=[
                        html.Div(
                            className='hilite',
                            children=[
                                html.H1(
                                    className='high3',
                                    children=[df_duration]
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
                    children=['Navigation Hours October:']
                ),
                html.Div(
                    className='circle2',
                    children=[
                        html.Div(
                            className='hilite',
                            children=[
                                html.H1(
                                    className='high3',
                                    children=[df_duration_oct]
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
                    children=['Navigation Hours November:']
                ),
                html.Div(
                    className='circle2',
                    children=[
                        html.Div(
                            className='hilite',
                            children=[
                                html.H1(
                                    className='high3',
                                    children=[df_duration_nov]
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
                    children=['Navigation Hours December:']
                ),
                html.Div(
                    className='circle2',
                    children=[
                        html.Div(
                            className='hilite',
                            children=[
                                html.H1(
                                    className='high3',
                                    children=[df_duration_dec]
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
                    figure=race_fig
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    figure=race_fig_oct
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
                    figure=race_fig_nov
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    figure=race_fig_dec
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
                    figure=status_fig_oct
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
                    figure=status_fig_nov
                )
            ]
        ),
        html.Div(
            className='graph2',
            children=[
                dcc.Graph(
                    figure=status_fig_dec
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
                    figure=age_totals_fig
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
                    figure=support_totals_fig
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
                    # figure=insurance_fig
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
                    # figure=insurance_fig
                    figure=insurance_totals_fig
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
                    # figure=insurance_fig
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
                    figure=location_totals_fig
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
                    figure=pf_totals_fig
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
                # Horizontal Bar chart for zip code:
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
                    # srcDoc=map_html
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