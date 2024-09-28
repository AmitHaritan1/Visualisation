import streamlit as st
import pandas as pd
# import matplotlib.pyplot as plt
# from bidi.algorithm import get_display

import numpy as np
import requests

import folium
# from folium import plugins
# from matplotlib import cm
from matplotlib.colors import Normalize
from streamlit_folium import folium_static

# import folium
import matplotlib
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px


def fetch_data_and_create_df(api_dict):
    """
    Fetches data from APIs and creates pandas DataFrames.

    Parameters:
    - api_dict (dict): A dictionary where keys are table names and values are resource IDs.

    Returns:
    - df_dict (dict): A dictionary where keys are table names and values are pandas DataFrames.
    """
    df_dict = {}

    for table_name, resource_id in api_dict.items():
        url = 'https://data.gov.il/api/3/action/datastore_search'
        params = {'resource_id': resource_id}  # You can adjust the limit as needed
        response = requests.get(url, params=params)

        if response.status_code == 200:
            json_data = response.json()
            records = json_data['result']['records']
            df_dict[table_name] = pd.DataFrame(records)
        else:
            print(f'Failed to retrieve data from {table_name}: {response.status_code}')

    return df_dict

def plot_top5_waste_types(city, measurer_type, k):
    """
    :param city:
    :param measurer_type:
    :param k:
    """
    israel_filtered = df[df['סוג נקודת המדידה'].isin(measurer_type)]
    israel_avg_values = israel_filtered[columns_to_convert].mean()

    if city != 'כל הארץ':
        # Filter the data for the selected יישוב and סוג נקודת המדידה
        df_filtered = df[(df['יישוב'] == city) & (df['סוג נקודת המדידה'].isin(measurer_type))]
        # Calculate the average for each column
        avg_values = df_filtered[columns_to_convert].mean()
        total_diff = avg_values - israel_avg_values
    else:
        # Calculate the average for each column
        israel_avg_values = israel_filtered[columns_to_convert].mean()
        total_diff = israel_avg_values
        avg_values = israel_avg_values

    # Get the top k highest average columns
    topk = total_diff.nlargest(k)

    # Convert Hebrew text to right-to-left for the labels
    labels = [label for label in topk.index]

    # Get the corresponding colors for the top k columns
    topk_colors = [color_map[column] for column in topk.index]

    # Create the figure with Plotly graph_objects
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=topk.values,
        marker_color=topk_colors,
        hovertemplate=(
                '<b>%{x}</b><br>' +  # Column label (waste type)
                f'Average in {city}:' + '%{y:.2f}<br>' +  # City average
                'National Average: %{customdata[0]:.2f}<br>' +  # National average
                'Difference: %{customdata[1]:.2f}<extra></extra>'  # Difference between city and national
        ),
        customdata=np.stack((avg_values[topk.index], israel_avg_values[topk.index], topk.values), axis=-1)  # National average and difference
    )])

    # Update layout for Hebrew labels
    fig.update_layout(
        title={
            'text': f'חמשת סוגי הפסולת המובילים עבור {city}',
            'x': 1.0,  # Align to the right
            'xanchor': 'right'
        },
        xaxis_title='סוג פסולת',
        yaxis_title='ערך ממוצע',
        font=dict(family='Arial, sans-serif', size=14),
        xaxis_tickangle=-45,
        bargap=0.2,
        height=400
    )

    # Show the interactive Plotly plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def calculate_avg_coords(df):
    """
    Function to calculate average coordinates of all city_df coordinates
    :param df:
    :return:
    """

    latitudes = []
    longitudes = []

    for coords in df['נ.צ כתובת']:
        try:
            lat, lon = map(float, coords.split(','))
            latitudes.append(lat)
            longitudes.append(lon)
        except ValueError:
            continue

    return [sum(latitudes) / len(latitudes), sum(longitudes) / len(longitudes)] if latitudes else [0, 0]

def update_map(city, measurer_type_list):
    # Function to update the map based on the selected city and selected columns
    # Filter data for the selected city
    if city != 'כל הארץ':
        # Filter the data for the selected city
        city_df = filtered_df[filtered_df['יישוב'] == city]
    else:
        city_df = df

    # Recalculate average coordinates for the selected city
    if not city_df.empty and city != 'כל הארץ':
        avg_coords = calculate_avg_coords(city_df)
        zoom_start = 12
    else:
        avg_coords = [31.813, 35.163]  # Default coordinates if no data
        zoom_start = 10

    # Create a new map centered at the average coordinates of the selected city
    mymap = folium.Map(location=avg_coords, zoom_start=zoom_start)

    # Calculate min and max values across selected columns
    min_value = city_df[list(measurer_type_list)].min().min()  # Convert to list for selection
    max_value = city_df[list(measurer_type_list)].max().max()

    # Add CircleMarkers to the map with color based on the average of the selected columns
    for index, row in city_df.iterrows():
        try:
            lat, lon = map(float, row['נ.צ כתובת'].split(','))

            # Calculate the average of the selected columns
            avg_value = row[list(measurer_type_list)].mean()  # Convert to list for selection

            # Get the color for the CircleMarker based on the average value
            color = get_color_scale(avg_value, min_value, max_value)

            # Add a CircleMarker with the calculated color
            folium.CircleMarker(
                location=[lat, lon],
                radius=7,  # Size of the marker
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Average Value: {avg_value:.2f}"
            ).add_to(mymap)
        except ValueError:
            continue

    # Return the map
    return mymap

def plot_infrastructure_condition(df, city):
    if city != 'כל הארץ':
        # Filter the data for the selected city
        df_filtered = df[df['יישוב'] == city]
    else:
        df_filtered = df

    # Define the columns to plot
    columns_to_plot = ['מדרכה', 'אבנישפה', 'גדרות', 'צמחייה']

    # Define the condition categories and their corresponding colors
    condition_categories = ['לא תקין (מוזנח)', 'סביר', 'תקין (מטופח)','לא רלוונטי',]
    color_map = {
        'לא רלוונטי':'#b0b0b0',  # Darker gray
        'לא תקין (מוזנח)': '#fc5858',  # Darker red
        'סביר': '#f0e68c',  # Light khaki (Darker yellow)
        'תקין (מטופח)': '#98df8a'  # The specified green
    }

    # Initialize an empty dictionary to hold counts for each condition category
    condition_counts = {condition: [] for condition in condition_categories}

    # Loop over each column and count the occurrences of each condition
    for col in columns_to_plot:
        for condition in condition_categories:
            count = df_filtered[df_filtered[col] == condition].shape[0]
            condition_counts[condition].append(count)

    # Create the stacked bar plot
    fig = go.Figure()

    # Add bars for each condition category with corresponding colors
    for condition, counts in condition_counts.items():
        fig.add_trace(go.Bar(
            x=columns_to_plot,
            y=counts,
            name=condition,  # Legend will now show the condition categories
            marker_color=color_map[condition]  # Use the color map for coloring
        ))

    # Update layout for stacked bars and Hebrew text
    fig.update_layout(
        barmode='stack',
        title={
            'text': f'מצב התשתיות עבור {city}',
            'x': 1.0,  # Align to the right
            'xanchor': 'right'
        },
        xaxis_title='קטגוריות',
        yaxis_title='כמות',
        font=dict(family='Arial, sans-serif', size=14),
        xaxis_tickangle=-45,
        height=400,
        legend_title='מצב'
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def plot_behaviors_teorshlilihiyuvi(city):
    # Filter the data for the selected city
    st.markdown("""
        <style>
        .stRadio > label {
            float: right;  /* Align the label text to the right */
        }
        .stRadio [role=radiogroup] {
            justify-content: flex-end;  /* Align the radio buttons to the right */
        }
        </style>
    """, unsafe_allow_html=True)

    x_axis_val = st.radio(
        'בחר עמודת ציר איקס',  # Label for the radio buttons in Hebrew
        ('מגדר', 'גיל') # Options for the radio buttons
        ,horizontal=True
    )
    x_axis_dict = {'מגדר':'gender', 'גיל':'age'}
    x_axis_col = x_axis_dict[x_axis_val]

    if city != 'כל הארץ':
        # Filter the data for the selected city
        df_filtered = data_frames['behaviors'][data_frames['behaviors']['יישוב'] == city]
    else:
        df_filtered = data_frames['behaviors']


    # Group by the chosen x-axis column (either 'gender' or 'age') and count occurrences of 'teorshlilihiyuvi'
    grouped_data = df_filtered.groupby([x_axis_col, 'teorshlilihiyuvi']).size().unstack().fillna(0)

    # Create the figure with Plotly graph_objects
    fig = go.Figure()

    # Add bars for each 'teorshlilihiyuvi' category
    for value in grouped_data.columns:
        fig.add_trace(go.Bar(
            x=grouped_data.index,
            y=grouped_data[value],
            name=f' התנהגות {value}'
        ))

    title_text =  ' פיזור תיאור שלילי חיובי לפי ' + x_axis_val
    subtitle_text = ' עבור ' + city
    # Update layout for Hebrew labels and other visual elements
    fig.update_layout(
        title={
            'text': f"{title_text} <br><sup>{subtitle_text}</sup>",
            'x': 1.0,  # Align to the right
            'xanchor': 'right'
        },
        xaxis_title=x_axis_val,
        yaxis_title='ספירת מופעים',
        font=dict(family='Arial, sans-serif', size=14),
        xaxis_tickangle=-45,
        bargap=0.2,
        height=600
    )
    # Show the interactive Plotly plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def get_color_scale(avg_value, min_value, max_value):
    """
    Function to normalize selected values and map them to a color scale
    :param avg_value:
    :param min_value:
    :param max_value:
    :return:
    """
    norm = Normalize(vmin=min_value, vmax=max_value)
    cmap = matplotlib.colormaps.get_cmap('RdYlGn_r')
    return cm.colors.rgb2hex(cmap(norm(avg_value)))


def plot_top_k_behaviors(city, k):
    # Filter the dataframe for the chosen city
    df = data_frames['behaviors']
    if city != 'כל הארץ':
        # Filter the data for the selected city
        df_filtered = df[df['יישוב'] == city]
    else:
        df_filtered = df
    # List of the 'heged' columns
    heged_columns = [f'heged{i}' for i in range(1, 14)]
    # Melt the dataframe to turn all 'heged' columns into rows (ignoring null values)
    df_melted = df_filtered.melt(id_vars=['יישוב', 'point_type'], value_vars=heged_columns,
                        var_name='heged_type', value_name='phrase').dropna(subset=['phrase'])

    df_melted = df_melted[df_melted['phrase'].notnull() & (df_melted['phrase'] != '')]

    # Count the occurrences of each phrase
    phrase_counts = df_melted['phrase'].value_counts()
    # Get the top k phrases
    top_k_phrases = phrase_counts.nlargest(k)
    others_count = phrase_counts.sum() - top_k_phrases.sum()

    # Combine the top k phrases with the "Others" slice
    all_phrases = pd.concat([top_k_phrases, pd.Series({'Others': others_count})])

    # Calculate the percentage of each phrase, including "Others"
    total_phrases_count = phrase_counts.sum()
    phrases_percentage = (all_phrases / total_phrases_count) * 100

    # Create a donut chart using Plotly
    fig = px.pie(
        names=phrases_percentage.index,
        values=phrases_percentage.values,
        hole=0.4,  # Creates the donut hole
        title=f'Top {k} Phrases and Others in {city}',  # This title will be overridden
        labels={'phrase': 'Phrase', 'count': 'Count'},
    )

    # Update the layout to customize the title
    fig.update_layout(
        title={
            'text': f' {k} הביטויים המובילים ו"אחרים" ב-{city}',
            'x': 1.0,  # Align to the right
            'xanchor': 'right'
        }
    )
    fig.update_traces(sort=False, direction="clockwise")
    fig.update_traces(textinfo='percent', texttemplate='%{percent:.1%}')

    # Show the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def plot_waste_levels_by_city(city):
    # Define the fixed waste levels in the correct order
    waste_levels_order = ['ריק', '1/4', '1/2', '3/4', 'מלא']

    # Filter the data where 'האם יש פחים בנקודת המדידה' is 'כן' and the selected city
    if city != 'כל הארץ':
        # Filter the data for the selected city
        df_filtered = data_frames['bin_storage'][
            (data_frames['bin_storage']['האם יש פחים בנקודת המדידה'] == 'כן') &
            (data_frames['bin_storage']['יישוב'] == city)
            ]
    else:
        df_filtered = data_frames['bin_storage'][
        (data_frames['bin_storage']['האם יש פחים בנקודת המדידה'] == 'כן')]

    # Count occurrences of each 'מפלס הפסולת בפח' level
    waste_level_counts = df_filtered['מפלס הפסולת בפח'].value_counts()

    # Reindex the series to ensure all levels are present, even if their count is zero
    waste_level_counts = waste_level_counts.reindex(waste_levels_order, fill_value=0)

    # Convert Hebrew text to right-to-left for the labels (this ensures the order is correct in Hebrew)
    labels = waste_levels_order

    # Optional: Define custom colors if needed (replace color_map with your color choices)
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A'][:len(waste_levels_order)]

    # Create the figure with Plotly graph_objects
    fig = go.Figure(data=[go.Bar(x=labels, y=waste_level_counts.values, marker_color=colors)])

    # Update layout for Hebrew labels
    fig.update_layout(
        title={
            'text': f'מפלסי הפסולת בפחים עבור {city}',
            'x': 1.0,  # Align to the right
            'xanchor': 'right'
        },
        xaxis_title='מפלס הפסולת בפח',
        yaxis_title='כמות',
        font=dict(family='Arial, sans-serif', size=14),
        xaxis_tickangle=-45,
        bargap=0.2,
        height=600
    )

    # Show the interactive Plotly plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def prepare_data(data_frame):
    """
    Function to prepare the data
    :param data_frame:
    :return:
    """
    bin_storage_df = data_frames['bin_storage']
    behaviors_df = data_frames['behaviors']
    infrastructures_df = data_frames['infrastructures']
    dirt_information_df = data_frames['dirt_information']
    # prepare dirt_information
    columns_to_convert = [
        'בדלי סיגריות', 'קופסאות סיגריות', 'מסכות כירורגיות',
        'מכלי משקה למיניהם', 'פקקים של מכלי משקה', 'אריזות מזון Take Away נייר',
        'אריזות מזון Take Away פלסטיק', 'צלחות חדפ', 'סכום חדפ',
        'כוסות שתייה קרה חדפ', 'כוסות שתייה חמה חדפ', 'אריזות של חטיפים',
        'זכוכית לא מכלי משקה או לא ניתן לזיה', 'נייר אחר לא אריזות מזון',
        'פלסטיק אחר שקיות פלסטיק ורכיבי פלס', 'פסולת אורגנית', 'פסולת בלתי חוקית שקית אשפה מלאה שה',
        'פסולת אחרת למשל בגדים סוללות חומרי', 'צואת כלבים', 'כתמי מסטיק',
        'פריט פסולת גדול', 'אריזות קרטון', 'גרפיטי', 'אחר1', 'אחר2'
    ]

    dirt_information_df.dropna(subset=['נ.צ כתובת', 'יישוב'], inplace=True)
    dirt_information_df[columns_to_convert] = dirt_information_df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    return bin_storage_df, behaviors_df, infrastructures_df, dirt_information_df


if __name__ == '__main__':
    # Dictionary containing table names and resource IDs
    api_resource_ids = {
        'bin_storage': '3436bb7f-8b67-49be-94a4-3c34c9cc1e7a',
        'behaviors': 'ece44fa9-f47c-4116-a16e-9477e0d4d2dc',
        'infrastructures': '7b32b590-d130-4f41-bb63-1ca462d91f3a',
        'dirt_information': '94400680-6a74-4c4b-be55-704e20ca4e76'
    }
    # Fetch data and create DataFrames

    data_frames = fetch_data_and_create_df(api_resource_ids)
    bin_storage_df, behaviors_df, infrastructures_df, dirt_information_df = prepare_data(data_frames)

    columns_to_convert = [
        'בדלי סיגריות', 'קופסאות סיגריות', 'מסכות כירורגיות',
        'מכלי משקה למיניהם', 'פקקים של מכלי משקה', 'אריזות מזון Take Away נייר',
        'אריזות מזון Take Away פלסטיק', 'צלחות חדפ', 'סכום חדפ',
        'כוסות שתייה קרה חדפ', 'כוסות שתייה חמה חדפ', 'אריזות של חטיפים',
        'זכוכית לא מכלי משקה או לא ניתן לזיה', 'נייר אחר לא אריזות מזון',
        'פלסטיק אחר שקיות פלסטיק ורכיבי פלס', 'פסולת אורגנית', 'פסולת בלתי חוקית שקית אשפה מלאה שה',
        'פסולת אחרת למשל בגדים סוללות חומרי', 'צואת כלבים', 'כתמי מסטיק',
        'פריט פסולת גדול', 'אריזות קרטון', 'גרפיטי', 'אחר1', 'אחר2'
    ]

    color_map = {
        'בדלי סיגריות': '#1f77b4',
        'קופסאות סיגריות': '#ff7f0e',
        'מסכות כירורגיות': '#2ca02c',
        'מכלי משקה למיניהם': '#d62728',
        'פקקים של מכלי משקה': '#9467bd',
        'אריזות מזון Take Away נייר': '#8c564b',
        'אריזות מזון Take Away פלסטיק': '#e377c2',
        'צלחות חדפ': '#7f7f7f',
        'סכום חדפ': '#bcbd22',
        'כוסות שתייה קרה חדפ': '#17becf',
        'כוסות שתייה חמה חדפ': '#ffbb78',
        'אריזות של חטיפים': '#98df8a',
        'זכוכית לא מכלי משקה או לא ניתן לזיה': '#ff9896',
        'נייר אחר לא אריזות מזון': '#c49c94',
        'פלסטיק אחר שקיות פלסטיק ורכיבי פלס': '#f7b6d2',
        'פסולת אורגנית': '#dbdb8d',
        'פסולת בלתי חוקית שקית אשפה מלאה שה': '#c5b0d5',
        'פסולת אחרת למשל בגדים סוללות חומרי': '#9edae5',
        'צואת כלבים': '#ffbb78',
        'כתמי מסטיק': '#aec7e8',
        'פריט פסולת גדול': '#ffbb78',
        'אריזות קרטון': '#f7b6d2',
        'גרפיטי': '#98df8a',
        'אחר1': '#ff9896',
        'אחר2': '#c49c94'
    }

    data = data_frames['dirt_information']
    # Drop rows where 'נ.צ כתובת' or 'יישוב' is null
    filtered_df = data.dropna(subset=['נ.צ כתובת', 'יישוב'])
    cities_sort = ['כל הארץ'] + sorted(filtered_df['יישוב'].unique())
    measurer_type_list = filtered_df['סוג נקודת המדידה'].unique()

    df = filtered_df
    # Assuming df is your DataFrame
    df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

    # st.set_page_config(layout="wide")
    st.set_page_config(
        page_title="המשרד להגנת הסביבה - ניקיון במרחב הציבורי",
        page_icon="🚮",
        layout="wide",
        initial_sidebar_state="expanded")

    st.markdown(
        """
        <style>
        .container {
            display: flex;
            justify-content: flex-end;
        }
        </style>
        <div class="container">
            <h1>🚮 המשרד להגנת הסביבה - ניקיון במרחב הציבורי</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sample DataFrame (replace with your actual data)
    data = data_frames['dirt_information']
    # Get unique cities from the 'יישוב' column
    cities = sorted(filtered_df['יישוב'].unique())
    # Convert specified columns to numeric

    # Split screen into two columns
    col1, col2 = st.columns([3, 3])

    st.markdown(
        """
        <style>
            div[data-testid="column"]:nth-of-type(1)
            {
                text-align: end;
            } 
    
            div[data-testid="column"]:nth-of-type(2)
            {
                text-align: end;
            } 
            .right-align {
            display: flex;
            justify-content: flex-end;
        }
        </style>
        """,unsafe_allow_html=True
    )

    with st.sidebar:
        st.markdown(
            """
            <style>
                section[data-testid="stSidebar"] {
                    width: 100px !important; # Set the width to your desired value
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        # Dropdown for city selection
        st.markdown('<div class="right-align">', unsafe_allow_html=True)

        city = st.selectbox('בחר/י עיר', cities_sort)
        k = st.selectbox('בחר/י כמות להצגה', [3, 5, 10])
        measurer_type = st.multiselect('בחר סוג מדידה:', measurer_type_list, measurer_type_list)

    with col1:
        # st.header('Top Waste Types')
        plot_top5_waste_types(city, measurer_type, k)
        plot_infrastructure_condition(infrastructures_df, city)
        plot_behaviors_teorshlilihiyuvi(city)
        plot_top_k_behaviors(city,k)
        plot_waste_levels_by_city(city)

    # In the second column, show the map
    with col2:
        st.header('מיפוי נקודות המדידה')
        # Multiselect for selecting waste types (columns)
        selected_columns = st.multiselect(
            'בחר/י את סוג הפסולת:',
            columns_to_convert,
            default=['בדלי סיגריות']  # Default selection
        )
        # plot_map(city, ['בדלי סיגריות', 'קופסאות סיגריות'])  # You can adjust selected columns here
        folium_map = update_map(city, selected_columns)
        folium_static(folium_map)
