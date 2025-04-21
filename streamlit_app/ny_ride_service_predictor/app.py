import streamlit as st
import openai
from openai import OpenAI
import pandas as pd
import pickle
from datetime import datetime
import holidays
from geopy.distance import geodesic
import requests
import folium
from streamlit_folium import st_folium
import os  
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import speech_recognition as sr
from langchain.schema import Document
from pydub import AudioSegment
import tempfile
import io
#streamlit_app/ny_ride_service_predictor/

try:
    with open('streamlit_app/ny_ride_service_predictor/xgbr_price.pkl', 'rb') as f:
        xgbr = pickle.load(f)
    with open('streamlit_app/ny_ride_service_predictor/xgbr_duration.pkl', 'rb') as f:
        xgbr2 = pickle.load(f)
    with open('streamlit_app/ny_ride_service_predictor/encoder_pu_price.pkl', 'rb') as f:
        encoder_pu_price = pickle.load(f)
    with open('streamlit_app/ny_ride_service_predictor/encoder_do_price.pkl', 'rb') as f:
        encoder_do_price = pickle.load(f)
    with open('streamlit_app/ny_ride_service_predictor/encoder_pu_duration.pkl', 'rb') as f:
        encoder_pu_duration = pickle.load(f)
    with open('streamlit_app/ny_ride_service_predictor/encoder_do_duration.pkl', 'rb') as f:
        encoder_do_duration = pickle.load(f)
    with open('streamlit_app/ny_ride_service_predictor/coords_df.pkl', 'rb') as f:
        coords_df = pickle.load(f)
except FileNotFoundError:
    st.error("One or more necessary data files were not found. Please check the file paths.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading data files: {e}")
    st.stop()


features = [
            'second_of_day',
            'day_of_year',
            'weekend',
            'holiday',
            'morning_rush',
            'evening rush',
            'prcp',
            'temp',
            'distance',
            'airport',
            'congestion',
            'class_0',  
            'class_1',
            'class_2',
            'PULocationID_price_encoded',
            'DOLocationID_price_encoded']

duration_features = [
            'second_of_day',
            'day_of_year',
            'weekend',
            'holiday',
            'morning_rush',
            'evening rush',
            'prcp',
            'temp',
            'distance',
            'airport',
            'congestion',
            'class_0', 
            'class_1',
            'class_2',
            'PULocationID_duration_encoded',
            'DOLocationID_duration_encoded']

openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(api_key=openai_api_key)

def speech_to_text(audio_file_path):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file_path) as source:
            audio_data = r.record(source)
        text = r.recognize_google(audio_data)
        return Document(page_content=text)
    except sr.UnknownValueError:
        return "Speech Recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Speech Recognition service; {e}"
    except FileNotFoundError:
        return None  # Handle case where no audio was recorded
    except Exception as e:
        return f"An error occurred during speech recognition: {e}"


def get_bot_response(user_query):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", 
            messages=[
                {"role": "system", "content": """You are a helpful assistant that can understand user inputs. Your primary goal is to identify pickup and dropoff location names a user provides."""},
                {"role": "user", "content": user_query},
            ],
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"An error occurred with OpenAI: {e}"


def find_location_match(user_text, location_options, threshold=60):
    best_match, score = process.extractOne(user_text, location_options)
    if score >= threshold:
        return best_match
    else:
        return None


def calc_distance(coord1, coord2):
    return geodesic(coord1, coord2).miles

def get_weather_data(latitude, longitude):
    temp = 0
    prcp = 0.0
    api_url = f"https://api.weather.gov/points/{latitude},{longitude}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            point_data = response.json()
            forecast_url = point_data['properties']['forecastHourly']
            observation_stations_url = point_data['properties']['observationStations']

            forecast_response = requests.get(forecast_url, timeout=10)
            if forecast_response.status_code == 200:
                forecast_data = forecast_response.json()
                periods = forecast_data['properties']['periods']
                if periods:
                    temp = periods[0]['temperature']

            observation_response = requests.get(observation_stations_url, timeout=10)
            if observation_response.status_code == 200:
                stations_data = observation_response.json()
                if stations_data['features']:
                    first_station_url = stations_data['features'][0]['id']
                    station_observation_response = requests.get(first_station_url + '/observations/latest', timeout=10)
                    if station_observation_response.status_code == 200:
                        observation_data = station_observation_response.json()
                        prcp_mm = observation_data['properties'].get('precipLastHour', {}).get('value')
                        if prcp_mm is not None:
                            prcp = [prcp_mm]
                        else:
                            prcp = [0.0]
    except requests.exceptions.RequestException as e:
        st.warning(f"Error fetching weather data: {e}")
    
    return temp, prcp


st.title("NY Ride Service Price and Duration Prediction")


st.markdown(
    """
    <style>
    .stHorizontalBlock.st-emotion-cache-ocqkz7.eu6p4el0 {
        
        height: 100px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Create Mappings
location_options_df = coords_df[['Zone', 'LocationID']].copy()
location_options = [f"{row['Zone']} ({row['LocationID']})" for index, row in coords_df.iterrows()]
sorted_location_options = sorted(location_options)
sorted_location_options_df = location_options_df.sort_values(by='Zone').reset_index(drop=True)
zone_id_map = {option: int(option.split('(')[-1][:-1]) for option in sorted_location_options}
location_to_zone = {v: k for k, v in zone_id_map.items()}

# Select Boxes
pickup_options_list = [f"{row['Zone']} ({row['LocationID']})" for index, row in coords_df.iterrows()]
sorted_pickup_options = sorted(pickup_options_list)
dropoff_options_list = [f"{row['Zone']} ({row['LocationID']})" for index, row in coords_df.iterrows()]
sorted_dropoff_options = sorted(dropoff_options_list)

pickup_options_list2 = [f"{row['Zone']}" for index, row in coords_df.iterrows()]
sorted_pickup_options2 = sorted(pickup_options_list2)
dropoff_options_list2 = [f"{row['Zone']}" for index, row in coords_df.iterrows()]
sorted_dropoff_options2 = sorted(dropoff_options_list2)

# Initialize session state
if 'pickup_option' not in st.session_state:
    st.session_state['pickup_option'] = sorted_pickup_options[0] if sorted_pickup_options else None
if 'dropoff_option' not in st.session_state:
    st.session_state['dropoff_option'] = sorted_dropoff_options[0] if sorted_dropoff_options else None
if 'run_prediction' not in st.session_state:
    st.session_state['run_prediction'] = False

# Sidebar for User Input
st.sidebar.header("Input Pickup and Dropoff Locations")


# Sidebar Select Boxes
default_pickup = st.session_state.get('pickup_option', sorted_pickup_options[0] if sorted_pickup_options else None)
pickup_index = sorted_pickup_options.index(default_pickup) if default_pickup in sorted_pickup_options else 0
st.session_state['pickup_option'] = st.sidebar.selectbox("Pickup Location", sorted_pickup_options, index=pickup_index, key='pickup_selectbox')

default_dropoff = st.session_state.get('dropoff_option', sorted_dropoff_options[0] if sorted_dropoff_options else None)
dropoff_index = sorted_dropoff_options.index(default_dropoff) if default_dropoff in sorted_dropoff_options else 0
st.session_state['dropoff_option'] = st.sidebar.selectbox("Dropoff Location", sorted_dropoff_options, index=dropoff_index, key='dropoff_selectbox')

current_time = st.sidebar.time_input("Current Time", datetime.now().time(), key="current_time_input")
current_date = st.sidebar.date_input("Current Date", datetime.now().date(), key="current_date_input")

#if st.sidebar.button("Run Prediction"):
    #st.session_state['run_prediction'] = True
#else:
    #st.session_state['run_prediction'] = False









def find_nearest_location(click_lat, click_lng, locations_df):
    nearest_location = None
    min_distance = float('inf')
    click_coords = (click_lat, click_lng)
    for index, row in locations_df.iterrows():
        location_coords = (row['latitude'], row['longitude'])
        distance = geodesic(click_coords, location_coords).miles
        if distance < min_distance:
            min_distance = distance
            nearest_location = f"{row['Zone']} ({row['LocationID']})"
    return nearest_location

st.markdown(
    """
    <style>
    .st-emotion-cache-8atqhb.e1mlolmg0 {
        height: 500px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,)

# Create the Folium map 
nyc_coords = [40.7128, -74.0060]
m = folium.Map(location=nyc_coords, zoom_start=11)

# Add interactive markers 
for index, row in coords_df.iterrows():
    tooltip = f"ID: {row['LocationID']}, Zone: {row['Zone']}"
    folium.Marker(
        [row['latitude'], row['longitude']],
        tooltip=tooltip,
        icon=folium.Icon(color='blue', icon='taxi', prefix='fa')
    ).add_to(m)


# lowers the position of the run button
st.markdown(
    """
    <style>
    button.st-emotion-cache-b0y9n5.em9zgd02 {
        margin-top: 27px 
    }
    button.st-emotion-cache-ocsh0s.em9zgd02 {
        margin-top: 27px
    </style>
    """,
    unsafe_allow_html=True,)
# Display the map 
st.subheader("Click on a marker to set Pickup or Dropoff (Optional)")
map_data = st_folium(m, width=700, height=500, returned_objects=["last_object_clicked"])
# Voice input widget
spoken_audio_bytes = st.audio_input(
    "Speak your request:",
    key="speech_input"
)

stt_input = None
if spoken_audio_bytes:
    stt_input = speech_to_text(spoken_audio_bytes)
    stt_input = stt_input.page_content
    
else:
    pass
user_input = st.text_input("Or type your request:", stt_input if stt_input else "", key="text_input_chatbot").strip('page_content')
chat_placeholder = st.empty()

# Process map click events
if map_data and map_data.get("last_object_clicked"):
    clicked = map_data["last_object_clicked"]
    click_lat = clicked.get('lat')
    click_lng = clicked.get('lng')
    if click_lat is not None and click_lng is not None:
        nearest = find_nearest_location(click_lat, click_lng, coords_df)
        if nearest:
            
            if st.session_state['pickup_option'] == sorted_pickup_options[0]: 
                st.session_state['pickup_option'] = nearest
                st.sidebar.info(f"Pickup set from map: {st.session_state['pickup_option']}")
            elif st.session_state['dropoff_option'] == sorted_dropoff_options[0]: 
                st.session_state['dropoff_option'] = nearest
                st.sidebar.info(f"Dropoff set from map: {st.session_state['dropoff_option']}")  

       
if user_input:
    pickup_match = None
    dropoff_match = None

    try:
        # Get the bot's response to interpret the user input
        bot_response = get_bot_response(user_input)
        #st.info(f"Bot: {bot_response}")

        # Try to extract pickup and dropoff from the bot's response
        pickup_zone_phrase = None
        dropoff_zone_phrase = None

        if "pickup" in bot_response.lower() and "dropoff" in bot_response.lower():
            try:
                parts = bot_response.lower().split("pickup")
                if len(parts) > 1:
                    pickup_part = parts[1]
                    dropoff_parts = pickup_part.split("dropoff")
                    if len(dropoff_parts) > 1:
                        pickup_zone_phrase = dropoff_parts[0].strip().replace(":", "").strip()
                        dropoff_zone_phrase = dropoff_parts[1].strip().replace(":", "").strip()
            except Exception as e:
                st.warning(f"Error parsing bot response: {e}")

        if pickup_zone_phrase and dropoff_zone_phrase:
            pickup_match = find_location_match(pickup_zone_phrase, sorted_pickup_options2)
            dropoff_match = find_location_match(dropoff_zone_phrase, sorted_dropoff_options2)

            if pickup_match and dropoff_match:
                st.info(f"MaxiMile identified pickup: {pickup_match}, dropoff: {dropoff_match}")

                # Get Location IDs and update session state (same as before)
                pickup_location_df = sorted_location_options_df[sorted_location_options_df['Zone'] == pickup_match]
                dropoff_location_df = sorted_location_options_df[sorted_location_options_df['Zone'] == dropoff_match]

                if not pickup_location_df.empty and not dropoff_location_df.empty:
                    pickup_location_id = pickup_location_df['LocationID'].iloc[0]
                    dropoff_location_id = dropoff_location_df['LocationID'].iloc[0]

                    st.session_state['pickup_option'] = f"{pickup_match} ({pickup_location_id})"
                    st.session_state['dropoff_option'] = f"{dropoff_match} ({dropoff_location_id})"

                    st.success("Pickup and Dropoff locations updated. Click 'Run Prediction'!")
                    
                else:
                    st.warning("Could not find the identified zones in the location data.")
            #else:
                #st.warning("Chatbot identified pickup or dropoff, but could not find a close match in the location list.")
        #else:
            #st.info("Please specify pickup and dropoff locations in your request.")

    except Exception as e:
        st.error(f"Error processing input: {e}")

# location and run buttons
cols = st.columns([3, 3, 2]) 

with cols[0]:
    default_pickup = st.session_state.get('pickup_option', sorted_pickup_options[0] if sorted_pickup_options else None)
    pickup_index = sorted_pickup_options.index(default_pickup) if default_pickup in sorted_pickup_options else 0
    st.session_state['pickup_option'] = st.selectbox("Pickup Location", sorted_pickup_options, index=pickup_index, key='pickup_selectbox_main')
    pickup_info_placeholder = st.empty() 

with cols[1]:
    default_dropoff = st.session_state.get('dropoff_option', sorted_dropoff_options[0] if sorted_dropoff_options else None)
    dropoff_index = sorted_dropoff_options.index(default_dropoff) if default_dropoff in sorted_dropoff_options else 0
    st.session_state['dropoff_option'] = st.selectbox("Dropoff Location", sorted_dropoff_options, index=dropoff_index, key='dropoff_selectbox_main')
    dropoff_info_placeholder = st.empty()

with cols[2]:
    if st.button("Run Prediction", key='run_button_below_map'):
        st.session_state['run_prediction'] = True
    else:
        st.session_state['run_prediction'] = False

# prediction calculations
if st.session_state['run_prediction']:
    
    pickup_option = st.session_state['pickup_option']
    dropoff_option = st.session_state['dropoff_option']
    try:
        pickup_location_id = zone_id_map[pickup_option]
        dropoff_location_id = zone_id_map[dropoff_option]
    except KeyError:
        st.error("Please select valid Pickup and Dropoff locations.")
        st.stop()


    now = datetime.combine(current_date, current_time)
    second_of_day = (now.hour * 3600) + (now.minute * 60) + now.second
    day_of_year = now.timetuple().tm_yday
    weekend = 1 if now.weekday() >= 5 else 0
    us_holidays = holidays.US()
    holiday = 1 if now.date() in us_holidays else 0
    morning_rush = 1 if now.weekday() < 5 and 7 <= now.hour <= 9 else 0
    evening_rush = 1 if now.weekday() < 5 and 16 <= now.hour <= 18 else 0
    congestion = 2.50 if morning_rush or evening_rush else 0.0

    pickup_row = coords_df[coords_df['LocationID'] == pickup_location_id].iloc[0]
    dropoff_row = coords_df[coords_df['LocationID'] == dropoff_location_id].iloc[0]
    pickup_coords = (pickup_row['latitude'], pickup_row['longitude'])
    dropoff_coords = (dropoff_row['latitude'], dropoff_row['longitude'])
    pickup_zone = pickup_row['Zone']
    dropoff_zone = dropoff_row['Zone']
    distance = round(calc_distance(pickup_coords, dropoff_coords), 2) if pickup_coords and dropoff_coords else 0.0

    temp, prcp = get_weather_data(pickup_row['latitude'], pickup_row['longitude'])

    


    test_input_dict = {
        'second_of_day': [second_of_day],
        'day_of_year': [day_of_year],
        'weekend': [weekend],
        'holiday': [holiday],
        'morning_rush': [morning_rush],
        'evening rush': [evening_rush],
        'PULocationID': [pickup_location_id],
        'DOLocationID': [dropoff_location_id],
        'prcp': [prcp[0]],
        'temp': [temp],
        'distance': [distance],
        'airport': [0.0],
        'congestion': [congestion],
        'class': [0]
    }
    test_input = pd.DataFrame(test_input_dict)
    test_input_duration = pd.DataFrame(test_input_dict.copy())


    test_input['PULocationID_price_encoded'] = encoder_pu_price.transform(test_input['PULocationID'])
    test_input['DOLocationID_price_encoded'] = encoder_do_price.transform(test_input['DOLocationID'])
    test_input_duration['PULocationID_duration_encoded'] = encoder_pu_duration.transform(test_input_duration['PULocationID'])
    test_input_duration['DOLocationID_duration_encoded'] = encoder_do_duration.transform(test_input_duration['DOLocationID'])

    test_input = pd.get_dummies(test_input, columns=['class'], prefix='class', drop_first=False)

    for i in range(3):
        col_name = f'class_{i}'
        if col_name not in test_input.columns:
            test_input[col_name] = 0
    
    test_input_duration = pd.get_dummies(test_input_duration, columns=['class'], prefix='class', drop_first=False)

    for i in range(3):
        col_name = f'class_{i}'
        if col_name not in test_input_duration.columns:
            test_input_duration[col_name] = 0

        
    airport_do_ids = [1, 132, 138]
    fhv_classes = int(test_input['class_1'].iloc[0]) == 1 or int(test_input['class_2'].iloc[0]) == 1
    airport = 0.0
    if dropoff_location_id in airport_do_ids and fhv_classes:
        airport = 2.50
    elif pickup_location_id in airport_do_ids and fhv_classes:
        airport = 2.50

    elif pickup_location_id in [132, 138] and int(test_input['class_0'].iloc[0]) == 0:
        airport = 1.75
    test_input['airport'] = airport


    # Predict Yellow Cab Duration
    predicted_duration_yellow = xgbr2.predict(test_input_duration[duration_features])[0]
    test_input['durationsec'] = predicted_duration_yellow
    test_df = test_input[features].copy()


    st.subheader("Prediction Results")
    col_yellow, col_uber, col_lyft = st.columns([1, 1, 1])

    # Yellow Cab Prediction
    predicted_price_yellow = xgbr.predict(test_df)[0]
        
    test_input_uber = test_input_duration.copy()
    test_input_uber[['class_0', 'class_1', 'class_2']] = [0, 1, 0]
    predicted_duration_uber = xgbr2.predict(test_input_uber[duration_features])[0]
    test_input_uber_price = test_df.copy()
    test_input_uber_price[['class_0', 'class_1', 'class_2']] = [0, 1, 0]
    predicted_price_uber = xgbr.predict(test_input_uber_price)[0]
    

    test_input_lyft = test_input_duration.copy()
    test_input_lyft[['class_0', 'class_1', 'class_2']] = [0, 0, 1]
    predicted_duration_lyft = xgbr2.predict(test_input_lyft[duration_features])[0]
    test_input_lyft_price = test_df.copy()
    test_input_lyft_price[['class_0', 'class_1', 'class_2']] = [0, 0, 1]
    predicted_price_lyft = xgbr.predict(test_input_lyft_price)[0]

    # Collect the predicted prices and durations
    prices = {
        "Yellow Cab": predicted_price_yellow,
        "Uber": predicted_price_uber,
        "Lyft": predicted_price_lyft,
    }
    durations = {
        "Yellow Cab": predicted_duration_yellow / 60,
        "Uber": predicted_duration_uber / 60,
        "Lyft": predicted_duration_lyft / 60,
    }
    
    # Determine the cheapest and shortest
    cheapest_service = min(prices, key=prices.get)
    shortest_service = min(durations, key=durations.get)

    expensive_service = max(prices, key=prices.get)
    longest_service = max(durations, key=durations.get)

    for service in prices.keys():
        if service != cheapest_service and service != expensive_service:
            mid_price_service = service
    
    for service in durations.keys():
        if service != shortest_service and service != longest_service:
            mid_dur_service = service
    

    # Determine the column indices for the cheapest and shortest services
    cheapest_col_index = -1
    shortest_col_index = -1

    if cheapest_service == "Yellow Cab":
        cheapest_col_index = 0
    elif cheapest_service == "Uber":
        cheapest_col_index = 1
    elif cheapest_service == "Lyft":
        cheapest_col_index = 2

    if shortest_service == "Yellow Cab":
        shortest_col_index = 0
    elif shortest_service == "Uber":
        shortest_col_index = 1
    elif shortest_service == "Lyft":
        shortest_col_index = 2

    mid_price_col_index = -1
    mid_dur_col_index = -1

    if mid_price_service == "Yellow Cab":
        mid_price_col_index = 0
    elif mid_price_service == "Uber":
        mid_price_col_index = 1
    elif mid_price_service == "Lyft":
        mid_price_col_index = 2

    if mid_dur_service == "Yellow Cab":
        mid_dur_col_index = 0
    elif mid_dur_service == "Uber":
        mid_dur_col_index = 1
    elif mid_dur_service == "Lyft":
        mid_dur_col_index = 2


    expensive_col_index = -1
    longest_col_index = -1

    if expensive_service == "Yellow Cab":
        expensive_col_index = 0
    elif expensive_service == "Uber":
        expensive_col_index = 1
    elif expensive_service == "Lyft":
        expensive_col_index = 2

    if longest_service == "Yellow Cab":
        longest_col_index = 0
    elif longest_service == "Uber":
        longest_col_index = 1
    elif longest_service == "Lyft":
        longest_col_index = 2
        

    # Prices
    st.write("**Price Comparison**")
    price_cols = st.columns(3)
    price_styles = ["", "", ""] 
    if cheapest_col_index != -1:
        price_styles[cheapest_col_index] = "border: 2px solid lightgreen; !important; padding: 15px !important; border-radius: 5px !important;"
    
    if expensive_col_index != -1:
        price_styles[expensive_col_index] = "border: 2px solid red; !important; padding: 15px !important; border-radius: 5px !important;"

    if mid_price_col_index != -1:
        price_styles[mid_price_col_index] = "border: 2px solid yellow; !important; padding: 15px !important; border-radius: 5px !important;"

    with price_cols[0]:
        st.markdown(f"<div style='{price_styles[0]}'>", unsafe_allow_html=True)
        st.metric("Yellow Cab", f"${predicted_price_yellow:.2f}")
        st.markdown(f"<div style='{price_styles[0]}'>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)
    with price_cols[1]:
        st.markdown(f"<div style='{price_styles[1]}'>", unsafe_allow_html=True)
        st.metric("Uber", f"${predicted_price_uber:.2f}")
        st.markdown(f"<div style='{price_styles[1]}'>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)
    with price_cols[2]:
        st.markdown(f"<div style='{price_styles[2]}'>", unsafe_allow_html=True)
        st.metric("Lyft", f"${predicted_price_lyft:.2f}")
        st.markdown(f"<div style='{price_styles[2]}'>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True) 

    # Durations
    st.write("**Duration Comparison**")
    duration_cols = st.columns(3)
    duration_styles = ["", "", ""]  
    if shortest_col_index != -1:
        duration_styles[shortest_col_index] = "border: 2px solid lightgreen; !important; padding: 15px !important; border-radius: 5px !important;"
    
    if longest_col_index != -1:
        duration_styles[longest_col_index] = "border: 2px solid red; !important; padding: 15px !important; border-radius: 5px !important;"

    if mid_dur_col_index != -1:
        duration_styles[mid_dur_col_index] = "border: 2px solid yellow; !important; padding: 15px !important; border-radius: 5px !important;"

    with duration_cols[0]:
        st.markdown(f"<div style='{duration_styles[0]}'>", unsafe_allow_html=True)
        st.metric("Yellow Cab", f"{predicted_duration_yellow / 60:.2f} mins")
        st.markdown(f"<div style='{duration_styles[0]}'>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)
    with duration_cols[1]:
        st.markdown(f"<div style='{duration_styles[1]}'>", unsafe_allow_html=True)
        st.metric("Uber", f"{predicted_duration_uber / 60:.2f} mins")
        st.markdown(f"<div style='{duration_styles[1]}'>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)
    with duration_cols[2]:
        st.markdown(f"<div style='{duration_styles[2]}'>", unsafe_allow_html=True)
        st.metric("Lyft", f"{predicted_duration_lyft / 60:.2f} mins")
        st.markdown(f"<div style='{duration_styles[2]}'>", unsafe_allow_html=True)
        st.markdown(f"</div>", unsafe_allow_html=True)

    st.subheader("Trip Information")
    st.write(f"**Cheapest Service:** {cheapest_service}")
    st.write(f"**Shortest Service:** {shortest_service}")
    st.write(f"**Pickup Zone:** {pickup_zone} (ID: {pickup_location_id})")
    st.write(f"**Dropoff Zone:** {dropoff_zone} (ID: {dropoff_location_id})")
    st.write(f"**Distance:** {distance:.2f} miles")
    st.write(f"**Current Temperature:** {temp}°F")
    st.write(f"**Precipitation (last hour):** {prcp[0]} mm")
    st.write(f"**Time of Request:** {now.strftime('%I:%M:%S %p %m/%d/%Y')}")

    st.session_state['pickup_option'] = sorted_pickup_options[0] if sorted_pickup_options else None
    st.session_state['dropoff_option'] = sorted_dropoff_options[0] if sorted_dropoff_options else None
    st.session_state['run_prediction'] = False
