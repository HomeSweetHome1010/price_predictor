import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
import joblib
from flask import Flask


with open('r_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Function to load the model
@st.cache_resource
def load_model():
    with st.spinner("Loading model... Please wait."):
        with open('r_predictor.pkl', 'rb') as file:
            return pickle.load(file)


# Load the model and data
rf = load_model()
# data, encoder = load_and_encode_data()

areas = [
  {
    "area_name_en": "Al Barsha First",
    "nearest_metro_en": ["Sharaf Dg Metro Station"],
  },
  {
    "area_name_en": "Al Barsha South Fifth",
    "nearest_metro_en": [
      "Damac Properties",
      "Unknown",
      "Jumeirah Lakes Towers",
      "Nakheel Metro Station",
      "Harbour Tower",
    ],
  },
  {
    "area_name_en": "Al Barsha South Fourth",
    "nearest_metro_en": [
      "Unknown",
      "Nakheel Metro Station",
      "Dubai Internet City",
    ],
  },
  {
    "area_name_en": "Al Barshaa South Second",
    "nearest_metro_en": ["First Abu Dhabi Bank Metro Station"],
  },
  {
    "area_name_en": "Al Barshaa South Third",
    "nearest_metro_en": [
      "Unknown",
      "First Abu Dhabi Bank Metro Station",
      "Sharaf Dg Metro Station",
    ],
  },
  {
    "area_name_en": "Al Goze Fourth",
    "nearest_metro_en": ["Noor Bank Metro Station"],
  },
  { "area_name_en": "Al Hebiah Fifth", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Al Hebiah First",
    "nearest_metro_en": ["Sharaf Dg Metro Station", "Dubai Internet City"],
  },
  {
    "area_name_en": "Al Hebiah Fourth",
    "nearest_metro_en": [
      "Unknown",
      "Nakheel Metro Station",
      "Damac Properties",
      "Dubai Internet City",
    ],
  },
  {
    "area_name_en": "Al Hebiah Second",
    "nearest_metro_en": [
      "Unknown",
      "Sharaf Dg Metro Station",
      "Dubai Internet City",
    ],
  },
  { "area_name_en": "Al Hebiah Sixth", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Al Hebiah Third", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Al Jadaf",
    "nearest_metro_en": [
      "Healthcare City Metro Station",
      "Unknown",
      "Creek Metro Station",
      "Al Jadaf Metro Station",
    ],
  },
  {
    "area_name_en": "Al Khairan First",
    "nearest_metro_en": ["Unknown", "Creek Metro Station"],
  },
  {
    "area_name_en": "Al Kheeran",
    "nearest_metro_en": [
      "Rashidiya Metro Station",
      "Emirates Metro Station",
      "Al Jadaf Metro Station",
    ],
  },
  { "area_name_en": "Al Kifaf", "nearest_metro_en": ["Al Jafiliya Metro Station"] },
  {
    "area_name_en": "Al Merkadh",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
  {
    "area_name_en": "Al Qusais Industrial Fifth",
    "nearest_metro_en": ["Al Qusais Metro Station"],
  },
  {
    "area_name_en": "Al Qusais Industrial Fourth",
    "nearest_metro_en": ["Airport Free Zone"],
  },
  {
    "area_name_en": "Al Safouh First",
    "nearest_metro_en": ["Sharaf Dg Metro Station", "Dubai Internet City"],
  },
  {
    "area_name_en": "Al Safouh Second",
    "nearest_metro_en": ["Dubai Internet City", "Palm Jumeirah", "Mina Seyahi"],
  },
  {
    "area_name_en": "Al Satwa",
    "nearest_metro_en": [
      "Emirates Towers Metro Station",
      "Trade Centre Metro Station",
    ],
  },
  {
    "area_name_en": "Al Thanayah Fourth",
    "nearest_metro_en": ["Unknown", "Nakheel Metro Station", "Damac Properties"],
  },
  {
    "area_name_en": "Al Thanyah Fifth",
    "nearest_metro_en": [
      "Damac Properties",
      "Unknown",
      "Jumeirah Lakes Towers",
      "Harbour Tower",
      "Marina Mall Metro Station",
    ],
  },
  {
    "area_name_en": "Al Thanyah First",
    "nearest_metro_en": ["Dubai Internet City"],
  },
  {
    "area_name_en": "Al Thanyah Third",
    "nearest_metro_en": [
      "Unknown",
      "Nakheel Metro Station",
      "Damac Properties",
      "Dubai Internet City",
    ],
  },
  {
    "area_name_en": "Al Warsan First",
    "nearest_metro_en": ["Rashidiya Metro Station"],
  },
  {
    "area_name_en": "Al Wasl",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
  { "area_name_en": "Al Yelayiss 1", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Al Yelayiss 2", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Al Yufrah 1", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Bukadra",
    "nearest_metro_en": ["Unknown", "Creek Metro Station"],
  },
  {
    "area_name_en": "Burj Khalifa",
    "nearest_metro_en": [
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
  {
    "area_name_en": "Business Bay",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
  {
    "area_name_en": "Dubai Investment Park First",
    "nearest_metro_en": [
      "Unknown",
      "Ibn Battuta Metro Station",
      "DANUBE Metro Station",
    ],
  },
  {
    "area_name_en": "Dubai Investment Park Second",
    "nearest_metro_en": ["Unknown"],
  },
  {
    "area_name_en": "Hadaeq Sheikh Mohammed Bin Rashid",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "First Abu Dhabi Bank Metro Station",
      "Noor Bank Metro Station",
    ],
  },
  {
    "area_name_en": "Hessyan First",
    "nearest_metro_en": ["Unknown", "UAE Exchange Metro Station"],
  },
  {
    "area_name_en": "Island 2",
    "nearest_metro_en": ["Unknown", "Business Bay Metro Station"],
  },
  { "area_name_en": "Jabal Ali", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Jabal Ali First",
    "nearest_metro_en": [
      "Ibn Battuta Metro Station",
      "Harbour Tower",
      "Unknown",
      "ENERGY Metro Station",
    ],
  },
  {
    "area_name_en": "Jabal Ali Industrial Second",
    "nearest_metro_en": ["UAE Exchange Metro Station"],
  },
  {
    "area_name_en": "Jumeirah First",
    "nearest_metro_en": [
      "Unknown",
      "Emirates Towers Metro Station",
      "Trade Centre Metro Station",
    ],
  },
  {
    "area_name_en": "Jumeirah Second",
    "nearest_metro_en": ["Unknown", "Business Bay Metro Station"],
  },
  { "area_name_en": "Madinat Al Mataar", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Madinat Dubai Almelaheyah",
    "nearest_metro_en": ["Al Ghubaiba Metro Station", "Unknown"],
  },
  { "area_name_en": "Madinat Hind 4", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Marsa Dubai",
    "nearest_metro_en": [
      "Unknown",
      "Dubai Marina",
      "Jumeirah Lakes Towers",
      "Mina Seyahi",
      "Marina Towers",
      "Jumeirah Beach Residency",
      "Marina Mall Metro Station",
      "Jumeirah Beach Resdency",
    ],
  },
  {
    "area_name_en": "Me'Aisem First",
    "nearest_metro_en": ["Unknown", "Harbour Tower", "Damac Properties"],
  },
  {
    "area_name_en": "Mirdif",
    "nearest_metro_en": ["Etisalat Metro Station", "Rashidiya Metro Station"],
  },
  {
    "area_name_en": "Muhaisanah First",
    "nearest_metro_en": ["Rashidiya Metro Station"],
  },
  {
    "area_name_en": "Nad Al Hamar",
    "nearest_metro_en": ["Rashidiya Metro Station"],
  },
  {
    "area_name_en": "Nad Al Shiba First",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
      "Creek Metro Station",
    ],
  },
  { "area_name_en": "Nadd Hessa", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Palm Deira",
    "nearest_metro_en": ["Unknown", "Palm Deira Metro Stations"],
  },
  {
    "area_name_en": "Palm Jumeirah",
    "nearest_metro_en": [
      "Palm Jumeirah",
      "Unknown",
      "Al Sufouh",
      "Mina Seyahi",
      "Knowledge Village",
      "Jumeirah Beach Resdency",
    ],
  },
  { "area_name_en": "Port Saeed", "nearest_metro_en": ["Deira City Centre"] },
  {
    "area_name_en": "Ras Al Khor Industrial First",
    "nearest_metro_en": ["Unknown"],
  },
  {
    "area_name_en": "Rega Al Buteen",
    "nearest_metro_en": ["Al Rigga Metro Station"],
  },
  { "area_name_en": "Saih Shuaib 1", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Saih Shuaib 2",
    "nearest_metro_en": ["Unknown", "UAE Exchange Metro Station"],
  },
  {
    "area_name_en": "Trade Center First",
    "nearest_metro_en": ["Emirates Towers Metro Station"],
  },
  {
    "area_name_en": "Trade Center Second",
    "nearest_metro_en": ["Trade Centre Metro Station"],
  },
  {
    "area_name_en": "Um Hurair Second",
    "nearest_metro_en": ["Healthcare City Metro Station"],
  },
  {
    "area_name_en": "Um Suqaim Third",
    "nearest_metro_en": ["Unknown", "First Abu Dhabi Bank Metro Station"],
  },
  { "area_name_en": "Wadi Al Safa 2", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Wadi Al Safa 3", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Wadi Al Safa 4", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Wadi Al Safa 5", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Wadi Al Safa 6",
    "nearest_metro_en": [
      "Unknown",
      "First Abu Dhabi Bank Metro Station",
      "Sharaf Dg Metro Station",
    ],
  },
  { "area_name_en": "Wadi Al Safa 7", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Warsan Fourth",
    "nearest_metro_en": ["Unknown", "Rashidiya Metro Station"],
  },
  {
    "area_name_en": "World Islands",
    "nearest_metro_en": ["Unknown", "Noor Bank Metro Station"],
  },
  {
    "area_name_en": "Zaabeel First",
    "nearest_metro_en": ["Al Jafiliya Metro Station", "Unknown"],
  },
  {
    "area_name_en": "Zaabeel Second",
    "nearest_metro_en": [
      "Financial Centre",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
]

st.markdown(
    """
    <style>
    /* Custom select box focus styling */
    .custom-select:focus {
        border: 2px solid green !important;
        outline: none;
    }
    </style>
    <script>
    // JavaScript to apply 'custom-select' style class to all select elements
    const selectElements = document.querySelectorAll('div[data-testid="stSelectbox"] select');
    selectElements.forEach((el) => {
        el.classList.add('custom-select');
    });
    </script>
    """,
    unsafe_allow_html=True
)

# st.title("Property Price Estimator")

st.markdown("""
<style>
    /* Custom styling for title */
    .big-title {
        font-size: 30px !important;
        color: #1E88E5 !important;
        padding-bottom: 24px !important;
        font-weight: bold !important;
        text-align: center !important;
    }
</style>
""", unsafe_allow_html=True)

# Using markdown with HTML to apply custom styling
# st.markdown("<h1 class='big-title'>Property Price Estimator</h1>", unsafe_allow_html=True)

st.markdown("""
<style>
    /* Base style for select boxes */
    div[data-baseweb="select"] > div:first-child {
        border-color: #ccc !important;
        transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    }
    
    /* Override all states to remove red */
    div[data-baseweb="select"] > div:first-child,
    div[data-baseweb="select"] > div:first-child:hover,
    div[data-baseweb="select"] > div:first-child:active,
    div[data-baseweb="select"] > div:first-child:focus,
    div[data-baseweb="select"] > div:first-child[data-focusvisible="true"] {
        border-color: rgb(173, 216, 230) !important;
        box-shadow: none !important;
    }
    
    /* Override hover state */
    div[data-baseweb="select"]:hover > div:first-child {
        border-color: rgb(173, 216, 230) !important;
        box-shadow: 0 0 0 1px rgb(173, 216, 230) !important;
    }
    
    /* Override focused/expanded state */
    div[data-baseweb="select"] > div:first-child[aria-expanded="true"],
    div[data-baseweb="select"] > div:first-child[data-focusvisible="true"] {
        border-color: rgb(173, 216, 230) !important;
        box-shadow: 0 0 0 2px rgb(173, 216, 230) !important;
    }
    
    /* Style the options dropdown */
    div[role="listbox"] {
        border-color: rgb(173, 216, 230) !important;
    }
    
    /* Style the selected option */
    div[data-baseweb="select"] span[aria-selected="true"] {
        background-color: rgba(173, 216, 230, 0.2) !important;
    }
    
    /* Style the hovered option */
    div[data-baseweb="select"] div[role="option"]:hover {
        background-color: rgba(173, 216, 230, 0.1) !important;
    }
    
    /* Number input styling */
    input[type="number"] {
        border-color: #ccc !important;
    }
    
    input[type="number"]:hover,
    input[type="number"]:focus {
        border-color: rgb(173, 216, 230) !important;
        box-shadow: 0 0 0 2px rgb(173, 216, 230) !important;
    }
    
    /* Remove red outline on invalid state */
    div[data-baseweb="select"] > div:first-child[aria-invalid="true"],
    input[type="number"][aria-invalid="true"] {
        border-color: rgb(173, 216, 230) !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Select input options
trans_group = st.selectbox('Transaction Type', ['Cash', 'Mortgages'])
reg_type_en = st.selectbox('Registration Type', ["Off-Plan", "Ready Property"])
property_type_en = st.selectbox('Property Type', ["Apartment", "Villa"])
property_usage_en = st.selectbox('Property Usage', [
    "Commercial", "Residential"
])
area_name_en = st.selectbox("Location", [area["area_name_en"] for area in areas])

# selected_location = next((area for area in areas if area["area_name_en"] == area_name_en), None)

# nearest_metro = None
# if selected_location:
#     nearest_metro = st.selectbox("Nearest Metro", selected_location["nearest_metro_en"])


# rooms_value = st.selectbox('Rooms', ["Studio", "1 B/R", "2 B/R", "3 B/R", "4 B/R", "5 B/R", "6 B/R", "7 B/R", "Office"])

rooms = {
    "Studio": 1,
    "1 B/R": 2,
    "2 B/R": 3,
    "3 B/R": 5,
    "4 B/R": 6,
    "5 B/R": 7,
    "6 B/R": 8,
    "7 B/R": 9,
    "Office": 4
}

# Create a select box for room selection
selected_room = st.selectbox('Room', list(rooms.keys()))

# Get the mapped value for the selected room
rooms_value = rooms[selected_room]

procedure_area = st.number_input('Property Area (sq.m)', min_value=0.0, step=1.0)


import streamlit as st

# Hide the "Made with Streamlit" footer and main menu
hide_streamlit_style = """
    <style>
      #MainMenu {display: none;}
    footer {display: none;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)





# Prediction
if st.button('Predict Price'):
    try:
        # has_parking = 1 if has_parking == 'Yes' else 0

        trans_group = 'Sales' if trans_group == 'Cash' else 'Mortgages'

        # Prepare the input query
        input_data = pd.DataFrame({
             'property_usage_en': [property_usage_en],
              'property_type_en': [property_type_en],
                'reg_type_en': [reg_type_en],
               "area_name_en": [area_name_en],
           'rooms_value': [rooms_value],
            'procedure_area': [procedure_area],
            'trans_group_en': [trans_group],
        })

        # input_data.reset_index(drop=True, inplace=True)

        # Apply the encoder to the input query
        query_encoded = encoder.transform(input_data)

        # Align features with model's expected input
        query_encoded = query_encoded.reindex(columns=rf.feature_names_in_, fill_value=0)
        
          # Debugging: Print shape of the transformed data

        # Make prediction
        prediction = np.exp(rf.predict(query_encoded))

        lower_bound = int((prediction * 0.9) // 100 * 100)
        upper_bound = int((prediction * 1.01) // 100 * 100)

        

        st.markdown(f"<h1 style='font-size:20px;'>Predicted price for this property is between AED {lower_bound:,} - AED {upper_bound:,} </h1>",
    unsafe_allow_html=True)

    
    except Exception as e:
        st.error(f"An error occurred: {e}")
       