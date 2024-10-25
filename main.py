import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
import joblib
from flask import Flask


with open('trans_encoder1.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Function to load the model
@st.cache_resource
def load_model():
    with st.spinner("Loading model... Please wait."):
        with open('trans_predictor1.pkl', 'rb') as file:
            return pickle.load(file)


# Load the model and data
rf = load_model()
# data, encoder = load_and_encode_data()

st.title("Property Price Estimator")

# Select input options
trans_group = st.selectbox('Transaction Type', ['Sales', 'Mortgages'])

property_type_en = st.selectbox('Property Type', ['Unit', 'Villa', 'Land', 'Building'])

reg_type_en = st.selectbox('Registration type', ['Off-Plan Properties', 'Existing Properties'])

property_usage_en = st.selectbox('Property Usage', ['Commercial', 'Residential', 'Hospitality ', 'Industrial',
       'Agricultural', 'Multi-Use', 'Storage', 'Residential / Commercial', 'Other'])
area_name_en = st.selectbox('Area Name', ['Al Muteena', 'Burj Khalifa', 'Al Warsan First',
       'Al Thanyah Fifth', "Me'Aisem First", 'Marsa Dubai',
       'Wadi Al Safa 5', 'Jabal Ali First', 'Al Khairan First',
       'Nadd Hessa', 'Al Jadaf', 'Madinat Al Mataar',
       'Dubai Investment Park First', 'Wadi Al Safa 6',
       'Al Hebiah Fourth', 'Al Garhoud', 'Al Barshaa South Third',
       'Al Yufrah 3', 'Al Hebiah Third', 'Business Bay',
       'Al Hebiah Fifth', 'Hadaeq Sheikh Mohammed Bin Rashid',
       'Al Yelayiss 2', 'Al Barsha South Fourth', 'Al Waheda',
       'Al Barsha Third', 'Al Suq Al Kabeer', 'Al Barsha South Fifth',
       'Al Hebiah Sixth', 'Wadi Al Safa 3', 'Al Thanayah Fourth',
       'Wadi Al Safa 4', 'Mirdif', 'Al Hebiah First', 'Al Merkadh',
       'Wadi Al Safa 2', 'Abu Hail', 'Al Hebiah Second', 'Mankhool',
       'Palm Jumeirah', 'Al Thanyah Third', 'Al Thanyah First',
       'Al Goze Fourth', 'Zaabeel Second', 'Jumeirah Third',
       'Al Saffa Second', 'Nad Al Hamar', 'Dubai Investment Park Second',
       'Wadi Al Safa 7', 'Saih Shuaib 2', 'Wadi Alshabak',
       'Madinat Dubai Almelaheyah', 'Al Safouh First', 'Al Mizhar First',
       'Al Aweer First', 'Um Suqaim Third', 'Oud Al Muteena First',
       'Al Yelayiss 1', 'Al Yufrah 1', 'Madinat Hind 4', 'World Islands',
       'Nad Al Shiba First', 'Al Warqa First', 'Al Wasl',
       'Al Barshaa South Second', 'Al Khawaneej Second', 'Mena Jabal Ali',
       'Al Ruwayyah', 'Nad Al Shiba Fourth', 'Al Warqa Third',
       'Al Yufrah 2', 'Al Twar Third', 'Saih Shuaib 1', 'Al Kifaf',
       'Hor Al Anz', 'Al Satwa', 'Al Mizhar Second', 'Al Kheeran',
       'Al Nahda Second', 'Oud Metha', 'Wadi Al Amardi', 'Nad Al Shiba',
       'Al Barshaa South First', 'Al Bada', 'Warsan Fourth',
       'Ras Al Khor Industrial First', 'Naif', 'Al Rashidiya',
       'Zaabeel First', 'Al Barsha Second', 'Al Murqabat',
       'Al Twar Fourth', 'Jabal Ali Industrial Second',
       'Muhaisanah First', 'Palm Deira', 'Al Saffa First', 'Al Dhagaya',
       'Saih Shuaib 4', 'Trade Center Second', 'Nad Al Shiba Third',
       'Al Warqa Second', 'Port Saeed', 'Al Jafliya', 'Al Mizhar Third',
       'Al Goze Industrial Second', "Me'Aisem Second", 'Al Warqa Fourth',
       'Al Mararr', 'Jumeirah First', 'Jabal Ali', 'Al Warsan Second',
       'Al Goze First', 'Al Raffa', 'Al Nahda First', 'Al Aweer Second',
       'Um Suqaim Second', 'Al Twar First', 'Island 2',
       'Jabal Ali Industrial First', 'Al Khawaneej First', 'Al Hamriya',
       'Eyal Nasser', 'Um Al Sheif', 'Sikkat Al Khail North',
       'Um Suqaim First', 'Muhaisanah Third', 'Zareeba Duviya',
       'Al Safouh Second', 'Um Suqaim', 'Al Barsha First', 'Al Manara',
       'Ghadeer Al tair', 'Al Khabeesi', 'Al Mamzer', 'Um Hurair Second',
       'Al Mizhar', 'Al Qusais First', 'Al Ras', 'Al Baraha',
       'Saih Shuaib 3', 'Al Twar Second', 'Ras Al Khor Industrial Second',
       'Um Hurair First', 'Al Qusais Industrial Fifth', 'Nad Rashid',
       'Al Qusais Industrial Fourth', 'Palm Jabal Ali', 'Hor Al Anz East',
       'Tawaa Al Sayegh', 'Al Khawaneej', 'Al Karama', 'Al Buteen',
       'Al Barsha', 'Al Lusaily', 'Jumeirah Second', 'Bukadra',
       'Um Ramool', 'Al Qusais Second', 'Al Hudaiba', 'Hessyan First',
       'Al Goze Third', 'Trade Center First', 'Al Ttay', 'Madinat Hind 3',
       'Jabal Ali Third', 'Al Safaa', 'Muhaisanah Second',
       'Nad Al Shiba Second', 'Al Rega', 'Al Qusais Industrial First',
       'Rega Al Buteen', 'Al Qusais', 'Al-Bastakiyah', 'Nad Shamma',
       'Tawi Al Muraqqab', 'Mushrif', 'Ras Al Khor Industrial Third',
       'Saih Aldahal', 'Al-Muhaisnah North', 'Jumeirah',
       'Al-Murar Qadeem', 'Al-Zarouniyyah', 'Muragab', 'Al Qoaz',
       'Lehbab', 'Al Goze Industrial Third', 'Al Sabkha', 'Margham',
       'Muhaisanah Fourth', 'Muhaisna', 'Al Warsan Third',
       'Al Yelayiss 5', 'Al Qusais Industrial Third', 'Al-Safiyyah',
       'Al Rowaiyah First', 'Al Rowaiyah Third', 'Madinat Hind 1',
       'Al Musalla (Dubai)', 'Dubai International Airport',
       'Al Qusais Industrial Second', 'Ras Al Khor', 'Al-Murar Jadeed',
       'Grayteesah', 'Lehbab First', 'Al Goze Industrial Fourth',
       'Al Goze Industrial First', 'Al Eyas', 'Al Asbaq',
       'Jabal Ali Industrial Third', 'Al-Tawar', 'Al Rowaiyah Second',
       'Al-Aweer', 'Lehbab Second', 'Al Khairan  Second', 'Naif North',
       'Al-Riqqa East', 'Hessyan Second', 'Al-Shumaal',
       'Al-Souq Al Kabeer (Deira)', 'Al Fahidi', 'Al Baharna',
       'Bur Dubai', 'Al Yelayiss 3', 'Hatta', 'Mugatrah', 'Al-Raulah',
       'Madinat Hind 2', 'Naif South', 'Al Maha',
       'Al-Dzahiyyah Al-Jadeedah', 'Al-Nakhal', 'Burj Nahar',
       'Al-Riqqa West', 'Al Layan1', 'Al Yelayiss 4', 'Madinat Latifa',
       'Al Yufrah 4', 'Cornich Deira', 'Al-Mustashfa West', 'Al-Nahdah',
       'Sikkat Al Khail South', 'Um Esalay', 'Umm Addamin', 'Nazwah',
       'Muashrah Al Bahraana', 'Al-Qiyadah', 'Shandagha West',
       'Al Marmoom', 'Le Hemaira', 'Al-Baloosh', 'Al-Cornich',
       'Shandagha East'])

nearest_metro_en = st.selectbox('Nearest Metro', ['Salah Al Din Metro Station',
       'Buj Khalifa Dubai Mall Metro Station', 'Rashidiya Metro Station',
       'Damac Properties', 'Jumeirah Beach Resdency', 'Marina Towers',
      'Mina Seyahi', 'Ibn Battuta Metro Station',
       'Creek Metro Station', 'Al Jadaf Metro Station',
       'DANUBE Metro Station', 'Nakheel Metro Station',
       'Jumeirah Lakes Towers', 'Airport Terminal 1 Metro Station',
       'Sharaf Dg Metro Station', 'Business Bay Metro Station',
       'First Abu Dhabi Bank Metro Station', 'Dubai Internet City',
       'Harbour Tower', 'Abu Hail Metro Station',
       'Al Fahidi Metro Station', 'Jumeirah Beach Residency',
       'Noor Bank Metro Station', 'Marina Mall Metro Station',
       'Abu Baker Al Siddique Metro Station', 'ADCB Metro Station',
       'Palm Jumeirah', 'Financial Centre',
       'Healthcare City Metro Station', 'Dubai Marina',
       'Etisalat Metro Station', 'ENERGY Metro Station',
       'Al Jafiliya Metro Station', 'Al Qusais Metro Station',
       'Trade Centre Metro Station', 'Emirates Metro Station',
       'Terminal 3 ', 'Al Nahda Metro Station', 'Knowledge Village',
       'Oud Metha Metro Station', 'Al Sufouh',
       'Emirates Towers Metro Station', 'Baniyas Square Metro Station',
       'Al Rigga Metro Station', 'UAE Exchange Metro Station',
       'STADIUM Metro Station', 'Media City', 'Palm Deira Metro Stations',
       'Deira City Centre', 'Al Ras Metro Station', 'Airport Free Zone',
       'Al Ghubaiba Metro Station', 'Al Qiyadah Metro Station',
       'Burjuman Metro Station', 'GGICO Metro Station',
       'Union Metro Station', 'Unknown'])

room_value = st.selectbox('Rooms', [ 'Studio', '1 B/R' , '2 B/R', '3 B/R', 'Office', 'Others'])

procedure_area = st.number_input('Property Area', min_value=0.0, step=1.0)



has_parking = st.selectbox('Has Parking?', ['Yes', 'No',])

room_value_mapping = {
    'Studio': 1,
    '1 B/R': 2,
    '2 B/R': 3,
    '3 B/R': 5,
    'Office': 4,
    'Others': 6
}


# Prediction
if st.button('Predict Price'):
    try:
        has_parking = 1 if has_parking == 'Yes' else 0

        room_value = room_value_mapping.get(room_value, 1)  # Default to 0 if not found

       

        # Prepare the input query
        input_data = pd.DataFrame({
             'property_usage_en': [property_usage_en],
             'property_type_en': [property_type_en],
              'reg_type_en': [reg_type_en],
               'area_name_en': [area_name_en],
                'nearest_metro_en': [nearest_metro_en],
          
            'room_value': [room_value],
            'has_parking': [has_parking],
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

        st.markdown(f"<h1 style='font-size:20px;'>Predicted price for this property is AED {int(prediction):,} </h1>",
    unsafe_allow_html=True)


    except Exception as e:
        st.error(f"An error occurred: {e}")
       