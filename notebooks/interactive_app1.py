import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import matplotlib.pyplot as plt

# Load and prepare data
@st.cache_data
def load_data():
    data = pd.read_csv('data/cab_data.csv')
    # Convert dates to datetime if needed
    # data['date_column'] = pd.to_datetime(data['date_column'])
    return data

data = load_data()

payment_type_mapping = {
    1: 'Credit Card',
    2: 'Cash',
    3: 'Dispute',
    4: 'No Charge'
}
data['payment_type_name'] = data['payment_type'].map(payment_type_mapping)
# Sidebar for filters
rate_code = st.sidebar.multiselect('Rate Code Name', options=data['RatecodeID'].unique())
payment_type = st.sidebar.multiselect('Payment Type Name', options=data['payment_type'].unique())
vendor_id = st.sidebar.multiselect('Vendor ID', options=data['VendorID'].unique())
min_distance, max_distance = data['trip_distance'].min(), data['trip_distance'].max()
distance_range = st.sidebar.slider('Trip Distance (miles)', min_value=float(min_distance), max_value=float(max_distance), value=(float(min_distance), float(max_distance)))

# Applying filters
filtered_data = data
if rate_code:
    filtered_data = filtered_data[filtered_data['RatecodeID'].isin(rate_code)]
if payment_type:
    filtered_data = filtered_data[filtered_data['payment_type'].isin(payment_type)]
if vendor_id:
    filtered_data = filtered_data[filtered_data['VendorID'].isin(vendor_id)]
if distance_range:
    filtered_data = filtered_data[(filtered_data['trip_distance'] >= distance_range[0]) & (filtered_data['trip_distance'] <= distance_range[1])]

# Display summaries
st.header("Summary")
st.write(f"Total Amount: ${filtered_data['total_amount'].sum():,.2f}")
st.write(f"Record Count: {len(filtered_data)}")
st.write(f"Avg Trip Distance: {filtered_data['trip_distance'].mean():.2f} miles")
st.write(f"Avg Fare Amount: ${filtered_data['fare_amount'].mean():,.2f}")
st.write(f"Avg Tip Amount: ${filtered_data['tip_amount'].mean():,.2f}")
st.write(f"Avg Passenger Count: {filtered_data['passenger_count'].mean():.2f}")

# Map visualization
st.header("Visualization of pickup and dropoff locations in NYC")
st.pydeck_chart(pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=pdk.ViewState(
        latitude=filtered_data['pickup_latitude'].median(),
        longitude=filtered_data['pickup_longitude'].median(),
        zoom=11,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HexagonLayer',
            data=filtered_data,
            get_position='[pickup_longitude, pickup_latitude]',
            radius=100,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
        ),
    ],
))



# Apply mapping to the payment_type column to make it more readable
filtered_data['payment_type_name'] = filtered_data['payment_type'].map(payment_type_mapping)

# Aggregate fare amounts by payment type
fare_amounts_by_type = filtered_data.groupby('payment_type_name')['fare_amount'].mean()  # or sum(), depending on the desired metric

# Plotting
st.header("Average Fare Amount by Payment Type")
st.bar_chart(fare_amounts_by_type)