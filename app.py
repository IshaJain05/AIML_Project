import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Title of the app
st.title("Crime Prediction App")

# Placeholder for your dataset
@st.cache_data
def load_data():
    # You would replace this with actual dataset loading logic
    data = pd.DataFrame({
        'total_cognizable_sll_crimes': np.random.randint(1000, 5000, size=100),
        'crime_against_women_total': np.random.randint(100, 1000, size=100)
    })
    return data

# Load and prepare data
data = load_data()
X = data[['total_cognizable_sll_crimes', 'crime_against_women_total']]
y = data['total_cognizable_sll_crimes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# Input field in Streamlit
crime_against_women_total = st.number_input('Crime Against Women (Total)', min_value=0, max_value=10000, value=500)

# Predict button
if st.button('Predict'):
    crime_input = scaler.transform([[crime_against_women_total, crime_against_women_total]])
    prediction = rf.predict(crime_input)[0]
    st.write(f'Predicted total cognizable SLL crimes: {prediction:.2f}')
