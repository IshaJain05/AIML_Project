from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Load and preprocess the data
file_path = 'C:/Users/ishaj/Sem2_Project/DistrictwiseSLLCrimes.csv'

def preprocess_data():
    df = pd.read_csv(file_path)
    
    # Convert year column to numeric (extract only year)
    df['year'] = pd.to_datetime(df['year'], errors='coerce').dt.year
    df.dropna(subset=['year'], inplace=True)
    
    # Columns to be converted to numeric
    cols_to_convert = [
        'crime_against_women_total',
        'juvenile_justice_care_and_protection_of_children',
        'prohibition_of_child_marriage',
        'sc_and_st_related_crimes',
        'prevention_of_damage_to_public_property',
        'arms_total',
        'explosives_and_explosive_substances',
        'information_technology_or_intellectual_property_total',
        'prohibition_state',
        'excise',
        'ndps_total',
        'forest_act_1927_and_the_forest_conservation',
        'foreigner_and_passport_related_total',
        'food_drugs_and_essential_commodities_total',
        'gambling',
        'electricity',
        'other_sll_crimes'
    ]
    
    # Convert columns to numeric
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    return df, cols_to_convert

df, cols_to_convert = preprocess_data()

@app.route('/')
def index():
    states = df['state_name'].unique()
    state_districts = {
        state: df[df['state_name'] == state]['district_name'].unique().tolist()
        for state in states
    }
    return render_template('index.html', states=states, state_districts=state_districts)

@app.route('/predict', methods=['POST'])
def predict():
    state = request.form['state']
    district = request.form['district']
    future_years = int(request.form['years'])  # Number of future years for prediction
    
    # Filter the data for the selected state and district
    df_filtered = df[(df['state_name'] == state) & (df['district_name'] == district)]
    
    if df_filtered.empty:
        return "No data available for this state and district combination"
    
    # Linear Regression to predict future crime numbers
    X = df_filtered[['year'] + cols_to_convert]
    y = df_filtered['total_cognizable_sll_crimes']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict starting from the year 2025
    future_start_year = 2025
    future_year = future_start_year + future_years - 1  # Calculate the future year based on input
    future_X = [[future_year] + [df_filtered[col].mean() for col in cols_to_convert]]  # Use mean values for other features
    predicted_crime_future_year = model.predict(future_X)
    
    return f"Predicted crime for {district} in {future_year}: {predicted_crime_future_year[0]}"

if __name__ == '__main__':
    app.run(debug=True)
