from flask import Flask, request, render_template
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Initialize Flask app
app = Flask(__name__)

# Load and process data
file_path = 'D:/AIML_PROJECT/DistrictwiseSLLCrimes.csv'
data = pd.read_csv(file_path)
rajasthan_data = data[data['state_name'] == 'Rajasthan']
bharatpur_data = rajasthan_data[rajasthan_data['district_name'] == 'Bharatpur']

# Preprocess for model
scaler = StandardScaler()
bharatpur_data_scaled = scaler.fit_transform(bharatpur_data[['total_cognizable_sll_crimes', 'crime_against_women_total']])

X = bharatpur_data[['total_cognizable_sll_crimes', 'crime_against_women_total']]
y = bharatpur_data['total_cognizable_sll_crimes']

# Train Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Route for prediction page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = request.form['year']
    district = request.form['district']
    
    # Simulated predictions (replace these with actual prediction logic)
    predictions = {
        'crime_women': 150,
        'gambling': 30,
        'child_marriage': 5,
        'juvenile_justice': 10,
        'it_ip_crimes': 20
    }

    return render_template(
        'index.html', 
        prediction_crime_women=f"Predicted {predictions['crime_women']} cases",
        prediction_gambling=f"Predicted {predictions['gambling']} cases",
        prediction_child_marriage=f"Predicted {predictions['child_marriage']} cases",
        prediction_juvenile_justice=f"Predicted {predictions['juvenile_justice']} cases",
        prediction_it_ip_crimes=f"Predicted {predictions['it_ip_crimes']} cases"
    )

if __name__ == '__main__':
    app.run(debug=True)
