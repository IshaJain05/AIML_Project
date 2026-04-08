from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

file_path = "I:\\District-Crime-Prediction\\Districtwise SLL Crimes.csv"


def preprocess_data():
    df = pd.read_csv(file_path)

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df.dropna(subset=["year"], inplace=True)
    df["year"] = df["year"].astype(int)

    cols_to_convert = [
        "crime_against_women_total",
        "juvenile_justice_care_and_protection_of_children",
        "prohibition_of_child_marriage",
        "sc_and_st_related_crimes",
        "prevention_of_damage_to_public_property",
        "arms_total",
        "explosives_and_explosive_substances",
        "information_technology_or_intellectual_property_total",
        "prohibition_state",
        "excise",
        "ndps_total",
        "forest_act_1927_and_the_forest_conservation",
        "foreigner_and_passport_related_total",
        "food_drugs_and_essential_commodities_total",
        "gambling",
        "electricity",
        "other_sll_crimes"
    ]

    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.fillna(0, inplace=True)
    return df, cols_to_convert


df, cols_to_convert = preprocess_data()


def format_crime_name(name):
    return name.replace("_", " ").title()


@app.route("/")
def index():
    states = sorted(df["state_name"].dropna().unique())
    state_districts = {
        state: sorted(
            df[df["state_name"] == state]["district_name"].dropna().unique().tolist()
        )
        for state in states
    }
    return render_template("index.html", states=states, state_districts=state_districts)


@app.route("/predict", methods=["POST"])
def predict():
    state = request.form["state"]
    district = request.form["district"]
    future_years = int(request.form["years"])

    df_filtered = df[
        (df["state_name"] == state) & (df["district_name"] == district)
    ].copy()

    if df_filtered.empty:
        return render_template(
            "result.html",
            district=district,
            year="N/A",
            predictions=[],
            total_prediction=0,
            error="No data available for this state and district combination."
        )

    future_start_year = 2025
    future_year = future_start_year + future_years - 1

    predictions = []

    for target_col in cols_to_convert:
        feature_columns = ["year"] + [col for col in cols_to_convert if col != target_col]

        X = df_filtered[feature_columns]
        y = df_filtered[target_col]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        future_input = [future_year]
        for col in cols_to_convert:
            if col != target_col:
                future_input.append(df_filtered[col].mean())

        predicted_value = model.predict([future_input])[0]
        predicted_value = max(0, round(float(predicted_value), 2))

        predictions.append({
            "crime_type": format_crime_name(target_col),
            "predicted_value": predicted_value
        })

    total_prediction = round(sum(item["predicted_value"] for item in predictions), 2)

    return render_template(
        "result.html",
        district=district,
        year=future_year,
        predictions=predictions,
        total_prediction=total_prediction,
        error=None
    )


if __name__ == "__main__":
    app.run(debug=True)