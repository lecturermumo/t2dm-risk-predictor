from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np

app = Flask(__name__)
booster = xgb.Booster()
booster.load_model("xgboost_model_booster.json")

FEATURE_NAMES = [
    'Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Fasting_Glucose',
    'Family_History', 'Physical_Activity', 'Alcohol_Use',
    'Education_Level', 'Income_Level', 'Smoking',
    'Gender', 'Marital_Status', 'Visit_Type', 'Facility_Name'
]

def categorize_risk(prob):
    if prob <= 0.33:
        return "Low"
    elif prob <= 0.66:
        return "Medium"
    else:
        return "High"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            input_data = []
            for feature in FEATURE_NAMES:
                value = request.form.get(feature)
                input_data.append(float(value))
            dmatrix = xgb.DMatrix(np.array([input_data]), feature_names=FEATURE_NAMES)
            risk_score = booster.predict(dmatrix)[0]
            risk_level = categorize_risk(risk_score)
            return render_template("result.html", score=round(risk_score, 4), level=risk_level)
        except Exception as e:
            return f"Error: {e}"
    return render_template("form.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
