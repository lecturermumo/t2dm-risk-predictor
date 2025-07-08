
from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np

app = Flask(__name__)
booster = xgb.Booster()
booster.load_model("xgboost_model_booster.json")

def encode_age(age):
    age = int(age)
    if age <= 30:
        return 1
    elif age <= 45:
        return 2
    elif age <= 60:
        return 3
    else:
        return 4

def encode_bmi(bmi):
    bmi = float(bmi)
    if bmi < 18.5:
        return 1
    elif bmi < 25:
        return 2
    elif bmi < 30:
        return 3
    else:
        return 4

def encode_glucose(glucose):
    glucose = float(glucose)
    if glucose < 5.6:
        return 1
    elif glucose < 6.9:
        return 2
    else:
        return 3

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
            # Raw form values
            age = request.form["Age"]
            bmi = request.form["BMI"]
            systolic = float(request.form["Systolic_BP"])
            diastolic = float(request.form["Diastolic_BP"])
            glucose = request.form["Fasting_Glucose"]

            # Encoded features
            age_cat = encode_age(age)
            bmi_cat = encode_bmi(bmi)
            glucose_cat = encode_glucose(glucose)

            # Other categorical fields (already encoded from dropdown)
            family_history = int(request.form["Family_History"])
            physical_activity = int(request.form["Physical_Activity"])
            alcohol_use = int(request.form["Alcohol_Use"])
            education = int(request.form["Education_Level"])
            income = int(request.form["Income_Level"])
            smoking = int(request.form["Smoking"])

            features = [age_cat, bmi_cat, systolic, diastolic, glucose_cat,
                        family_history, physical_activity, alcohol_use,
                        education, income, smoking]

            dmatrix = xgb.DMatrix(np.array([features]))
            prob = booster.predict(dmatrix)[0]
            level = categorize_risk(prob)
            return render_template("result.html", score=round(prob * 100, 2), level=level)
        except Exception as e:
            return f"Error occurred: {e}"
    return render_template("form.html")
