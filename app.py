from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)

model = joblib.load("customer_churn_model.pkl")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["gender"])
    senior = int(request.form["senior"])
    partner = int(request.form["partner"])
    dependents = int(request.form["dependents"])
    tenure = int(request.form["tenure"])
    internet = int(request.form["internet"])
    contract = int(request.form["contract"])
    monthly = float(request.form["monthly"])
    total = float(request.form["total"])

    data = pd.DataFrame([[
        gender,
        senior,
        partner,
        dependents,
        tenure,
        internet,
        contract,
        monthly,
        total
    ]], columns=[
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Internet Service",
        "Contract",
        "Monthly Charges",
        "Total Charges"
    ])

    prediction = model.predict(data)

    probability = model.predict_proba(data)[0][1]
    probability = round(probability * 100, 2)

    if prediction[0] == 1:
        result = "Customer will Churn"
    else:
        result = "Customer will Not Churn"

    return render_template(
        "index.html",
        prediction=result,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)