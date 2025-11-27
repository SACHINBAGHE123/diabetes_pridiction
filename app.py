from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract input values from form
        features = [
            float(request.form["Pregnancies"]),
            float(request.form["Glucose"]),
            float(request.form["BloodPressure"]),
            float(request.form["SkinThickness"]),
            float(request.form["Insulin"]),
            float(request.form["BMI"]),
            float(request.form["DiabetesPedigree"]),
            float(request.form["Age"])
        ]

        # Convert to numpy structure
        input_data = np.array([features])

        # Predict
        prediction = model.predict(input_data)[0]
        result_prob = model.predict_proba(input_data)[0][1]

        result = "Diabetes Detected" if prediction == 1 else "No Diabetes"
        probability = f"{result_prob*100:.2f}%"

        return render_template("index.html", 
                               prediction=result,
                               probability=probability)

    except Exception as e:
        return render_template("index.html", error=str(e))


# if __name__ == "__main__":
#     app.run(debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
