from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model & scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/", methods=["GET","POST"])
def home():
    if request.method == "POST":
        features = [float(x) for x in request.form.values()]
        final = scaler.transform([features])
        prediction = model.predict(final)

        return render_template("index.html", prediction_text=f"Predicted House Value: {prediction[0]:.2f}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)