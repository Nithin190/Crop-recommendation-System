from flask import Flask, render_template, request, redirect, send_file
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import io
import os

app = Flask(__name__)

client = MongoClient("mongodb://localhost:27017")
db = client["cropDB"]
collection = db["crop_recommendations"]
history_collection = db["prediction_history"]


df = pd.read_csv("Crop_recommendation.csv")
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {
        "N": float(request.form["N"]),
        "P": float(request.form["P"]),
        "K": float(request.form["K"]),
        "temperature": float(request.form["temperature"]),
        "humidity": float(request.form["humidity"]),
        "ph": float(request.form["ph"]),
        "rainfall": float(request.form["rainfall"])
    }

    input_df = pd.DataFrame([input_data])
    predicted_crop = model.predict(input_df)[0]
    input_data["recommended_crop"] = predicted_crop
    history_collection.insert_one(input_data)

    return render_template("index.html", prediction=predicted_crop)


@app.route("/history")
def history():
    records = list(history_collection.find({}, {"_id": 0}))
    return render_template("history.html", records=records)


@app.route("/download_csv")
def download_csv():
    records = list(history_collection.find({}, {"_id": 0}))
    df = pd.DataFrame(records)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="prediction_history.csv"
    )


@app.route("/chart")
def chart():
    records = list(history_collection.find({}, {"_id": 0}))
    df = pd.DataFrame(records)
    crop_counts = df["recommended_crop"].value_counts()

 
    plt.figure(figsize=(10, 6))
    crop_counts.plot(kind="bar", color="green")
    plt.title("Recommended Crop Frequency")
    plt.xlabel("Crop")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_path = "static/crop_chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("chart.html", chart_url=chart_path)


if __name__ == "__main__":
    app.run(debug=True)
