# Import Flask tools
from flask import Flask, request, jsonify

# Import pandas for dataframe creation
import pandas as pd

# Import joblib to load saved models
import joblib

# Import re for text cleaning
import re

# Import time to measure response time
import time

# Create Flask app
app = Flask(__name__)

# Load saved models
sales_model = joblib.load("models/sales_forecasting_model.pkl")
sentiment_model = joblib.load("models/sentiment_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")


# Function to clean text
def clean_text(text):
    # Convert text to lowercase
    text = str(text).lower()

    # Remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Home route
@app.route("/", methods=["GET"])
def home():
    return "Real-Time Retail Sales Analytics API is running successfully"


# Sentiment prediction API
@app.route("/api/sentiment", methods=["POST"])
def predict_sentiment():
    # Start timer
    start_time = time.time()

    try:
        # Get JSON data from request
        data = request.get_json()

        # Check whether review_text is present
        if not data or "review_text" not in data:
            return jsonify({"error": "review_text is required"}), 400

        # Get review text
        review_text = data["review_text"]

        # Clean the input review
        cleaned_review = clean_text(review_text)

        # Convert review to TF-IDF format
        review_vector = tfidf_vectorizer.transform([cleaned_review])

        # Predict sentiment
        prediction = sentiment_model.predict(review_vector)[0]

        # Convert numeric result to label
        sentiment_label = "Positive" if prediction == 1 else "Negative"

        # End timer
        end_time = time.time()

        # Calculate response time
        response_time = round(end_time - start_time, 4)

        # Print monitoring info in terminal
        print(f"/api/sentiment completed in {response_time} seconds")

        # Return JSON output
        return jsonify({
            "input_review": review_text,
            "predicted_sentiment": sentiment_label,
            "response_time_seconds": response_time
        })

    except Exception as e:
        # End timer
        end_time = time.time()

        # Calculate response time even for errors
        response_time = round(end_time - start_time, 4)

        # Print error log
        print(f"/api/sentiment failed in {response_time} seconds")

        return jsonify({
            "error": str(e),
            "response_time_seconds": response_time
        }), 500


# Sales prediction API
@app.route("/api/sales", methods=["POST"])
def predict_sales():
    # Start timer
    start_time = time.time()

    try:
        # Get JSON data from request
        data = request.get_json()

        # Validate required fields
        required_fields = [
            "day_number",
            "is_festival_season",
            "avg_delivery_days",
            "avg_customer_tenure_days"
        ]

        for field in required_fields:
            if not data or field not in data:
                return jsonify({"error": f"{field} is required"}), 400

        # Create input dataframe
        input_data = pd.DataFrame({
            "day_number": [data["day_number"]],
            "is_festival_season": [data["is_festival_season"]],
            "avg_delivery_days": [data["avg_delivery_days"]],
            "avg_customer_tenure_days": [data["avg_customer_tenure_days"]]
        })

        # Predict sales
        prediction = sales_model.predict(input_data)[0]

        # End timer
        end_time = time.time()

        # Calculate response time
        response_time = round(end_time - start_time, 4)

        # Print monitoring info in terminal
        print(f"/api/sales completed in {response_time} seconds")

        # Return JSON response
        return jsonify({
            "input": {
                "day_number": data["day_number"],
                "is_festival_season": data["is_festival_season"],
                "avg_delivery_days": data["avg_delivery_days"],
                "avg_customer_tenure_days": data["avg_customer_tenure_days"]
            },
            "predicted_sales": round(float(prediction), 2),
            "response_time_seconds": response_time
        })

    except Exception as e:
        # End timer
        end_time = time.time()

        # Calculate response time even for errors
        response_time = round(end_time - start_time, 4)

        # Print error log
        print(f"/api/sales failed in {response_time} seconds")

        return jsonify({
            "error": str(e),
            "response_time_seconds": response_time
        }), 500


# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)