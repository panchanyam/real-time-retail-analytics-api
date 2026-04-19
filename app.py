# Import Flask tools

from flask import Flask, render_template, request, jsonify

# Import pandas for dataframe handling
import pandas as pd

# Import joblib to load saved models
import joblib

# Import re for text cleaning
import re

# Create Flask app
app = Flask(__name__)

# Load saved sentiment model
sentiment_model = joblib.load("models/sentiment_model.pkl")

# Load saved TF-IDF vectorizer
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load saved sales forecasting model
sales_model = joblib.load("models/sales_forecasting_model.pkl")

# Load cleaned dataset for dashboard numbers
df = pd.read_csv("data/retail_dataset.csv")


# Function to clean user review text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Return cleaned text
    return text


# Home page route
@app.route("/")
def home():
    # Calculate simple KPI values from dataset
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_id"].nunique()
    total_sales = round(df["final_amount"].sum(), 2)
    positive_reviews = int((df["sentiment"] == 1).sum())

    # Send these values to HTML page
    return render_template(
        "index.html",
        total_orders=total_orders,
        total_customers=total_customers,
        total_sales=total_sales,
        positive_reviews=positive_reviews
    )


# Sentiment prediction route
@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    # Get review text from form
    review_text = request.form["review_text"]

    # Clean the review text
    cleaned_review = clean_text(review_text)

    # Convert text to TF-IDF features
    review_vector = tfidf_vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = sentiment_model.predict(review_vector)[0]

    # Convert numeric output into label
    if prediction == 1:
        sentiment_result = "Positive Review"
        sentiment_message = "This customer review shows satisfaction."
    else:
        sentiment_result = "Negative Review"
        sentiment_message = "This customer review shows dissatisfaction."

    # Recalculate dashboard KPI values
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_id"].nunique()
    total_sales = round(df["final_amount"].sum(), 2)
    positive_reviews = int((df["sentiment"] == 1).sum())

    # Return page with prediction result
    return render_template(
        "index.html",
        sentiment_result=sentiment_result,
        sentiment_message=sentiment_message,
        total_orders=total_orders,
        total_customers=total_customers,
        total_sales=total_sales,
        positive_reviews=positive_reviews
    )


# Sales prediction route
@app.route("/predict_sales", methods=["POST"])
def predict_sales():
    # Get day number from form
    day_number = int(request.form["day_number"])

    # Create dataframe for prediction
    input_data = pd.DataFrame({"day_number": [day_number]})

    # Predict sales
    predicted_sales = sales_model.predict(input_data)[0]

    # Round output
    predicted_sales = round(predicted_sales, 2)

    # Recalculate dashboard KPI values
    total_orders = df["order_id"].nunique()
    total_customers = df["customer_id"].nunique()
    total_sales = round(df["final_amount"].sum(), 2)
    positive_reviews = int((df["sentiment"] == 1).sum())

    # Return page with result
    return render_template(
        "index.html",
        sales_result=predicted_sales,
        total_orders=total_orders,
        total_customers=total_customers,
        total_sales=total_sales,
        positive_reviews=positive_reviews
    )

@app.route("/api/sentiment", methods=["POST"])
def api_sentiment():
    data = request.get_json()

    review_text = data.get("review_text")

    if not review_text:
        return jsonify({"error": "review_text is required"}), 400

    cleaned_review = clean_text(review_text)

    review_vector = tfidf_vectorizer.transform([cleaned_review])

    prediction = sentiment_model.predict(review_vector)[0]

    result = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "review_text": review_text,
        "prediction": result
    })

@app.route("/api/sales", methods=["POST"])
def api_sales():
    data = request.get_json()

    day_number = data.get("day_number")

    if day_number is None:
        return jsonify({"error": "day_number is required"}), 400

    input_df = pd.DataFrame({"day_number": [day_number]})

    prediction = sales_model.predict(input_df)[0]

    return jsonify({
        "day_number": day_number,
        "predicted_sales": round(float(prediction), 2)
    })


# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)