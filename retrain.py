# ============================================
# retrain.py
# Retrain models using new dataset
# ============================================

import pandas as pd
import joblib
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB


# ============================================
# STEP 1: LOAD NEW DATASET
# ============================================
new_data = pd.read_csv("new_retail_data.csv")

print("New dataset loaded successfully.")
print(new_data.head())

print("\nColumns in new dataset:")
print(new_data.columns.tolist())


# ============================================
# STEP 2: BASIC CLEANING
# ============================================
new_data.columns = new_data.columns.str.strip()

for col in new_data.select_dtypes(include="object").columns:
    new_data[col] = new_data[col].astype(str).str.strip()

new_data = new_data.replace("nan", pd.NA)


# ============================================
# STEP 3: PREPARE SALES FEATURES
# ============================================
new_data["order_date"] = pd.to_datetime(new_data["order_date"], errors="coerce")

# Create day_number from order_date
new_data["day_number"] = new_data["order_date"].dt.day

# Convert needed columns to numeric
new_data["is_festival_season"] = pd.to_numeric(new_data["is_festival_season"], errors="coerce")
new_data["delivery_days"] = pd.to_numeric(new_data["delivery_days"], errors="coerce")
new_data["customer_tenure_days"] = pd.to_numeric(new_data["customer_tenure_days"], errors="coerce")
new_data["final_amount"] = pd.to_numeric(new_data["final_amount"], errors="coerce")

# Fill missing values
new_data["day_number"] = new_data["day_number"].fillna(0)
new_data["is_festival_season"] = new_data["is_festival_season"].fillna(0)
new_data["delivery_days"] = new_data["delivery_days"].fillna(new_data["delivery_days"].median())
new_data["customer_tenure_days"] = new_data["customer_tenure_days"].fillna(
    new_data["customer_tenure_days"].median()
)
new_data["final_amount"] = new_data["final_amount"].fillna(new_data["final_amount"].median())


# ============================================
# STEP 4: RETRAIN SALES MODEL
# Create a fresh model instead of loading old one
# ============================================
X_sales = new_data[[
    "day_number",
    "is_festival_season",
    "delivery_days",
    "customer_tenure_days"
]]

y_sales = new_data["final_amount"]

sales_model = LinearRegression()
sales_model.fit(X_sales, y_sales)

joblib.dump(sales_model, "models/sales_forecasting_model.pkl")

print("Sales model retrained and saved successfully.")


# ============================================
# STEP 5: CLEAN TEXT FOR SENTIMENT MODEL
# ============================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


new_data["review_text"] = new_data["review_text"].fillna("No review provided")
new_data["clean_review"] = new_data["review_text"].apply(clean_text)


# ============================================
# STEP 6: PREPARE SENTIMENT TARGET
# ============================================
new_data["sentiment"] = pd.to_numeric(new_data["sentiment"], errors="coerce")
new_data["sentiment"] = new_data["sentiment"].fillna(0)

X_sentiment = new_data["clean_review"]
y_sentiment = new_data["sentiment"]


# ============================================
# STEP 7: RETRAIN SENTIMENT MODEL
# Create fresh vectorizer and fresh model
# ============================================
tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X_sentiment_tfidf = tfidf_vectorizer.fit_transform(X_sentiment)

sentiment_model = MultinomialNB()
sentiment_model.fit(X_sentiment_tfidf, y_sentiment)

joblib.dump(sentiment_model, "models/sentiment_model.pkl")
joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.pkl")

print("Sentiment model retrained and saved successfully.")


# ============================================
# STEP 8: FINAL MESSAGE
# ============================================
print("\nAll models updated successfully.")