# Title:
Real Time Retail Analytics and Customer Insight Platform
# Introduction:
This project is a real time retail analytics and customer insight platform developed using machine learning and Flask API. The system is designed to analyze retail data, predict sales performance, and understand customer sentiment from feedback. It provides a complete end to end pipeline including data processing, model training, API deployment, and automated model retraining.
# Objective:
The main objective of this project is to build an intelligent system that can help retail businesses make data driven decisions. It focuses on predicting sales trends and analyzing customer feedback to improve business performance and customer satisfaction.
# Dataset Description
The dataset used in this project consists of retail transaction and customer feedback data. The key columns include order date, customer identifier, product identifier, product category, final transaction amount, customer review text, sentiment label, customer tenure in days, festival season indicator, and delivery duration.
Data Processing:
The raw data is cleaned and prepared before model training. The cleaning process includes handling missing values, removing duplicates, converting data types, and standardizing text fields. Additional features such as day number from the order date are created to improve model performance.
# Machine Learning Models:
## Sales Forecasting Model
A regression model is used to predict the final transaction amount. The model is trained using features such as day number, festival season indicator, delivery duration, and customer tenure.
## Sentiment Analysis Model
A text classification model is used to analyze customer reviews. The review text is cleaned and converted into numerical format using TF IDF vectorization. A classification algorithm is then used to predict sentiment as positive or negative.
# Model Training and Retraining
The project includes a retraining pipeline that allows the models to be updated using new incoming data. The retraining script loads the new dataset, prepares the features, retrains the models, and saves the updated models. This ensures that the system stays accurate over time.
# API Development
A Flask API is developed to serve the machine learning models. The API allows users to send input data and receive predictions for sales and sentiment. It provides a simple and efficient interface for integrating the models into real world applications.
# Project Structure
The project is organized into different folders for better maintainability. The data folder contains datasets, the models folder stores trained models, and the main application files include the API and retraining scripts.
# Technologies Used
Python is used as the primary programming language. Pandas and NumPy are used for data processing. Scikit learn is used for machine learning models. Flask is used for API development. Joblib is used for saving and loading models.
# How to Run the Project
- Install the required libraries using the requirements file.
- Run the training or retraining script to generate models.
- Start the Flask application to launch the API.
- Use tools like Postman to send requests and test predictions.
# Applications

# Title
Real Time Retail Analytics and Customer Insight Platform
# Introduction
This project is a real time retail analytics and customer insight platform developed using machine learning and Flask API. The system is designed to analyze retail data, predict sales performance, and understand customer sentiment from feedback. It provides a complete end to end pipeline including data processing, model training, API deployment, and automated model retraining.
# Objective
The main objective of this project is to build an intelligent system that can help retail businesses make data driven decisions. It focuses on predicting sales trends and analyzing customer feedback to improve business performance and customer satisfaction.
# Dataset Description
The dataset used in this project consists of retail transaction and customer feedback data. The key columns include order date, customer identifier, product identifier, product category, final transaction amount, customer review text, sentiment label, customer tenure in days, festival season indicator, and delivery duration.
# Data Processing
The raw data is cleaned and prepared before model training. The cleaning process includes handling missing values, removing duplicates, converting data types, and standardizing text fields. Additional features such as day number from the order date are created to improve model performance.
# Machine Learning Models
# Sales Forecasting Model
A regression model is used to predict the final transaction amount. The model is trained using features such as day number, festival season indicator, delivery duration, and customer tenure.
# Sentiment Analysis Model
A text classification model is used to analyze customer reviews. The review text is cleaned and converted into numerical format using TF IDF vectorization. A classification algorithm is then used to predict sentiment as positive or negative.
# Model Training and Retraining
The project includes a retraining pipeline that allows the models to be updated using new incoming data. The retraining script loads the new dataset, prepares the features, retrains the models, and saves the updated models. This ensures that the system stays accurate over time.
# API Development
A Flask API is developed to serve the machine learning models. The API allows users to send input data and receive predictions for sales and sentiment. It provides a simple and efficient interface for integrating the models into real world applications.
# Project Structure
The project is organized into different folders for better maintainability. The data folder contains datasets, the models folder stores trained models, and the main application files include the API and retraining scripts.
# Technologies Used
Python is used as the primary programming language. Pandas and NumPy are used for data processing. Scikit learn is used for machine learning models. Flask is used for API development. Joblib is used for saving and loading models.
# How to Run the Project
- Install the required libraries using the requirements file.
- Run the training or retraining script to generate models.
- Start the Flask application to launch the API.
- Use tools like Postman to send requests and test predictions.
# Applications
This project can be used in retail businesses for sales prediction and customer behavior analysis. It helps in understanding customer satisfaction and improving marketing strategies. It can also be extended for real time dashboards and business intelligence systems.
# Conclusion
This project demonstrates a complete machine learning workflow from data processing to deployment. It combines predictive analytics and natural language processing to deliver meaningful insights for retail businesses. The inclusion of automated retraining makes the system adaptable and scalable for real time use cases.
This project demonstrates a complete machine learning workflow from data processing to deployment. It combines predictive analytics and natural language processing to deliver meaningful insights for retail businesses. The inclusion of automated retraining makes the system adaptable and scalable for real time use cases.
