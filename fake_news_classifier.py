import streamlit as st
import joblib

# Load your vectorizer and models (make sure these paths are correct)
vectorizer = joblib.load("vectorizer.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
rf_model = joblib.load("random_forest_model.pkl")

# App title
st.title("📰 Fake News Detection with Multiple Models")

# Language warning
st.warning("⚠️ This demo currently supports **English-language news only**. Using other languages or mixed-language text may result in incorrect predictions.")

# User input
news_text = st.text_area("Enter News Article Text", height=200)

if st.button("Detect Fake News"):
    if news_text.strip() == "":
        st.warning("Please enter some news text.")
    else:
        # Vectorize the input
        vectorized_input = vectorizer.transform([news_text])

        # Run each model and display results

        st.subheader("Naive Bayes Result")
        nb_prediction = nb_model.predict(vectorized_input)[0]
        st.info(f"Prediction: {'Fake' if nb_prediction == 1 else 'Real'}")

        st.subheader("Logistic Regression Result")
        lr_prediction = lr_model.predict(vectorized_input)[0]
        st.info(f"Prediction: {'Fake' if lr_prediction == 1 else 'Real'}")



