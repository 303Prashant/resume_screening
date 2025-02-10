import streamlit as st
import pickle
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer

with open('attrition_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Attrition Risk Prediction", layout="wide")
st.markdown("Predict the attrition risk based on employee feedback.")
st.title("Employee Attrition Risk Prediction")


feedback = st.text_area("Enter Employee Feedback", height=200, max_chars=500, placeholder="Type here...")

def predict_feedback(feedback_text):
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(feedback_text)["compound"]
    
    negative_words = ["stress", "burnout", "toxic", "micromanagement", "poor",]
    negative_count = sum(1 for word in feedback_text.split() if word in negative_words)
    
  
    features = np.array([sentiment_score, negative_count]).reshape(1, -1)
    prediction = model.predict(features)
    
    return prediction[0]
if st.button("Predict Attrition Risk"):
    if feedback:
        prediction = predict_feedback(feedback)
        if prediction == "High":
            st.error("Attrition Risk is **High**. Take action immediately!")
        elif prediction == "Medium":
            st.warning("Attrition Risk is **Medium**. Consider improving engagement.")
        else:
            st.success("Attrition Risk is **Low**. Keep up the good work!")
    else:
        st.error("Please enter feedback to predict attrition risk.")
