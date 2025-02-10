import streamlit as st
import pdfplumber
import docx
import google.generativeai as genai
import time
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
import os
# yha abhi security dalni h
google_api_key = "my api"

genai.configure(api_key=google_api_key)

# === MODEL LOADING ===
class ResumeClass(nn.Module):
    def __init__(self, input_size=100, hidden_size=50, num_classes=2):
        super(ResumeClass, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Try loading the model
try:
    model = torch.load("resume_screening.pth",)
    model.eval()
except Exception as e:
    st.error(f"Error loading model: {e}")

# === TEXT EXTRACTION FUNCTIONS ===

def extract_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip() if text.strip() else None

def extract_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    return text.strip() if text.strip() else None

# === AI Functions (LLM Prompt Engineering) ===
def summarize_resume(resume_text):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
    I am hiring assistant. I want summary of resume as:
    Key Skills
    Work Experience
    Highest Qualification
    Certifications

    Resume Text: {resume_text}
    """
    response = model.generate_content(prompt)
    return response.text

def required_skills(job_description):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
    extract skills from job description:
    Job Description: {job_description}
    """
    response = model.generate_content(prompt)
    return response.text.split(", ")

def explain_match(resume_text, job_description, match_score):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"""
    compare the job description with resume and show the match score.
    Resume:
    {resume_text}

    Job Description:
    {job_description}
    
    Match Score: {match_score}%
    
    give detailed explanation why match score is that. 
    **Matched Skills:** 
    **Missing Skills:**
    **Experience Match:** 
    **Suggestions:**
    """
    response = model.generate_content(prompt)
    return response.text

# Function to extract keywords from job description
def extract_keywords(job_description):
    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit([job_description])
    return vectorizer.get_feature_names_out()

# Function to predict match score
def predict_match(resume_text, job_description):
    keywords = extract_keywords(job_description)
    score = sum(1 for word in keywords if word in resume_text.lower()) / len(keywords)
    return round(score * 100, 2)

# === STREAMLIT UI ===
st.title("AI Resume Screening Tool")
st.write("Upload a resume and get a match score for the Software Engineer role.")

uploaded_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Enter Job Description")

if uploaded_file and job_description:
    file_extension = uploaded_file.name.split(".")[-1]

    # Extract text based on file type
    resume_text = extract_pdf(uploaded_file) if file_extension == "pdf" else extract_docx(uploaded_file)

    if resume_text:
        # st.subheader("Extracted Resume Content (Preview)")
        # st.text_area("Resume Text", resume_text, height=150)

        # Calculate match score
        match_score = predict_match(resume_text, job_description)

        # Display results
        st.write(f"**Match Score: {match_score}%**")
        if match_score > 70:
            st.success("This resume is a strong match!")
        elif match_score > 40:
            st.warning("This resume is a moderate match.")
        else:
            st.error("This resume is a weak match.")

        # Get AI Summary
        summary = summarize_resume(resume_text)
        st.subheader("AI-Extracted Resume Summary")
        st.write(summary)

        # Get Required Skills
        required_skills = required_skills(job_description)
        st.subheader("Required Skills from Job Description")
        st.write(", ".join(required_skills))

        # Explain Match Score
        explanation = explain_match(resume_text, job_description, match_score)
        st.subheader("AI Explanation for Match Score")
        st.write(explanation)

    else:
        st.error("Unable to extract text from the resume. Please upload a readable document.")


#         # Display results
#         st.write(f"**Match Score: {match_score}%**")
#         if match_score > 70:
#             st.success("This resume is a strong match!")
#         elif match_score > 40:
#             st.warning("This resume is a moderate match.")
#         else:
#             st.error("This resume is a weak match.")

# streamlit run app.py