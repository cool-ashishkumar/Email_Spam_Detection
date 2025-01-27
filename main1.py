from fastapi import FastAPI
import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import CountVectorizer

# Initialize FastAPI app
app = FastAPI()

# Load the saved model and CountVectorizer
model = pickle.load(open("email_spam_model.pkl", "rb"))
cv = pickle.load(open("count_vectorizer.pkl", "rb"))

# Define preprocessing function
def preprocess_email(email):
    # Remove special characters, numbers, and extra spaces
    email = re.sub(r"[^a-zA-Z\s]", "", email)
    email = re.sub(r"\s+", " ", email).strip()
    email = email.lower()  # Convert to lowercase
    return email

# Streamlit UI
st.title("Email Spam Detection")
st.write("Predict if an email is **Spam** or **Not Spam**.")

# Input text box for the user
user_email = st.text_area("Enter your email:", placeholder="Type your email here...")

if st.button("Predict Spam"):
    if user_email.strip() == "":
        st.error("Please enter a valid email!")
    else:
        try:
            # Preprocess the input
            preprocessed_email = preprocess_email(user_email)

            # Vectorize the input
            email_vectorized = cv.transform([preprocessed_email]).toarray()

            # Predict spam
            prediction = model.predict(email_vectorized)[0]
            spam = "Spam" if prediction == 1 else "Not Spam"
            st.success(f"The email is: {spam}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
