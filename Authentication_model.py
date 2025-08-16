import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image

# Load the trained model
pickle_in = open("classifier.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_note_authentication(variance, skewness, curtosis, entropy):
    """
    Authenticate the Bank Notes.
    """
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return prediction

def main():
    # Page Config
    st.set_page_config(page_title="Bank Note Authenticator", page_icon="🏦", layout="centered")

    # Custom CSS
    st.markdown("""
        <style>
        body {
            background-color: #f0f4f8;
        }
        .main {
            background: linear-gradient(135deg, #ffffff, #e6ecff);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            font-family: 'Segoe UI', sans-serif;
            text-align: center;
            color: #002b5c;
        }
        .stTextInput>div>div>input {
            border: 2px solid #b3c7e6;
            border-radius: 8px;
            padding: 8px;
        }
        .stButton>button {
            background-color: #002b5c;
            color: white;
            font-size: 16px;
            border-radius: 10px;
            padding: 10px 20px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #004080;
            transform: scale(1.05);
        }
        .prediction-box {
            background-color: #e6f7ff;
            border-left: 5px solid #0099cc;
            padding: 15px;
            border-radius: 8px;
            font-size: 16px;
            color: #00334d;
            margin-top: 15px;
        }
        .footer {
            text-align: center;
            font-size: 13px;
            margin-top: 20px;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown("<h1>🏦 Bank Note Authenticator</h1>", unsafe_allow_html=True)
    st.write("This app predicts whether a bank note is **authentic** or **fake** using a Machine Learning model trained on statistical features.")

    # Input fields
    variance = st.text_input("📊 Variance", placeholder="Enter variance value")
    skewness = st.text_input("📈 Skewness", placeholder="Enter skewness value")
    curtosis = st.text_input("📉 Curtosis", placeholder="Enter curtosis value")
    entropy = st.text_input("♾ Entropy", placeholder="Enter entropy value")

    result = ""
    if st.button("🔍 Predict"):
        if variance and skewness and curtosis and entropy:
            try:
                variance = float(variance)
                skewness = float(skewness)
                curtosis = float(curtosis)
                entropy = float(entropy)
                result = predict_note_authentication(variance, skewness, curtosis, entropy)

                # Show result meaning
                if result[0] == 0:
                    st.markdown("<div class='prediction-box'>❌ Prediction: <b>Fake Note</b> (0)</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='prediction-box'>✅ Prediction: <b>Genuine Note</b> (1)</div>", unsafe_allow_html=True)
            except ValueError:
                st.error("❌ Please enter valid numeric values for all fields.")
        else:
            st.warning("⚠ Please fill in all fields before predicting.")

    if st.button("ℹ About"):
        st.markdown("""
        ### ℹ About This App
        This **Bank Note Authentication App** uses a Machine Learning model to determine whether a given banknote is **authentic** or **fake**.

        #### 🔍 How It Works:
        1. You provide four numbers about the bank note:
            - **Variance** 📊 — How much the image data varies from the average. Large differences can signal unusual patterns.
            - **Skewness** 📈 — Tells if the data is tilted to one side. Counterfeit notes often show abnormal tilting patterns.
            - **Curtosis** 📉 — Measures how sharp or flat the data curve is. Fake notes can have curves that are too sharp or too flat compared to real ones.
            - **Entropy** ♾ — Shows how complex or "messy" the image is. Real notes have a certain level of natural complexity that fakes may lack.
        2. The model processes these inputs and makes a prediction.
        3. The output will be:
           - **0 → Fake Note** (Likely counterfeit)
           - **1 → Genuine Note** (Likely authentic)

        #### 📦 Technology Stack:
        - **Python**
        - **scikit-learn** for Machine Learning
        - **Streamlit** for Web Deployment-Framework
        - **Pickle** for Model Serialization

        #### 🛡 Why Bank Note Authentication?
        Counterfeit currency is a major issue worldwide. This project demonstrates how AI can help quickly and reliably identify fake notes, reducing fraud risks.

        #### 👨‍💻 Developer:
        - **BISWAJIT SOW**
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("<div class='footer'>© 2025 Bank Note Authenticator | Engineered by SweetPoison</div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
