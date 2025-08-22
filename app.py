import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://localhost:8000/predict"

# ------------------ Custom CSS ------------------
st.markdown("""
    <style>
    /* Background */
    body {
        background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        font-family: 'Arial', sans-serif;
    }

    /* Navbar Styling - Amazon-like */
    .navbar {
        background-color: #ffffff;
        padding: 15px 30px;
        border-bottom: 2px solid #f0f0f0;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }

    .navbar-logo {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
    }

    .navbar-links a {
        margin: 0 15px;
        text-decoration: none;
        color: #333;
        font-size: 16px;
        font-weight: 500;
        transition: color 0.3s;
    }

    .navbar-links a:hover {
        color: #0073e6;
    }

    /* Title Section */
    .main-title {
        text-align: center;
        color: #2c3e50;
        font-size: 36px;
        font-weight: bold;
        margin-top: -10px;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #555;
        font-size: 18px;
        margin-bottom: 20px;
    }

    /* Image Frame */
    .image-frame {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }

    .image-frame img {
        border: 3px solid #ddd;
        border-radius: 10px;
        padding: 4px;
        max-width: 80%;
        background: white;
    }

    /* Prediction Card */
    .prediction-card {
        background: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        font-size: 26px;
        font-weight: bold;
        color: #0073e6;
    }

    .confidence {
        font-size: 18px;
        color: #444;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Navbar ------------------
st.markdown("""
<div class="navbar">
    <div class="navbar-logo">😃 EmotionAI</div>
    <div class="navbar-links">
        <a href="#">Home</a>
        <a href="#">About</a>
        <a href="#">Help</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ------------------ Main Logic ------------------
def main():
    st.markdown('<div class="main-title">Upload and Detect</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered human emotion detection system</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        # Display image with border
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="JPEG")
        st.markdown('<div class="image-frame">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Convert image to bytes for API
        img_bytes = img_buffer.getvalue()

        try:
            response = requests.post(API_URL, files={"file": img_bytes})

            if response.status_code == 200:
                result = response.json()
                st.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-label">Emotion: {result['emotion']}</div>
                        <div class="confidence">Confidence: {result['confidence']:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.error(f"API Error: {response.status_code}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to the API. Make sure the FastAPI server is running.")

if __name__ == "__main__":
    main()
