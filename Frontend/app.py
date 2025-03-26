# import streamlit as st
# import requests

# st.title("🎤 Stress Detection using AI")

# uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])

# if uploaded_file is not None:
#     st.audio(uploaded_file, format='audio/wav')

#     with st.spinner("🔍 Analyzing..."):
#         audio_bytes = uploaded_file.read()
#         response = requests.post("http://localhost:8000/predict/", files={"file": audio_bytes})

#         try:
#             if response.status_code == 200:
#                 emotion = response.json().get("emotion", "Unknown")
#                 st.success(f"Predicted Emotion: **{emotion}**")
#             else:
#                 st.error(f"❌ Error: {response.text}")
#         except Exception as e:
#             st.error(f"❌ Error processing response: {e}")



import streamlit as st
import requests
from PIL import Image
import io

st.title("Mental Health Detection using AI")
st.write("Detect Mental Health using either speech or an image.")

# Buttons for selecting method
option = st.radio("Choose Detection Method:", ["Mental Health using Speech", "Mental Health using Image"])

if option == "Mental Health using Speech":
    uploaded_file = st.file_uploader("Upload an Audio File", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        with st.spinner("🔍 Analyzing..."):
            audio_bytes = uploaded_file.read()
            response = requests.post("http://localhost:8000/predict/audio", files={"file": audio_bytes})
            try:
                if response.status_code == 200:
                    emotion = response.json().get("emotion", "Unknown")
                    st.success(f"Predicted Emotion: **{emotion}**")
                else:
                    st.error(f"❌ Error: {response.text}")
            except Exception as e:
                st.error(f"❌ Error processing response: {e}")

elif option == "Mental Health using Image":
    uploaded_image = st.file_uploader("Upload an Image File", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        with st.spinner("🔍 Analyzing..."):
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")
            image_bytes = image_bytes.getvalue()
            response = requests.post("http://localhost:8000/predict/image", files={"file": image_bytes})
            try:
                if response.status_code == 200:
                    emotion = response.json().get("emotion", "Unknown")
                    st.success(f"Predicted Emotion: **{emotion}**")
                else:
                    st.error(f"❌ Error: {response.text}")
            except Exception as e:
                st.error(f"❌ Error processing response: {e}")