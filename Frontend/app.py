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

# import streamlit as st
# import requests
# from PIL import Image
# import io
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt

# # FastAPI Backend URL
# API_URL = "http://127.0.0.1:8000"

# st.title("🧠 Mental Health Detection & Chatbot")

# # Tabs for functionalities
# tab1, tab2, tab3 = st.tabs(["🎤 Audio Emotion", "🖼️ Image Emotion", "💬 Chatbot"])

# # 🎤 Audio Emotion Detection
# with tab1:
#     st.subheader("Upload an Audio File for Emotion Analysis")
#     audio_file = st.file_uploader("Choose a WAV file", type=["wav"])
    
#     if audio_file:
#         st.audio(audio_file, format='audio/wav')
        
#         # Display waveform preview
#         st.subheader("Audio Waveform Preview")
#         audio_bytes = audio_file.getvalue()
#         y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
#         fig, ax = plt.subplots()
#         librosa.display.waveshow(y, sr=sr, ax=ax)
#         ax.set_title("Waveform of Uploaded Audio")
#         st.pyplot(fig)
        
#         if st.button("Analyze Audio"):
#             files = {"file": audio_file.getvalue()}
#             response = requests.post(f"{API_URL}/predict/audio/", files=files)
            
#             if response.status_code == 200:
#                 result = response.json()
#                 st.success(f"Emotion Detected: {result['emotion']} (Confidence: {result['confidence']:.2f}%)")
#             else:
#                 st.error("Error processing the audio.")

# # 🖼️ Image Emotion Detection
# with tab2:
#     st.subheader("Upload an Image for Emotion Analysis")
#     image_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    
#     if image_file:
#         image = Image.open(image_file)
#         st.image(image, caption="Uploaded Image Preview", width=250)
        
#         if st.button("Analyze Image"):
#             files = {"file": image_file.getvalue()}
#             response = requests.post(f"{API_URL}/predict/image/", files=files)
            
#             if response.status_code == 200:
#                 result = response.json()
#                 st.success(f"Emotion Detected: {result['emotion']} (Confidence: {result['confidence']:.2f}%)")
#             else:
#                 st.error("Error processing the image.")

# # 💬 Chatbot
# with tab3:
#     st.subheader("Chat with the Mental Health Bot")
#     user_input = st.text_input("Enter your message:")
    
#     if user_input and st.button("Send"):
#         response = requests.post(f"{API_URL}/chatbot/", json={"user_message": user_input})
        
#         if response.status_code == 200:
#             result = response.json()
#             st.success(f"Bot: {result['response']}")
#         else:
#             st.error("Error getting chatbot response.")


# Frontend: Streamlit (app.py)

import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Mental Health Detection & Chatbot",
    layout="wide",  # 👈 This makes it full-screen
    initial_sidebar_state="expanded"
)

# FastAPI Backend URL
API_URL = "http://127.0.0.1:8000"

st.title("🧠 Mental Health Detection & Chatbot")

# Tabs for functionalities
tab1, tab2, tab3, tab4, tab5, tab7 = st.tabs([
    "🎤 Audio Emotion", "🖼️ Image Emotion", "💬 Chatbot", "📊 Mood Tracking", "📖 Mental Health Tips",  "🧘 Relaxation"
])

# 🎤 Audio Emotion Detection
with tab1:
    st.subheader("Upload an Audio File for Emotion Analysis")
    audio_file = st.file_uploader("Choose a WAV file", type=["wav"])
    
    if audio_file:
        st.audio(audio_file, format='audio/wav')
        
        if st.button("Analyze Audio"):
            files = {"file": audio_file.getvalue()}
            response = requests.post(f"{API_URL}/predict/audio/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Emotion Detected: {result['emotion']} (Confidence: {result['confidence']:.2f}%)")
            else:
                st.error("Error processing the audio.")

# 🖼️ Image Emotion Detection
with tab2:
    st.subheader("Upload an Image for Emotion Analysis")
    image_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
    
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image Preview", width=250)
        
        if st.button("Analyze Image"):
            files = {"file": image_file.getvalue()}
            response = requests.post(f"{API_URL}/predict/image/", files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Emotion Detected: {result['emotion']} (Confidence: {result['confidence']:.2f}%)")
            else:
                st.error("Error processing the image.")

# 💬 Chatbot
with tab3:
    st.subheader("Chat with the Mental Health Bot")
    user_input = st.text_input("Enter your message:")
    
    if user_input and st.button("Send"):
        response = requests.post(f"{API_URL}/chatbot/", json={"user_message": user_input})
        
        if response.status_code == 200:
            result = response.json()
            st.success(f"Bot: {result['response']}")
        else:
            st.error("Error getting chatbot response.")

# 📊 Mood Tracking
# 📊 Mood Tracking
# 📊 Mood Tracking
with tab4:
    st.subheader("📊 Mood Tracking & Analysis")

    # Select Mood
    mood = st.selectbox("How are you feeling today?", ["Happy", "Sad", "Anxious", "Stressed", "Neutral"])
    
    if st.button("Save Mood", key="save_mood"):
        response = requests.post(f"{API_URL}/track_mood/", json={"mood": mood})
        if response.status_code == 200:
            st.success(f"✅ Mood '{mood}' saved successfully!")
        else:
            st.error("❌ Error saving mood.")

    # Display Mood History
    st.subheader("📜 Mood History")
    history_response = requests.get(f"{API_URL}/get_mood_history/")
    if history_response.status_code == 200:
        mood_history = history_response.json()["mood_history"]
        if mood_history:
            st.write("📝 Previous moods: ", ", ".join(mood_history))

            # Mood Trend Visualization
            st.subheader("📈 Mood Trend")
            plt.figure(figsize=(8, 4))
            plt.plot(mood_history, marker='o', linestyle='-', color='blue')
            plt.xlabel("Mood Entries")
            plt.ylabel("Mood Type")
            plt.title("Mood Trends Over Time")
            st.pyplot(plt)
        else:
            st.write("No mood history available.")
    else:
        st.error("❌ Error fetching mood history.")

    # AI-Based Mood Suggestions
    st.subheader("💡 AI Mood-Based Suggestions")
    if st.button("Get Suggestions", key="mood_suggestions"):
        suggestion_response = requests.get(f"{API_URL}/mood_suggestions/")
        if suggestion_response.status_code == 200:
            st.write(suggestion_response.json()["suggestions"])
        else:
            st.error("❌ Error fetching suggestions.")



# 📖 Mental Health Tips
with tab5:
    st.subheader("AI-Based Mental Health Tips")
    if st.button("Get Tips"):
        response = requests.get(f"{API_URL}/mental_health_tips/")
        if response.status_code == 200:
            st.write(response.json()["tips"])
        else:
            st.error("Error fetching tips.")


# 🧘 Relaxation Suggestions
with tab7:
    st.subheader("🧘 Mindfulness & Relaxation Suggestions")
    
    if st.button("Get Suggestions", key="relaxation_suggestions"):
        response = requests.get(f"{API_URL}/relaxation_suggestions/")
        if response.status_code == 200:
            st.write(response.json()["suggestions"])
        else:
            st.error("❌ Error fetching relaxation suggestions.")
