# from fastapi import FastAPI, File, UploadFile
# import tensorflow as tf
# import numpy as np
# import uvicorn
# from io import BytesIO
# import librosa

# app = FastAPI()

# # Define emotion labels (based on your dataset)
# emotion_labels = [
#     "Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"
# ]

# # Load the model
# try:
#     model = tf.keras.models.load_model("model.h5")
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Model loading failed: {e}")

# def preprocess_audio(audio_bytes):
#     """Extract MFCC features, normalize, and reshape for model input"""
#     try:
#         print("🔄 Processing audio...")
#         y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)

#         # Extract 40 MFCCs
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         mfcc_delta = librosa.feature.delta(mfcc)  # First derivative (dynamics)
#         mfcc_delta2 = librosa.feature.delta(mfcc, order=2)  # Second derivative

#         # Stack features together
#         mfccs = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

#         # Normalize features
#         mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

#         # Compute mean to get a fixed shape
#         mfccs = np.mean(mfccs, axis=1).reshape(1, -1)

#         print("✅ Audio processed successfully!")
#         return mfccs
#     except Exception as e:
#         print(f"❌ Error in audio preprocessing: {e}")
#         return None

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         print("🔄 Received file:", file.filename)
#         audio_bytes = await file.read()

#         processed_audio = preprocess_audio(audio_bytes)
#         if processed_audio is None:
#             return {"error": "Audio processing failed"}

#         print("🔄 Making prediction...")
#         prediction = model.predict(processed_audio)

#         # Get the predicted emotion
#         predicted_label = emotion_labels[np.argmax(prediction)]

#         print(f"✅ Prediction: {predicted_label}")

#         return {"emotion": predicted_label}
#     except Exception as e:
#         print(f"❌ Error in prediction: {e}")
#         return {"error": str(e)}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# from fastapi import FastAPI, File, UploadFile
# import tensorflow as tf
# import numpy as np
# import uvicorn
# from io import BytesIO
# import librosa
# import cv2

# app = FastAPI()

# # Define emotion labels
# emotion_labels = [
#     "Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"
# ]

# # Load models
# try:
#     audio_model = tf.keras.models.load_model("model.h5")
#     image_model = tf.keras.models.load_model("cnn_emotion_model.h5")
#     print("✅ Models loaded successfully!")
# except Exception as e:
#     print(f"❌ Model loading failed: {e}")

# def preprocess_audio(audio_bytes):
#     try:
#         print("🔄 Processing audio...")
#         y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         mfcc_delta = librosa.feature.delta(mfcc)
#         mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
#         mfccs = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
#         mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)
#         mfccs = np.mean(mfccs, axis=1).reshape(1, -1)
#         print("✅ Audio processed successfully!")
#         return mfccs
#     except Exception as e:
#         print(f"❌ Error in audio preprocessing: {e}")
#         return None

# def preprocess_image(image_bytes):
#     try:
#         print("🔄 Processing image...")
#         img_array = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (48, 48)) / 255.0
#         img = img.reshape(1, 48, 48, 1)
#         print("✅ Image processed successfully!")
#         return img
#     except Exception as e:
#         print(f"❌ Error in image preprocessing: {e}")
#         return None

# @app.post("/predict/audio/")
# async def predict_audio(file: UploadFile = File(...)):
#     try:
#         audio_bytes = await file.read()
#         processed_audio = preprocess_audio(audio_bytes)
#         if processed_audio is None:
#             return {"error": "Audio processing failed"}
#         prediction = audio_model.predict(processed_audio)
#         predicted_label = emotion_labels[np.argmax(prediction)]
#         return {"emotion": predicted_label}
#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/predict/image/")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         processed_image = preprocess_image(image_bytes)
#         if processed_image is None:
#             return {"error": "Image processing failed"}
#         prediction = image_model.predict(processed_image)
#         predicted_label = emotion_labels[np.argmax(prediction)]
#         return {"emotion": predicted_label}
#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# from fastapi import FastAPI, File, UploadFile
# import tensorflow as tf
# import numpy as np
# import uvicorn
# from io import BytesIO
# import librosa
# import cv2

# app = FastAPI()

# # Define emotion labels
# emotion_labels = [
#     "Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"
# ]

# # Load models
# try:
#     audio_model = tf.keras.models.load_model("model.h5")
#     image_model = tf.keras.models.load_model("cnn_emotion_model.h5")
#     print("✅ Models loaded successfully!")
# except Exception as e:
#     print(f"❌ Model loading failed: {e}")

# def preprocess_audio(audio_bytes):
#     try:
#         print("🔄 Processing audio...")
#         y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)

#         # Trim silence
#         y, _ = librosa.effects.trim(y)

#         # Ensure fixed length (e.g., 3 seconds)
#         max_length = sr * 3  # 3 seconds
#         if len(y) < max_length:
#             y = np.pad(y, (0, max_length - len(y)))
#         else:
#             y = y[:max_length]

#         # Extract Log-MFCCs
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         mfcc = librosa.power_to_db(mfcc)  # Convert to log scale

#         # Extract Other Features
#         rms = librosa.feature.rms(y=y)
#         zcr = librosa.feature.zero_crossing_rate(y)

#         # Stack Features Together
#         features = np.vstack([mfcc, rms, zcr])

#         # Normalize
#         features = (features - np.min(features)) / (np.max(features) - np.min(features))

#         # Compute mean for fixed shape
#         features = np.mean(features, axis=1).reshape(1, -1)

#         print("✅ Audio processed successfully!")
#         return features
#     except Exception as e:
#         print(f"❌ Error in audio preprocessing: {e}")
#         return None


# def preprocess_image(image_bytes):
#     try:
#         print("🔄 Processing image...")
#         img_array = np.frombuffer(image_bytes, np.uint8)
#         img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (48, 48)) / 255.0
#         img = img.reshape(1, 48, 48, 1)
#         print("✅ Image processed successfully!")
#         return img
#     except Exception as e:
#         print(f"❌ Error in image preprocessing: {e}")
#         return None

# @app.post("/predict/audio/")
# async def predict_audio(file: UploadFile = File(...)):
#     try:
#         audio_bytes = await file.read()
#         processed_audio = preprocess_audio(audio_bytes)
#         if processed_audio is None:
#             return {"error": "Audio processing failed"}

#         print("🔄 Making prediction...")
#         prediction = audio_model.predict(processed_audio)
#         print("Raw Prediction:", prediction)  # Debugging

#         predicted_label = emotion_labels[np.argmax(prediction)]
#         return {"emotion": predicted_label}
#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/predict/image/")
# async def predict_image(file: UploadFile = File(...)):
#     try:
#         image_bytes = await file.read()
#         processed_image = preprocess_image(image_bytes)
#         if processed_image is None:
#             return {"error": "Image processing failed"}

#         print("🔄 Making prediction...")
#         prediction = image_model.predict(processed_image)
#         print("Raw Prediction:", prediction)  # Debugging

#         predicted_label = emotion_labels[np.argmax(prediction)]
#         return {"emotion": predicted_label}
#     except Exception as e:
#         return {"error": str(e)}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# AIzaSyDdifJhrztNdBYGKGWM1xDtQr3vP2GSTds



from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import uvicorn
import requests
import librosa
import cv2
import base64
from io import BytesIO

# 🎯 Define constants
GEMINI_API_KEY = "AIzaSyDdifJhrztNdBYGKGWM1xDtQr3vP2GSTds"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

app = FastAPI()

# Define emotion labels
emotion_labels = [
    "Anxiety", 
    "Depression", 
    "Stress", 
    "Well-being", 
    "Resilience", 
    "Burnout", 
    "Mindfulness"
]

# Load TensorFlow Models
try:
    audio_model = tf.keras.models.load_model("model.h5")
    image_model = tf.keras.models.load_model("cnn_emotion_model.h5")
    print("✅ Models Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Models: {e}")

# 🎤 Function to preprocess audio
def preprocess_audio(audio_bytes):
    try:
        print("🔄 Processing audio...")
        y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
        y, _ = librosa.effects.trim(y)  # Trim silence

        # Ensure fixed length (7 sec)
        max_length = sr * 7
        if len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)))
        else:
            y = y[:max_length]

        # Extract features (MFCC + RMS + ZCR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        features = np.vstack([mfcc, rms, zcr])

        # Normalize
        features = (features - np.min(features)) / (np.max(features) - np.min(features))
        features = np.mean(features, axis=1).reshape(1, -1)

        print("✅ Audio processed successfully!")
        return features
    except Exception as e:
        print(f"❌ Error in audio preprocessing: {e}")
        return None

# 🖼️ Function to preprocess image
def preprocess_image(image_bytes):
    try:
        print("🔄 Processing image...")
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48)) / 255.0
        img = img.reshape(1, 48, 48, 1)

        print("✅ Image processed successfully!")
        return img
    except Exception as e:
        print(f"❌ Error in image preprocessing: {e}")
        return None

# 🔮 Function to get emotion prediction from Gemini API
def get_gemini_emotion(audio_bytes):
    try:
        print("🔄 Sending audio data to Gemini API...")

        # Convert audio bytes to Base64
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        # Prepare the request payload (explicitly marking as audio input)
        request_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}}
                    ]
                }
            ]
        }

        # Send request to Gemini API
        response = requests.post(GEMINI_API_URL, json=request_payload)

        # Check response
        if response.status_code == 200:
            response_data = response.json()
            emotion = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Unknown")
            print(f"✅ Gemini API Response: {emotion}")
            return emotion
        else:
            print(f"❌ Gemini API Error: {response.text}")
            return "Unknown"

    except Exception as e:
        print(f"❌ Gemini API Request Error: {e}")
        return "Unknown"

# 🎤 API Endpoint for Audio Prediction
@app.post("/predict/audio/")
async def predict_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()

        # 1️⃣ Predict using ML Model
        processed_audio = preprocess_audio(audio_bytes)
        if processed_audio is None:
            return {"error": "Audio processing failed"}

        model_prediction = audio_model.predict(processed_audio)
        model_emotion = emotion_labels[np.argmax(model_prediction)]

        # 2️⃣ Predict using Gemini API
        gemini_emotion = get_gemini_emotion(audio_bytes)

        # 3️⃣ Combine Results
        if model_emotion.lower() == gemini_emotion.lower():
            final_emotion = model_emotion
        else:
            final_emotion = f"Model: {model_emotion}, Gemini: {gemini_emotion}"

        return {"emotion": final_emotion}
    except Exception as e:
        return {"error": str(e)}

# 🖼️ API Endpoint for Image Prediction
@app.post("/predict/image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # Process image
        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            return {"error": "Image processing failed"}

        print("🔄 Making prediction...")
        prediction = image_model.predict(processed_image)
        predicted_label = emotion_labels[np.argmax(prediction)]

        return {"emotion": predicted_label}
    except Exception as e:
        return {"error": str(e)}

# 🚀 Run FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
