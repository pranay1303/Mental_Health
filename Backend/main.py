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



# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel
# import tensorflow as tf
# import numpy as np
# import uvicorn
# import requests
# import librosa
# import cv2
# import base64
# from io import BytesIO

# # 🎯 Define Constants
# GEMINI_API_KEY = "AIzaSyDdifJhrztNdBYGKGWM1xDtQr3vP2GSTds"
# GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

# app = FastAPI()

# # Define emotion labels
# emotion_labels = [
#     "Anxiety", "Depression", "Stress", "Well-being", "Resilience", "Burnout", "Mindfulness"
# ]

# # Load TensorFlow Models
# try:
#     audio_model = tf.keras.models.load_model("model.h5")
#     image_model = tf.keras.models.load_model("cnn_emotion_model.h5")
#     print("✅ Models Loaded Successfully!")
# except Exception as e:
#     print(f"❌ Error Loading Models: {e}")

# # 📥 Request Model for Chatbot
# class ChatRequest(BaseModel):
#     user_message: str

# # 🎤 Function to preprocess audio
# def preprocess_audio(audio_bytes):
#     try:
#         print("🔄 Processing audio...")
#         y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
#         y, _ = librosa.effects.trim(y)

#         # Ensure fixed length (7 sec)
#         max_length = sr * 7
#         y = np.pad(y, (0, max_length - len(y))) if len(y) < max_length else y[:max_length]

#         # Extract features (MFCC + RMS + ZCR)
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         rms = librosa.feature.rms(y=y)
#         zcr = librosa.feature.zero_crossing_rate(y)
#         features = np.vstack([mfcc, rms, zcr])

#         # Normalize
#         features = (features - np.min(features)) / (np.max(features) - np.min(features))
#         features = np.mean(features, axis=1).reshape(1, -1)

#         print("✅ Audio processed successfully!")
#         return features
#     except Exception as e:
#         print(f"❌ Error in audio preprocessing: {e}")
#         return None

# # 🖼️ Function to preprocess image
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

# # 🔮 Function to get emotion prediction from Gemini API for audio
# def get_gemini_emotion(audio_bytes):
#     try:
#         print("🔄 Sending audio data to Gemini API...")
#         audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
#         request_payload = {
#             "contents": [
#                 {
#                     "role": "user",
#                     "parts": [
#                         {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
#                         {"text": "Analyze the mental health state in this audio and return only a 1-2 word label."}
#                     ]
#                 }
#             ]
#         }
#         response = requests.post(GEMINI_API_URL, json=request_payload)
#         if response.status_code == 200:
#             response_data = response.json()
#             emotion = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Unknown")
#             print(f"✅ Gemini API Response: {emotion}")
#             return emotion
#         else:
#             print(f"❌ Gemini API Error: {response.text}")
#             return "Unknown"
#     except Exception as e:
#         print(f"❌ Gemini API Request Error: {e}")
#         return "Unknown"

# # 🤖 Chatbot Function
# def get_gemini_chat_response(user_message):
#     try:
#         print(f"📩 Received chatbot message: {user_message}")
        
#         request_payload = {"contents": [{"role": "user", "parts": [{"text": user_message}]}]}
#         response = requests.post(GEMINI_API_URL, json=request_payload)

#         if response.status_code == 200:
#             response_data = response.json()
#             chat_response = response_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't understand that.")
#             print(f"✅ Gemini Chatbot Response: {chat_response}")
#             return chat_response
#         else:
#             print(f"❌ Gemini API Error: {response.text}")
#             return "Error fetching response"
#     except Exception as e:
#         print(f"❌ Chatbot API Request Error: {e}")
#         return "Error in chatbot processing"

# # 🚀 API Endpoints
# @app.post("/predict/audio/")
# async def predict_audio(file: UploadFile = File(...)):
#     try:
#         audio_bytes = await file.read()
#         processed_audio = preprocess_audio(audio_bytes)
#         if processed_audio is None:
#             return {"error": "Audio processing failed"}

#         model_prediction = audio_model.predict(processed_audio)
#         confidence = float(np.max(model_prediction)) * 100
#         model_emotion = emotion_labels[np.argmax(model_prediction)]

#         gemini_emotion = get_gemini_emotion(audio_bytes)

#         final_emotion = model_emotion if model_emotion.lower() == gemini_emotion.lower() else f"Model: {model_emotion}, Gemini: {gemini_emotion}"
        
#         return {"emotion": final_emotion, "confidence": confidence}
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
#         confidence = float(np.max(prediction)) * 100
#         predicted_label = emotion_labels[np.argmax(prediction)]

#         return {"emotion": predicted_label, "confidence": confidence}
#     except Exception as e:
#         return {"error": str(e)}

# @app.post("/chatbot/")
# async def chatbot_endpoint(chat_request: ChatRequest):
#     print("🔄 Processing chatbot request...")
#     response = get_gemini_chat_response(chat_request.user_message)
#     return {"response": response}

# mood_history = []

# class MoodRequest(BaseModel):
#     mood: str

# @app.post("/track_mood/")
# async def track_mood(mood_request: MoodRequest):
#     mood_history.append(mood_request.mood)
#     print(f"✅ Mood Tracked: {mood_request.mood}")
#     return {"status": "success", "mood": mood_request.mood}

# @app.get("/get_mood_history/")
# async def get_mood_history():
#     return {"mood_history": mood_history}

# @app.get("/mood_suggestions/")
# async def mood_suggestions():
#     if not mood_history:
#         return {"suggestions": "No mood data available."}

#     latest_mood = mood_history[-1]
#     request_payload = {
#         "contents": [{"role": "user", "parts": [{"text": f"Suggest mental health advice for someone feeling {latest_mood}."}]}]
#     }
#     response = requests.post(GEMINI_API_URL, json=request_payload)
    
#     if response.status_code == 200:
#         suggestion = response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No suggestions available.")
#         return {"suggestions": suggestion}
#     else:
#         return {"suggestions": "Error fetching suggestions"}

# @app.get("/mental_health_tips/")
# async def get_tips():
#     response = requests.post(GEMINI_API_URL, json={"contents": [{"role": "user", "parts": [{"text": "Give me a mental health tip."}]}]})
#     return {"tips": response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No tips available.")}


# @app.get("/relaxation_suggestions/")
# async def relaxation_suggestions():
#     response = requests.post(GEMINI_API_URL, json={"contents": [{"role": "user", "parts": [{"text": "Give me relaxation suggestions."}]}]})
#     return {"suggestions": response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No suggestions available.")}


# if __name__ == "__main__":
#     print("🚀 Starting FastAPI Server...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import uvicorn
import requests
import librosa
import cv2
import base64
from io import BytesIO

# 🎯 Define Constants
GEMINI_API_KEY = "AIzaSyDdifJhrztNdBYGKGWM1xDtQr3vP2GSTds"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

app = FastAPI()

# Emotion labels
emotion_labels = [
    "Anxiety", "Depression", "Stress", "Well-being", "Resilience", "Burnout", "Mindfulness"
]

# Load models
try:
    audio_model = tf.keras.models.load_model("model.h5")
    image_model = tf.keras.models.load_model("cnn_emotion_model.h5")
    print("✅ Models Loaded Successfully!")
except Exception as e:
    print(f"❌ Error Loading Models: {e}")

# Global variable to store latest detected emotion
latest_emotion = None

# 📥 Chat request model
class ChatRequest(BaseModel):
    user_message: str

# 🎤 Preprocess audio
def preprocess_audio(audio_bytes):
    try:
        print("🔄 Processing audio...")
        y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
        y, _ = librosa.effects.trim(y)
        max_length = sr * 7
        y = np.pad(y, (0, max_length - len(y))) if len(y) < max_length else y[:max_length]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        rms = librosa.feature.rms(y=y)
        zcr = librosa.feature.zero_crossing_rate(y)
        features = np.vstack([mfcc, rms, zcr])
        features = (features - np.min(features)) / (np.max(features) - np.min(features))
        features = np.mean(features, axis=1).reshape(1, -1)
        return features
    except Exception as e:
        print(f"❌ Error in audio preprocessing: {e}")
        return None

# 🖼️ Preprocess image
def preprocess_image(image_bytes):
    try:
        print("🔄 Processing image...")
        img_array = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48)) / 255.0
        img = img.reshape(1, 48, 48, 1)
        return img
    except Exception as e:
        print(f"❌ Error in image preprocessing: {e}")
        return None

# 🔮 Gemini prediction for audio
def get_gemini_emotion(audio_bytes):
    try:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        request_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}},
                        {"text": "Analyze the mental health state in this audio and return only a 1-2 word label."}
                    ]
                }
            ]
        }
        response = requests.post(GEMINI_API_URL, json=request_payload)
        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Unknown")
        return "Unknown"
    except:
        return "Unknown"

# 🤖 Gemini Chat
def get_gemini_chat_response(user_message):
    try:
        payload = {"contents": [{"role": "user", "parts": [{"text": user_message}]}]}
        response = requests.post(GEMINI_API_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Sorry, I couldn't understand that.")
        return "Error fetching response"
    except:
        return "Error in chatbot processing"

# 🚀 API Endpoints
@app.post("/predict/audio/")
async def predict_audio(file: UploadFile = File(...)):
    global latest_emotion
    audio_bytes = await file.read()
    processed_audio = preprocess_audio(audio_bytes)
    if processed_audio is None:
        return {"error": "Audio processing failed"}
    model_prediction = audio_model.predict(processed_audio)
    confidence = float(np.max(model_prediction)) * 100
    model_emotion = emotion_labels[np.argmax(model_prediction)]
    gemini_emotion = get_gemini_emotion(audio_bytes)
    final_emotion = model_emotion if model_emotion.lower() == gemini_emotion.lower() else f"Model: {model_emotion}, Gemini: {gemini_emotion}"
    latest_emotion = gemini_emotion  # Save latest detected emotion
    return {"emotion": final_emotion, "confidence": confidence}

@app.post("/predict/image/")
async def predict_image(file: UploadFile = File(...)):
    global latest_emotion
    image_bytes = await file.read()
    processed_image = preprocess_image(image_bytes)
    if processed_image is None:
        return {"error": "Image processing failed"}
    prediction = image_model.predict(processed_image)
    confidence = float(np.max(prediction)) * 100
    predicted_label = emotion_labels[np.argmax(prediction)]
    latest_emotion = predicted_label  # Save latest detected emotion
    return {"emotion": predicted_label, "confidence": confidence}

@app.post("/chatbot/")
async def chatbot_endpoint(chat_request: ChatRequest):
    response = get_gemini_chat_response(chat_request.user_message)
    return {"response": response}

mood_history = []

class MoodRequest(BaseModel):
    mood: str

@app.post("/track_mood/")
async def track_mood(mood_request: MoodRequest):
    mood_history.append(mood_request.mood)
    return {"status": "success", "mood": mood_request.mood}

@app.get("/get_mood_history/")
async def get_mood_history():
    return {"mood_history": mood_history}

@app.get("/mood_suggestions/")
async def mood_suggestions():
    if not mood_history:
        return {"suggestions": "No mood data available."}
    latest_mood = mood_history[-1]
    payload = {"contents": [{"role": "user", "parts": [{"text": f"Suggest mental health advice for someone feeling {latest_mood}."}]}]}
    response = requests.post(GEMINI_API_URL, json=payload)
    if response.status_code == 200:
        return {"suggestions": response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No suggestions available.")}
    return {"suggestions": "Error fetching suggestions"}

@app.get("/mental_health_tips/")
async def get_tips():
    if not latest_emotion:
        return {"tips": "No emotion detected yet."}
    prompt = f"Give me a mental health tip for someone experiencing {latest_emotion}."
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_API_URL, json=payload)
    if response.status_code == 200:
        return {"tips": response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No tips available.")}
    return {"tips": "Error fetching tips."}

@app.get("/relaxation_suggestions/")
async def relaxation_suggestions():
    if not latest_emotion:
        return {"suggestions": "No emotion detected yet."}
    prompt = f"Suggest some relaxation techniques for someone feeling {latest_emotion}."
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    response = requests.post(GEMINI_API_URL, json=payload)
    if response.status_code == 200:
        return {"suggestions": response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "No suggestions available.")}
    return {"suggestions": "Error fetching suggestions."}

if __name__ == "__main__":
    print("🚀 Starting FastAPI Server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
