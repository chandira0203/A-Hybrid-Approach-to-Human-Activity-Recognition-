import streamlit as st
import cv2
import numpy as np
import telepot
import requests
import geopy
from geopy.geocoders import Nominatim
from keras.models import load_model
from collections import deque
from datetime import datetime
import pytz
import tempfile
import os

# Load the trained model
MODEL_PATH = 'modelnew.h5'
model = load_model(MODEL_PATH)
TOKEN = "7759525166:AAFIMdCkeQv1JzV1_7Rh9637-FuPiqytKzo"  # Telegram Bot Token
CHAT_ID = "5272729527"  # Replace with your chat ID

def get_time():
    IST = pytz.timezone('Asia/Kolkata')
    return datetime.now(IST)

# Function to get user location automatically
def get_location():
    try:
        response = requests.get("https://ipinfo.io/json").json()
        loc = response.get("loc", "").split(",")  # Get latitude and longitude
        if len(loc) == 2:
            geolocator = Nominatim(user_agent="geoapi")
            location = geolocator.reverse(f"{loc[0]}, {loc[1]}")
            return location.address if location else "Unknown Location"
    except Exception as e:
        print("Error getting location:", e)
    return "Unknown Location"

def process_video(video_path):
    Q = deque(maxlen=128)
    vs = cv2.VideoCapture(video_path)
    true_count = 0
    send_alert = False
    filename = "savedImage.jpg"
    location = get_location()

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (128, 128)).astype("float32") / 255
        preds = model.predict(np.expand_dims(frame_resized, axis=0))[0]
        Q.append(preds)
        results = np.array(Q).mean(axis=0)
        label = (preds > 0.50)[0]
        
        if label:
            true_count += 1
            if true_count == 50 and not send_alert:
                cv2.imwrite(filename, frame)
                bot = telepot.Bot(TOKEN)
                bot.sendMessage(CHAT_ID, f"VIOLENCE ALERT!! \nLOCATION: {location} \nTIME: {get_time()}")
                bot.sendPhoto(CHAT_ID, photo=open(filename, 'rb'))
                send_alert = True
                
    vs.release()  # Release the video capture
    cv2.destroyAllWindows()  # Close all OpenCV windows
    return send_alert

# Streamlit UI
st.title(" A Hybrid Deep Learning Model for Violence Detection")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name  # Store the file path

    # Show the uploaded video
    st.video(temp_file_path)

    if st.button("Process Video"):
        alert_sent = process_video(temp_file_path)
        if alert_sent:
            st.error("Violence Detected! Alert sent.")
        else:
            st.success("No Violence Detected.")

    # Ensure file is closed before deleting
    os.remove(temp_file_path)  # Now it can be safely deleted
