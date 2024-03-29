import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Sign Language Detection")

# Load the pre-trained model
json_file = open("signlanguagedetectionmodel4.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("signlanguagedetectionmodel4.h5")

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.label = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'blank']

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Preprocessing steps
        cropframe = img[40:300, 0:300]
        cropframe = cv2.cvtColor(cropframe, cv2.COLOR_BGR2GRAY)
        cropframe = cv2.resize(cropframe, (48, 48))
        cropframe = cropframe.reshape(1, 48, 48, 1) / 255.0

        # Make prediction
        pred = model.predict(cropframe)
        prediction_label = self.label[np.argmax(pred)]
        accuracy = np.max(pred) * 100

        # Overlay text on the frame
        if prediction_label != 'blank':
            cv2.putText(img, f'{prediction_label}   {accuracy:.2f}%', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return img


webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if webrtc_ctx.video_processor:
    st.write("Webcam feed with Sign Language Detection")
    st.write("This might not work on mobile browsers due to the lack of WebRTC support.")
else:
    st.error("WebRTC is not available for this browser. Please try again using a different browser.")

