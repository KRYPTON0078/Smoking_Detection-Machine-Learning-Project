import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
from tensorflow.keras.models import load_model
import numpy as np
import pyttsx3
import threading
from queue import Queue

# Load the trained smoke detection model
model = load_model('smoke_detection_model.keras')

# Initialize detectors
handDetector = HandDetector()
meshDetector = FaceMeshDetector()
detector = FaceDetector()

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Define warning functions
def tts_worker():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

# Initialize the message queue and background thread
tts_queue = Queue()
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Initialize state variables to manage vocal alerts
prev_state = None

# Load video
video = cv2.VideoCapture(0)

pointX11 = pointY11 = 0  # Initialize hand landmark variables

while True:
    ret, frame = video.read()
    if not ret:
        break  # Exit loop if video ends

    # Detect face and get bounding boxes
    frame, bBoxes = detector.findFaces(frame, draw=False)

    # Perform face mesh detection
    frame, faces = meshDetector.findFaceMesh(frame, draw=False)

    # Perform hand detection
    hands, frame = handDetector.findHands(frame, draw=True)

    if hands:  # If hands are detected
        for hand in hands:
            if hand["type"] == "Right":  # Use only the right hand
                landmarks = hand["lmList"]
                pointX11, pointY11 = landmarks[11][0], landmarks[11][1]

    if faces:  # If face mesh is detected
        face_point = faces[0][14]  # Get the specific landmark point (nose tip)
        if pointX11 and pointY11:  # Ensure hand landmark is detected
            # Calculate distance between hand and face
            distance = meshDetector.findDistance((pointX11, pointY11), (face_point[0], face_point[1]))[0]
            print(distance)

            # Preprocess the frame for smoke detection
            img = cv2.resize(frame, (64, 64))
            img = np.expand_dims(img, axis=0)
            img = img / 255.0

            # Predict smoke presence
            prediction = model.predict(img)[0][0]
            if distance < 30 and prediction > 0.5:  # Smoking condition
                cvzone.putTextRect(frame, 'Smoking', (300, 110), scale=2, thickness=2, colorR=(0, 0, 255))  # Red text
                bbox_color = (0, 0, 255)  # Red bounding box
                if prev_state != "smoking":
                    tts_queue.put("Warning: Smoking detected.")
                    prev_state = "smoking"
            else:  # No Smoking condition
                cvzone.putTextRect(frame, 'No Smoking', (300, 110), scale=2, thickness=2, colorR=(0, 255, 0))  # Green text
                bbox_color = (255, 0, 0)  # Blue bounding box
                if prev_state != "no_smoking":
                    tts_queue.put("No smoking detected.")
                    prev_state = "no_smoking"
        else:
            bbox_color = (255, 0, 0)  # Default color is blue if no hand is detected
    else:
        bbox_color = (255, 0, 0)  # Default color is blue if no face is detected

    # Draw face bounding boxes with appropriate color
    if bBoxes:
        for bBox in bBoxes:
            x, y, w, h = bBox["bbox"]  # Extract bounding box coordinates
            cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 3)

    # Add your name to the frame
    cv2.putText(frame, 'Magne', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Magne', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()

# Signal the background thread to exit and wait for it to finish
tts_queue.put(None)
tts_thread.join()
