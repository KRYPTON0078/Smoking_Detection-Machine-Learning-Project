import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector

# Initialize detectors
handDetector = HandDetector()
meshDetector = FaceMeshDetector()
detector = FaceDetector()

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

            # Determine message and bounding box color
            if distance < 30:  # Smoking condition
                cvzone.putTextRect(frame, 'Smoking', (300, 110), scale=2, thickness=2, colorR=(0, 0, 255))  # Red text
                bbox_color = (0, 0, 255)  # Red bounding box
            else:  # No Smoking condition
                cvzone.putTextRect(frame, 'No Smoking', (300, 110), scale=2, thickness=2, colorR=(0, 255, 0))  # Green text
                bbox_color = (255, 0, 0)  # Blue bounding box
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
    cv2.putText(frame, 'Mejri Neder', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('neder', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
