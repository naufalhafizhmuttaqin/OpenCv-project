import cv2
import mediapipe as mp

# Initialize MediaPipe drawing and hand detection modules
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Open webcam
cap = cv2.VideoCapture(0)

# Configure MediaPipe Hands
with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, image = cap.read()
        if not success:
            break

        # Mirror the camera feed for natural interaction
        image = cv2.flip(image, 1)

        # Process the image to detect hands
        results = hands.process(image)

        # Draw hand landmarks if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        # Display the output
        cv2.imshow('Hand Gesture', image)

        # Press ESC to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
