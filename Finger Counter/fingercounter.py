import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and drawing utility
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Configure MediaPipe Hands
with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        # Mirror the camera feed
        img = cv2.flip(img, 1)

        # Convert image to RGB for MediaPipe processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        total_fingers = 0
        tip_ids = [4, 8, 12, 16, 20]  # Thumb to pinky fingertip landmark IDs

        # Process detected hands
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):

                # Get hand label (Left / Right)
                hand_label = results.multi_handedness[hand_idx].classification[0].label

                # Draw hand landmarks and connections
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = img.shape
                lm_list = []

                # Store landmark positions in pixel coordinates
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))

                fingers_up = 0

                if lm_list:
                    # Thumb detection (different logic for left and right hand)
                    if hand_label == "Right":
                        if lm_list[tip_ids[0]][1] > lm_list[tip_ids[0] - 1][1]:
                            fingers_up += 1
                    else:
                        if lm_list[tip_ids[0]][1] < lm_list[tip_ids[0] - 1][1]:
                            fingers_up += 1

                    # Other fingers detection (tip above PIP joint)
                    for id in range(1, 5):
                        if lm_list[tip_ids[id]][2] < lm_list[tip_ids[id] - 2][2]:
                            fingers_up += 1

                total_fingers += fingers_up

        # Display total finger count
        cv2.putText(img, f"Total Fingers: {total_fingers}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("Hand Counter Detection", img)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
