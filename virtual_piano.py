import cv2
import mediapipe as mp
import pygame
import numpy as np

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize Pygame for sound
pygame.mixer.init()

# Load piano sounds for each finger
sounds = [
    pygame.mixer.Sound("assets/sounds/a1.wav"),  # Left Thumb
    pygame.mixer.Sound("assets/sounds/d1.wav"),  # Left Index
    pygame.mixer.Sound("assets/sounds/e1.wav"),  # Left Middle
    pygame.mixer.Sound("assets/sounds/f1.wav"),  # Left Ring
    pygame.mixer.Sound("assets/sounds/g1.wav"),  # Left Pinky
    pygame.mixer.Sound("assets/sounds/a1.wav"),  # Right Thumb
    pygame.mixer.Sound("assets/sounds/b1.wav"),  # Right Index
    pygame.mixer.Sound("assets/sounds/c2.wav"),  # Right Middle
    pygame.mixer.Sound("assets/sounds/d2.wav"),  # Right Ring
    pygame.mixer.Sound("assets/sounds/e2.wav")   # Right Pinky
]

# Capture video from webcam
cap = cv2.VideoCapture(r'assets\video.mp4')

# Initialize finger states to track raised fingers
finger_states = [False] * 10  # False = not raised, True = raised

def detect_and_play(hand_landmarks, hand_label, img):
    h, w, _ = img.shape

    # Map finger indices to their positions and sounds
    if hand_label == "Left":
        finger_indices = [4, 8, 12, 16, 20]  # Thumb to Pinky
        sound_offset = 0  # Use sounds[0-4]
    else:
        finger_indices = [4, 8, 12, 16, 20]  # Thumb to Pinky
        sound_offset = 5  # Use sounds[5-9]

    for i, finger_index in enumerate(finger_indices):
        tip = hand_landmarks.landmark[finger_index]
        mcp = hand_landmarks.landmark[finger_index - 3]

        # Convert to pixel coordinates for visualization
        tip_x, tip_y = int(tip.x * w), int(tip.y * h)
        mcp_x, mcp_y = int(mcp.x * w), int(mcp.y * h)

        # Check if fingertip is above MCP joint
        is_raised = tip.y < mcp.y

        # Play sound only if state changes to "raised"
        global finger_states
        finger_index_global = i + sound_offset
        if is_raised and not finger_states[finger_index_global]:
            sounds[finger_index_global].play()
            finger_states[finger_index_global] = True
        elif not is_raised:
            finger_states[finger_index_global] = False

        # Draw circles on the detected landmarks
        circle_radius = 20 if is_raised else 5
        cv2.circle(img, (tip_x, tip_y), circle_radius, (0, 255, 0), -1)

        # Add feedback text for debugging or visualization
        cv2.putText(img, f"Key: {chr(67 + i + sound_offset)}", (tip_x, tip_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural interaction
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        hand_positions = []  # Store hand positions for re-labeling

        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get x-coordinate of the wrist
            wrist_x = hand_landmarks.landmark[0].x
            hand_positions.append((wrist_x, hand_index))

        # Sort hands by x-coordinate (leftmost first)
        hand_positions.sort(key=lambda x: x[0])

        # Reassign hand labels based on x-coordinate
        for new_index, (wrist_x, hand_index) in enumerate(hand_positions):
            hand_landmarks = results.multi_hand_landmarks[hand_index]
            hand_label = "Left" if new_index == 0 else "Right"  # Re-label based on position
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            detect_and_play(hand_landmarks, hand_label, frame)

    # Show the frame
    cv2.imshow("Virtual Piano", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()
pygame.mixer.quit()
