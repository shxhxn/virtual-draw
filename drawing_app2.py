import cv2
import mediapipe as mp
import numpy as np

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

ret, frame = cap.read()
frame = cv2.flip(frame, 1)
h, w, _ = frame.shape

canvas = np.zeros((h, w, 3), dtype=np.uint8)

# --- Color Palette Definition ---
PALETTE_H = 80
SWATCH_W = 100

palette = [
    {"label": "Green",   "color": (0, 255, 0)},
    {"label": "Red",     "color": (0, 0, 255)},
    {"label": "Blue",    "color": (255, 100, 0)},
    {"label": "Yellow",  "color": (0, 255, 255)},
    {"label": "White",   "color": (255, 255, 255)},
    {"label": "Eraser",  "color": (0, 0, 0)},
]

# --- Drawing Settings ---
drawing_color = (0, 255, 0)
brush_size = 5
eraser_size = 30
prev_point = None
selected_index = 0

# --- Hover-to-select state ---
hover_index = -1
hover_frames = 0
HOVER_THRESHOLD = 20

# --- Two-palm clear state ---
clear_frames = 0
CLEAR_THRESHOLD = 30

print("✅ Running.")
print("   ☝️  Index finger = draw / select color")
print("   ✊  Fist = pause")
print("   🖐️🖐️  Two open palms = clear canvas")
print("   Q = quit")


def get_finger_states(lm):
    fingers = []
    fingers.append(lm[4].x < lm[3].x)
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers

def is_drawing_gesture(fingers):
    return fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]

def is_clear_gesture(fingers, lm):
    """
    Relaxed open palm check — just needs the 4 main fingers up.
    Thumb is ignored since it's unreliable from the front.
    """
    index_up  = lm[8].y  < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up   = lm[16].y < lm[14].y
    pinky_up  = lm[20].y < lm[18].y
    return index_up and middle_up and ring_up and pinky_up

def draw_palette(frame, selected_index, hover_index, hover_frames):
    for i, swatch in enumerate(palette):
        x1 = i * SWATCH_W
        x2 = x1 + SWATCH_W

        cv2.rectangle(frame, (x1, 0), (x2, PALETTE_H), swatch["color"], cv2.FILLED)
        cv2.rectangle(frame, (x1, 0), (x2, PALETTE_H), (50, 50, 50), 2)

        if i == selected_index:
            cv2.rectangle(frame, (x1, 0), (x2, PALETTE_H), (255, 255, 255), 4)
            cv2.putText(frame, "v", (x1 + 38, PALETTE_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        if i == hover_index and hover_frames > 0:
            progress = int(SWATCH_W * hover_frames / HOVER_THRESHOLD)
            cv2.rectangle(frame, (x1, PALETTE_H - 8), (x1 + progress, PALETTE_H),
                          (255, 255, 255), cv2.FILLED)

        label_color = (0, 0, 0) if swatch["color"] != (0, 0, 0) else (200, 200, 200)
        cv2.putText(frame, swatch["label"], (x1 + 8, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1)

    return frame

def get_swatch_at(cx, cy):
    if cy < PALETTE_H:
        i = cx // SWATCH_W
        if 0 <= i < len(palette):
            return i
    return -1


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    h, w, _ = frame.shape
    gesture_label = ""
    hand_count = 0

    if result.multi_hand_landmarks:
        num_hands = len(result.multi_hand_landmarks)
        hand_count = num_hands

        # --- Check if both hands are open palms ---
        all_open_palms = False
        palm_statuses = []  # For debug display

        if num_hands == 2:
            palm_checks = []
            for hand_landmarks in result.multi_hand_landmarks:
                lm = hand_landmarks.landmark
                f = get_finger_states(lm)
                result_palm = is_clear_gesture(f, lm)
                palm_checks.append(result_palm)
                palm_statuses.append("OPEN" if result_palm else "CLOSED")
            all_open_palms = all(palm_checks)

        if all_open_palms:
            # --- TWO OPEN PALMS = CLEAR GESTURE ---
            clear_frames += 1
            prev_point = None
            hover_index = -1
            hover_frames = 0

            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark
                cx = int(lm[8].x * w)
                cy = int(lm[8].y * h)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 255), cv2.FILLED)

            if clear_frames >= CLEAR_THRESHOLD:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                clear_frames = 0
                gesture_label = "CLEARED!"
            else:
                remaining = CLEAR_THRESHOLD - clear_frames
                gesture_label = f"CLEARING IN {remaining}..."

        else:
            # --- Single hand drawing logic ---
            clear_frames = 0

            # Show palm debug if 2 hands detected but not both open
            if num_hands == 2 and palm_statuses:
                debug_text = f"Palms: {palm_statuses[0]} / {palm_statuses[1]}"
                cv2.putText(frame, debug_text, (10, h - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

            hand_landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            lm = hand_landmarks.landmark
            fingers = get_finger_states(lm)

            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)

            if is_drawing_gesture(fingers):
                swatch_i = get_swatch_at(cx, cy)

                if swatch_i >= 0:
                    gesture_label = f"SELECTING: {palette[swatch_i]['label']}"
                    prev_point = None

                    if swatch_i == hover_index:
                        hover_frames += 1
                    else:
                        hover_index = swatch_i
                        hover_frames = 0

                    if hover_frames >= HOVER_THRESHOLD:
                        selected_index = swatch_i
                        drawing_color = palette[swatch_i]["color"]
                        brush_size = eraser_size if swatch_i == len(palette) - 1 else 5
                        hover_frames = 0
                        gesture_label = f"SELECTED: {palette[swatch_i]['label']}"

                else:
                    hover_index = -1
                    hover_frames = 0
                    gesture_label = "DRAWING"
                    cv2.circle(frame, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

                    if cy > PALETTE_H:
                        if prev_point is not None and prev_point[1] > PALETTE_H:
                            cv2.line(canvas, prev_point, (cx, cy), drawing_color, brush_size)
                        prev_point = (cx, cy)
                    else:
                        prev_point = None

            else:
                gesture_label = "PAUSED"
                prev_point = None
                hover_index = -1
                hover_frames = 0
                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

    else:
        prev_point = None
        hover_index = -1
        hover_frames = 0
        clear_frames = 0

    # --- Blend canvas onto frame ---
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, canvas_fg)

    # --- Draw palette on top ---
    combined = draw_palette(combined, selected_index, hover_index, hover_frames)

    # --- UI Text ---
    cv2.putText(combined, f"Hands: {hand_count} | Gesture: {gesture_label}",
                (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(combined, "Two open palms = Clear | Q = Quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # --- Current color indicator ---
    cv2.circle(combined, (w - 40, h - 40), 20, drawing_color, cv2.FILLED)
    cv2.circle(combined, (w - 40, h - 40), 20, (255, 255, 255), 2)

    cv2.imshow("AR Drawing App", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
