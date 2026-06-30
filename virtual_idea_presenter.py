import cv2
import mediapipe as mp
import numpy as np
import math

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

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

# ─────────────────────────────────────────────
#  DESIGN TOKENS
# ─────────────────────────────────────────────
PALETTE_H       = 72          # Top bar height
SWATCH_R        = 22          # Radius of each palette circle
SWATCH_SPACING  = 64          # Center-to-center spacing
SWATCH_Y        = PALETTE_H // 2
SWATCH_X_START  = 48

WHEEL_CENTER    = (w - 90, h // 2)   # Color wheel lives on the right edge
WHEEL_R         = 70                  # Outer radius of wheel
WHEEL_INNER_R   = 28                  # Inner hole radius

HOVER_THRESHOLD = 12    # ~0.4s at 30fps
CLEAR_THRESHOLD = 15    # ~0.5s at 30fps

# ─────────────────────────────────────────────
#  PALETTE  (starts with 5 defaults + eraser)
# ─────────────────────────────────────────────
palette = [
    {"label": "Coral",   "color": (80,  110, 255)},
    {"label": "Mint",    "color": (160, 220, 120)},
    {"label": "Sky",     "color": (230, 180,  80)},
    {"label": "Lilac",   "color": (200, 140, 220)},
    {"label": "Gold",    "color": (40,  200, 255)},
    {"label": "Eraser",  "color": None},           # None = eraser
]
MAX_PALETTE_SLOTS = 9   # max swatches before eraser

selected_index  = 0
drawing_color   = palette[0]["color"]
brush_size      = 4
eraser_size     = 28

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
prev_point   = None
hover_index  = -1
hover_frames = 0
clear_frames = 0

# Color wheel state
wheel_open        = False
wheel_hover_angle = -1
wheel_hover_frames = 0
WHEEL_HOVER_THRESH = 10

# Notification
notif_text   = ""
notif_frames = 0
NOTIF_DUR    = 45


# ─────────────────────────────────────────────
#  HELPERS — COLOR
# ─────────────────────────────────────────────
def hsv_to_bgr(h_deg, s=1.0, v=1.0):
    """Convert HSV (h in 0-360) to BGR tuple."""
    h_norm = h_deg / 360.0
    rgb = np.array(
        cv2.cvtColor(
            np.uint8([[[int(h_norm * 179), int(s * 255), int(v * 255)]]]),
            cv2.COLOR_HSV2BGR
        )
    )[0][0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def bgr_to_hsv_hue(bgr):
    """Return hue (0-360) of a BGR color."""
    arr = np.uint8([[bgr]])
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)
    return int(hsv[0][0][0]) * 2  # OpenCV hue is 0-179


# ─────────────────────────────────────────────
#  HELPERS — GESTURE
# ─────────────────────────────────────────────
def get_finger_states(lm):
    fingers = [lm[4].x < lm[3].x]
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        fingers.append(lm[tip].y < lm[pip].y)
    return fingers

def is_drawing_gesture(fingers):
    return fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]

def is_clear_gesture(fingers, lm):
    return (lm[8].y < lm[6].y and lm[12].y < lm[10].y and
            lm[16].y < lm[14].y and lm[20].y < lm[18].y)

def is_fist(fingers):
    return not any(fingers[1:])

def is_two_fingers_up(fingers):
    """Index + middle up, rest down — used to open/close color wheel."""
    return fingers[1] and fingers[2] and not fingers[3] and not fingers[4]


# ─────────────────────────────────────────────
#  HELPERS — GEOMETRY
# ─────────────────────────────────────────────
def point_in_circle(px, py, cx, cy, r):
    return math.hypot(px - cx, py - cy) < r

def angle_of_point(px, py, cx, cy):
    """Angle in degrees, 0 = right, going clockwise."""
    ang = math.degrees(math.atan2(-(py - cy), px - cx))
    return ang % 360


# ─────────────────────────────────────────────
#  DRAW COLOR WHEEL
# ─────────────────────────────────────────────
def draw_color_wheel(overlay, center, outer_r, inner_r, hover_angle):
    cx, cy = center
    segments = 36
    for i in range(segments):
        start_ang = i * (360 / segments)
        end_ang   = (i + 1) * (360 / segments)
        hue       = int(start_ang)
        color     = hsv_to_bgr(hue)

        # Build filled arc using polygon
        pts = []
        for a in np.linspace(math.radians(start_ang), math.radians(end_ang), 6):
            pts.append([int(cx + outer_r * math.cos(a)), int(cy - outer_r * math.sin(a))])
        for a in np.linspace(math.radians(end_ang), math.radians(start_ang), 6):
            pts.append([int(cx + inner_r * math.cos(a)), int(cy - inner_r * math.sin(a))])

        pts = np.array(pts, dtype=np.int32)

        # Highlight hovered segment
        if hover_angle >= 0 and start_ang <= hover_angle < end_ang:
            bright = hsv_to_bgr(hue, 1.0, 1.0)
            cv2.fillPoly(overlay, [pts], bright)
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
        else:
            cv2.fillPoly(overlay, [pts], color)

    # Inner circle — dark background with "+" icon
    cv2.circle(overlay, center, inner_r, (20, 20, 20), cv2.FILLED)
    cv2.circle(overlay, center, inner_r, (60, 60, 60), 2)
    cv2.putText(overlay, "+", (cx - 10, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 140), 2)

    # Outer ring border
    cv2.circle(overlay, center, outer_r, (80, 80, 80), 2)

    # Label above wheel
    cv2.putText(overlay, "COLOR", (cx - 22, cy - outer_r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.putText(overlay, "WHEEL", (cx - 22, cy - outer_r + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


# ─────────────────────────────────────────────
#  DRAW TOP PALETTE BAR
# ─────────────────────────────────────────────
def draw_palette_bar(frame, palette, selected_index, hover_index, hover_frames):
    # Frosted dark bar
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, PALETTE_H), (18, 18, 18), cv2.FILLED)
    cv2.addWeighted(bar, 0.82, frame, 0.18, 0, frame)

    # Subtle separator line
    cv2.line(frame, (0, PALETTE_H), (w, PALETTE_H), (55, 55, 55), 1)

    eraser_idx = len(palette) - 1

    for i, swatch in enumerate(palette):
        cx = SWATCH_X_START + i * SWATCH_SPACING
        cy = SWATCH_Y
        is_eraser = (swatch["color"] is None)
        color = (40, 40, 40) if is_eraser else swatch["color"]

        # Hover ring + progress arc
        if i == hover_index and hover_frames > 0:
            progress = hover_frames / HOVER_THRESHOLD
            angle = int(360 * progress)
            cv2.ellipse(frame, (cx, cy), (SWATCH_R + 6, SWATCH_R + 6),
                        -90, 0, angle, (255, 255, 255), 2)

        # Selected: bright white ring
        if i == selected_index:
            cv2.circle(frame, (cx, cy), SWATCH_R + 5, (255, 255, 255), 2)

        # Swatch fill
        cv2.circle(frame, (cx, cy), SWATCH_R, color, cv2.FILLED)
        cv2.circle(frame, (cx, cy), SWATCH_R, (60, 60, 60), 1)

        # Eraser icon
        if is_eraser:
            cv2.putText(frame, "E", (cx - 7, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)
        # Selected dot
        elif i == selected_index:
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

    return frame


# ─────────────────────────────────────────────
#  DRAW HUD (bottom bar)
# ─────────────────────────────────────────────
def draw_hud(frame, gesture_label, hand_count, notif_text, notif_frames, clear_frames):
    hud_y = h - 44

    # Frosted bottom strip
    strip = frame.copy()
    cv2.rectangle(strip, (0, hud_y - 4), (w, h), (18, 18, 18), cv2.FILLED)
    cv2.addWeighted(strip, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, hud_y - 4), (w, hud_y - 4), (55, 55, 55), 1)

    # Gesture label
    cv2.putText(frame, gesture_label.upper(),
                (14, hud_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    # Hand count dots
    for i in range(hand_count):
        cv2.circle(frame, (w - 60 + i * 18, hud_y + 10), 5, (100, 220, 140), cv2.FILLED)

    # Clear progress bar at very bottom
    if clear_frames > 0:
        progress = int(w * clear_frames / CLEAR_THRESHOLD)
        cv2.rectangle(frame, (0, h - 4), (progress, h), (80, 200, 255), cv2.FILLED)

    # Notification toast
    if notif_frames > 0:
        alpha = min(1.0, notif_frames / 10.0) * min(1.0, notif_frames / NOTIF_DUR * 3)
        toast_w = 280
        toast_x = w // 2 - toast_w // 2
        toast_y = h // 2 - 24
        toast = frame.copy()
        cv2.rectangle(toast, (toast_x, toast_y), (toast_x + toast_w, toast_y + 48),
                      (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(toast, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (toast_x, toast_y), (toast_x + toast_w, toast_y + 48),
                      (80, 80, 80), 1)
        cv2.putText(frame, notif_text, (toast_x + 16, toast_y + 31),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1)

    return frame


# ─────────────────────────────────────────────
#  SWATCH HIT TEST
# ─────────────────────────────────────────────
def get_swatch_at(cx, cy):
    if cy < PALETTE_H:
        for i in range(len(palette)):
            sx = SWATCH_X_START + i * SWATCH_SPACING
            if math.hypot(cx - sx, cy - SWATCH_Y) < SWATCH_R + 8:
                return i
    return -1


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────
print("✅ AR Drawing App running.")
print("   ☝️  Index finger        = draw")
print("   ✌️  Two fingers up      = toggle color wheel")
print("   🖐️🖐️  Both palms open    = clear canvas")
print("   Q                     = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    h, w, _ = frame.shape
    gesture_label = "ready"
    hand_count = 0

    if result.multi_hand_landmarks:
        num_hands = len(result.multi_hand_landmarks)
        hand_count = num_hands

        # ── TWO PALM CLEAR CHECK ──
        all_open_palms = False
        if num_hands == 2:
            checks = []
            for hl in result.multi_hand_landmarks:
                lm = hl.landmark
                checks.append(is_clear_gesture(get_finger_states(lm), lm))
            all_open_palms = all(checks)

        if all_open_palms:
            clear_frames += 1
            prev_point = None
            hover_index = -1
            hover_frames = 0
            gesture_label = "clearing..."

            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

            if clear_frames >= CLEAR_THRESHOLD:
                canvas = np.zeros((h, w, 3), dtype=np.uint8)
                clear_frames = 0
                gesture_label = "cleared"
                notif_text = "Canvas cleared"
                notif_frames = NOTIF_DUR

        else:
            clear_frames = 0
            hl = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hl, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            lm = hl.landmark
            fingers = get_finger_states(lm)
            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)

            # ── TOGGLE COLOR WHEEL (✌️ two fingers) ──
            if is_two_fingers_up(fingers):
                gesture_label = "color wheel"
                prev_point = None
                hover_index = -1

                # Check if fingertip is on wheel
                dist = math.hypot(cx - WHEEL_CENTER[0], cy - WHEEL_CENTER[1])

                if WHEEL_INNER_R < dist < WHEEL_R and wheel_open:
                    # Hovering over a color segment
                    ang = angle_of_point(cx, cy, WHEEL_CENTER[0], WHEEL_CENTER[1])
                    wheel_hover_angle = ang
                    wheel_hover_frames += 1

                    if wheel_hover_frames >= WHEEL_HOVER_THRESH:
                        # Add this hue to palette (before eraser)
                        new_color = hsv_to_bgr(int(ang))
                        eraser = palette.pop()          # remove eraser temporarily
                        # Replace oldest non-default if full
                        if len(palette) >= MAX_PALETTE_SLOTS:
                            palette.pop(0)
                        palette.append({"label": f"H{int(ang)}", "color": new_color})
                        palette.append(eraser)          # put eraser back at end
                        selected_index = len(palette) - 2
                        drawing_color = new_color
                        brush_size = 4
                        wheel_hover_frames = 0
                        wheel_open = False
                        notif_text = f"Color added to palette"
                        notif_frames = NOTIF_DUR
                else:
                    wheel_hover_angle = -1
                    wheel_hover_frames = 0

                # Toggle wheel open/closed on fist→two-fingers transition
                # (handled by checking if finger just entered two-finger state)
                if not wheel_open:
                    wheel_open = True

            elif is_fist(fingers):
                if wheel_open:
                    wheel_open = False
                gesture_label = "paused"
                prev_point = None
                hover_index = -1
                hover_frames = 0

            # ── DRAWING GESTURE ──
            elif is_drawing_gesture(fingers):
                wheel_open = False
                swatch_i = get_swatch_at(cx, cy)

                if swatch_i >= 0:
                    gesture_label = f"selecting"
                    prev_point = None

                    if swatch_i == hover_index:
                        hover_frames += 1
                    else:
                        hover_index = swatch_i
                        hover_frames = 0

                    if hover_frames >= HOVER_THRESHOLD:
                        selected_index = swatch_i
                        if palette[swatch_i]["color"] is None:
                            drawing_color = (0, 0, 0)
                            brush_size = eraser_size
                        else:
                            drawing_color = palette[swatch_i]["color"]
                            brush_size = 4
                        hover_frames = 0
                        gesture_label = f"selected"
                        notif_text = f"{palette[swatch_i]['label']} selected"
                        notif_frames = NOTIF_DUR

                else:
                    hover_index = -1
                    hover_frames = 0
                    gesture_label = "drawing"

                    # Fingertip cursor
                    cv2.circle(frame, (cx, cy), brush_size + 4, drawing_color, 2)
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), cv2.FILLED)

                    if cy > PALETTE_H:
                        if prev_point is not None and prev_point[1] > PALETTE_H:
                            cv2.line(canvas, prev_point, (cx, cy), drawing_color, brush_size)
                        prev_point = (cx, cy)
                    else:
                        prev_point = None

            else:
                gesture_label = "paused"
                prev_point = None
                hover_index = -1
                hover_frames = 0
                wheel_open = False

    else:
        prev_point = None
        hover_index = -1
        hover_frames = 0
        clear_frames = 0

    # ── BLEND CANVAS ──
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, canvas_fg)

    # ── COLOR WHEEL OVERLAY ──
    if wheel_open:
        overlay = combined.copy()
        draw_color_wheel(overlay, WHEEL_CENTER, WHEEL_R, WHEEL_INNER_R, wheel_hover_angle)
        cv2.addWeighted(overlay, 0.92, combined, 0.08, 0, combined)

    # ── PALETTE BAR ──
    combined = draw_palette_bar(combined, palette, selected_index, hover_index, hover_frames)

    # ── CURRENT COLOR SWATCH (bottom right) ──
    swatch_color = (40, 40, 40) if drawing_color == (0, 0, 0) else drawing_color
    cv2.circle(combined, (w - 36, h - 56), 18, swatch_color, cv2.FILLED)
    cv2.circle(combined, (w - 36, h - 56), 18, (90, 90, 90), 2)

    # ── HUD ──
    if notif_frames > 0:
        notif_frames -= 1
    combined = draw_hud(combined, gesture_label, hand_count, notif_text, notif_frames, clear_frames)

    cv2.imshow("AR Drawing", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
