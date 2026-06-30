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
PALETTE_H       = 72
SWATCH_R        = 22
SWATCH_SPACING  = 64
SWATCH_Y        = PALETTE_H // 2
SWATCH_X_START  = 48

WHEEL_CENTER    = (w - 90, h // 2)
WHEEL_R         = 70
WHEEL_INNER_R   = 28

HOVER_THRESHOLD  = 12
CLEAR_THRESHOLD  = 15

# ─────────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────────
palette = [
    {"label": "Coral",   "color": (80,  110, 255)},
    {"label": "Mint",    "color": (160, 220, 120)},
    {"label": "Sky",     "color": (230, 180,  80)},
    {"label": "Lilac",   "color": (200, 140, 220)},
    {"label": "Gold",    "color": (40,  200, 255)},
    {"label": "Eraser",  "color": None},
]
MAX_PALETTE_SLOTS = 9

selected_index = 0
drawing_color  = palette[0]["color"]
brush_size     = 4
eraser_size    = 28

# ─────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────
prev_point    = None
hover_index   = -1
hover_frames  = 0
clear_frames  = 0

wheel_open         = False
wheel_hover_angle  = -1
wheel_hover_frames = 0
WHEEL_HOVER_THRESH = 10

notif_text   = ""
notif_frames = 0
NOTIF_DUR    = 45

# ── Stroke accumulator ──
current_stroke  = []          # live points for this pen-down segment
was_drawing     = False       # tracks pen-down→pen-up transition

# ── Correction flash ──
flash_frames    = 0
FLASH_DUR       = 8


# ─────────────────────────────────────────────
#  COLOR HELPERS
# ─────────────────────────────────────────────
def hsv_to_bgr(h_deg, s=1.0, v=1.0):
    h_norm = h_deg / 360.0
    rgb = np.array(cv2.cvtColor(
        np.uint8([[[int(h_norm * 179), int(s * 255), int(v * 255)]]]),
        cv2.COLOR_HSV2BGR))[0][0]
    return (int(rgb[0]), int(rgb[1]), int(rgb[2]))


# ─────────────────────────────────────────────
#  GESTURE HELPERS
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
    return fingers[1] and fingers[2] and not fingers[3] and not fingers[4]


# ─────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────
def point_in_circle(px, py, cx, cy, r):
    return math.hypot(px - cx, py - cy) < r

def angle_of_point(px, py, cx, cy):
    ang = math.degrees(math.atan2(-(py - cy), px - cx))
    return ang % 360

def fit_circle(pts):
    """
    Algebraic circle fit (Kasa method).
    Returns (cx, cy, r) or None if degenerate.
    """
    pts = np.array(pts, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones(len(x))])
    b = x**2 + y**2
    try:
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy, c = result
    r = math.sqrt(c + cx**2 + cy**2)
    if r < 5 or r > max(w, h):
        return None
    return (int(cx), int(cy), int(r))


def circularity_score(pts, cx, cy, r):
    """
    Fraction of points within ±20% of the fitted radius.
    Also checks angular coverage (must go > 270°).
    """
    pts = np.array(pts, dtype=np.float64)
    dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    tolerance = 0.20 * r
    close = np.sum(np.abs(dists - r) < tolerance) / len(dists)

    # Angular coverage
    angles = np.degrees(np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)) % 360
    angles_sorted = np.sort(angles)
    gaps = np.diff(angles_sorted)
    max_gap = np.max(gaps) if len(gaps) > 0 else 360
    coverage = (360 - max_gap) / 360

    return close, coverage


def linearity_score(pts):
    """
    How well points fit a straight line (PCA residuals).
    Returns a score 0-1 where 1 = perfectly straight.
    """
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s[0] < 1e-6:
        return 0.0
    # Ratio of variance explained by first principal component
    linearity = s[0] / (s[0] + s[1] + 1e-6)
    # Also check that residual spread is tight
    total_spread = np.max(np.hypot(centered[:, 0], centered[:, 1]))
    proj = centered @ np.array([1, 0])   # will be recomputed below properly
    # Recompute using actual principal axis
    vt = np.linalg.svd(centered, full_matrices=False)[2]
    axis = vt[0]
    projections = centered @ axis
    residuals = centered - np.outer(projections, axis)
    residual_rms = np.sqrt(np.mean(residuals[:, 0]**2 + residuals[:, 1]**2))
    tightness = max(0.0, 1.0 - residual_rms / (total_spread + 1e-6))
    return linearity * tightness


def bezier_smooth(pts, n_out=None):
    """
    Light cubic Bezier smoothing — resamples stroke through
    averaged control points. Preserves overall shape.
    """
    pts = np.array(pts, dtype=np.float64)
    if len(pts) < 4:
        return pts.astype(np.int32).tolist()

    # Smooth with a small sliding window average (window=3)
    window = 3
    smoothed = []
    for i in range(len(pts)):
        lo = max(0, i - window)
        hi = min(len(pts), i + window + 1)
        smoothed.append(pts[lo:hi].mean(axis=0))
    smoothed = np.array(smoothed)

    if n_out is None:
        n_out = len(pts)

    # Resample to n_out points along the smoothed path
    dists = np.sqrt(np.sum(np.diff(smoothed, axis=0)**2, axis=1))
    cumlen = np.concatenate([[0], np.cumsum(dists)])
    total = cumlen[-1]
    if total < 1:
        return pts.astype(np.int32).tolist()

    sample_at = np.linspace(0, total, n_out)
    out_x = np.interp(sample_at, cumlen, smoothed[:, 0])
    out_y = np.interp(sample_at, cumlen, smoothed[:, 1])
    return list(zip(out_x.astype(np.int32), out_y.astype(np.int32)))


# ─────────────────────────────────────────────
#  POST-STROKE CORRECTION
# ─────────────────────────────────────────────
CIRCLE_CLOSE_SCORE  = 0.85   # strict: 85% of points near fitted radius
CIRCLE_COVERAGE     = 0.75   # must cover 75% of the circle arc
LINE_SCORE          = 0.92   # strict: 92% linearity

def correct_stroke(pts, color, bsize, cnv):
    """
    Analyze a completed stroke and draw the corrected version onto cnv.
    Returns (corrected: bool, label: str)
    """
    if len(pts) < 6:
        # Too short — just smooth lightly
        smooth = bezier_smooth(pts)
        for i in range(1, len(smooth)):
            cv2.line(cnv, smooth[i-1], smooth[i], color, bsize)
        return False, ""

    # ── 1. Try circle snap ──
    fit = fit_circle(pts)
    if fit is not None:
        cx, cy, r = fit
        close, coverage = circularity_score(pts, cx, cy, r)
        if close >= CIRCLE_CLOSE_SCORE and coverage >= CIRCLE_COVERAGE:
            cv2.circle(cnv, (cx, cy), r, color, bsize)
            return True, "Circle snapped"

    # ── 2. Try line snap ──
    score = linearity_score(pts)
    if score >= LINE_SCORE:
        p1 = pts[0]
        p2 = pts[-1]
        cv2.line(cnv, p1, p2, color, bsize)
        return True, "Line snapped"

    # ── 3. Light Bezier smooth ──
    smooth = bezier_smooth(pts)
    for i in range(1, len(smooth)):
        cv2.line(cnv, smooth[i-1], smooth[i], color, bsize)
    return False, ""


def erase_stroke(pts, cnv, bsize):
    """Re-erase a stroke region after correction pass."""
    for i in range(1, len(pts)):
        cv2.line(cnv, pts[i-1], pts[i], (0, 0, 0), bsize)


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

        pts = []
        for a in np.linspace(math.radians(start_ang), math.radians(end_ang), 6):
            pts.append([int(cx + outer_r * math.cos(a)), int(cy - outer_r * math.sin(a))])
        for a in np.linspace(math.radians(end_ang), math.radians(start_ang), 6):
            pts.append([int(cx + inner_r * math.cos(a)), int(cy - inner_r * math.sin(a))])
        pts = np.array(pts, dtype=np.int32)

        if hover_angle >= 0 and start_ang <= hover_angle < end_ang:
            cv2.fillPoly(overlay, [pts], hsv_to_bgr(hue, 1.0, 1.0))
            cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
        else:
            cv2.fillPoly(overlay, [pts], color)

    cv2.circle(overlay, center, inner_r, (20, 20, 20), cv2.FILLED)
    cv2.circle(overlay, center, inner_r, (60, 60, 60), 2)
    cv2.putText(overlay, "+", (cx - 10, cy + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (140, 140, 140), 2)
    cv2.circle(overlay, center, outer_r, (80, 80, 80), 2)
    cv2.putText(overlay, "COLOR", (cx - 22, cy - outer_r - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)
    cv2.putText(overlay, "WHEEL", (cx - 22, cy - outer_r + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


# ─────────────────────────────────────────────
#  DRAW TOP PALETTE BAR
# ─────────────────────────────────────────────
def draw_palette_bar(frame, palette, selected_index, hover_index, hover_frames):
    bar = frame.copy()
    cv2.rectangle(bar, (0, 0), (w, PALETTE_H), (18, 18, 18), cv2.FILLED)
    cv2.addWeighted(bar, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, PALETTE_H), (w, PALETTE_H), (55, 55, 55), 1)

    for i, swatch in enumerate(palette):
        cx = SWATCH_X_START + i * SWATCH_SPACING
        cy = SWATCH_Y
        is_eraser = (swatch["color"] is None)
        color = (40, 40, 40) if is_eraser else swatch["color"]

        if i == hover_index and hover_frames > 0:
            progress = hover_frames / HOVER_THRESHOLD
            angle = int(360 * progress)
            cv2.ellipse(frame, (cx, cy), (SWATCH_R + 6, SWATCH_R + 6),
                        -90, 0, angle, (255, 255, 255), 2)

        if i == selected_index:
            cv2.circle(frame, (cx, cy), SWATCH_R + 5, (255, 255, 255), 2)

        cv2.circle(frame, (cx, cy), SWATCH_R, color, cv2.FILLED)
        cv2.circle(frame, (cx, cy), SWATCH_R, (60, 60, 60), 1)

        if is_eraser:
            cv2.putText(frame, "E", (cx - 7, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 2)
        elif i == selected_index:
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), cv2.FILLED)

    return frame


# ─────────────────────────────────────────────
#  DRAW HUD
# ─────────────────────────────────────────────
def draw_hud(frame, gesture_label, hand_count, notif_text, notif_frames,
             clear_frames, flash_frames):
    hud_y = h - 44

    strip = frame.copy()
    cv2.rectangle(strip, (0, hud_y - 4), (w, h), (18, 18, 18), cv2.FILLED)
    cv2.addWeighted(strip, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, hud_y - 4), (w, hud_y - 4), (55, 55, 55), 1)

    cv2.putText(frame, gesture_label.upper(),
                (14, hud_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

    for i in range(hand_count):
        cv2.circle(frame, (w - 60 + i * 18, hud_y + 10), 5, (100, 220, 140), cv2.FILLED)

    if clear_frames > 0:
        progress = int(w * clear_frames / CLEAR_THRESHOLD)
        cv2.rectangle(frame, (0, h - 4), (progress, h), (80, 200, 255), cv2.FILLED)

    # Correction flash — brief white vignette on canvas correction
    if flash_frames > 0:
        alpha = flash_frames / FLASH_DUR * 0.25
        flash_overlay = frame.copy()
        cv2.rectangle(flash_overlay, (0, PALETTE_H), (w, hud_y - 4),
                      (180, 220, 255), cv2.FILLED)
        cv2.addWeighted(flash_overlay, alpha, frame, 1 - alpha, 0, frame)

    if notif_frames > 0:
        toast_w = 300
        toast_x = w // 2 - toast_w // 2
        toast_y = h // 2 - 24
        toast = frame.copy()
        cv2.rectangle(toast, (toast_x, toast_y),
                      (toast_x + toast_w, toast_y + 48), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(toast, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (toast_x, toast_y),
                      (toast_x + toast_w, toast_y + 48), (80, 80, 80), 1)
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
print("   Lift finger after stroke = auto-correct")
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
    hand_count    = 0
    is_drawing_now = False

    if result.multi_hand_landmarks:
        num_hands  = len(result.multi_hand_landmarks)
        hand_count = num_hands

        # ── TWO PALM CLEAR ──
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
            current_stroke = []
            was_drawing = False
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

            lm      = hl.landmark
            fingers = get_finger_states(lm)
            cx      = int(lm[8].x * w)
            cy      = int(lm[8].y * h)

            # ── COLOR WHEEL ──
            if is_two_fingers_up(fingers):
                gesture_label = "color wheel"
                prev_point = None
                hover_index = -1
                dist = math.hypot(cx - WHEEL_CENTER[0], cy - WHEEL_CENTER[1])

                if WHEEL_INNER_R < dist < WHEEL_R and wheel_open:
                    ang = angle_of_point(cx, cy, WHEEL_CENTER[0], WHEEL_CENTER[1])
                    wheel_hover_angle = ang
                    wheel_hover_frames += 1

                    if wheel_hover_frames >= WHEEL_HOVER_THRESH:
                        new_color = hsv_to_bgr(int(ang))
                        eraser = palette.pop()
                        if len(palette) >= MAX_PALETTE_SLOTS:
                            palette.pop(0)
                        palette.append({"label": f"H{int(ang)}", "color": new_color})
                        palette.append(eraser)
                        selected_index = len(palette) - 2
                        drawing_color = new_color
                        brush_size = 4
                        wheel_hover_frames = 0
                        wheel_open = False
                        notif_text = "Color added to palette"
                        notif_frames = NOTIF_DUR
                else:
                    wheel_hover_angle = -1
                    wheel_hover_frames = 0

                if not wheel_open:
                    wheel_open = True

            elif is_fist(fingers):
                if wheel_open:
                    wheel_open = False
                gesture_label = "paused"
                prev_point = None
                hover_index = -1
                hover_frames = 0

            # ── DRAWING ──
            elif is_drawing_gesture(fingers):
                wheel_open = False
                swatch_i = get_swatch_at(cx, cy)

                if swatch_i >= 0:
                    # Hovering palette
                    gesture_label = "selecting"
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
                        notif_text = f"{palette[swatch_i]['label']} selected"
                        notif_frames = NOTIF_DUR

                else:
                    hover_index = -1
                    hover_frames = 0
                    gesture_label = "drawing"
                    is_drawing_now = True

                    # Live cursor
                    cv2.circle(frame, (cx, cy), brush_size + 4, drawing_color, 2)
                    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), cv2.FILLED)

                    if cy > PALETTE_H:
                        if prev_point is not None and prev_point[1] > PALETTE_H:
                            # Draw raw onto canvas
                            cv2.line(canvas, prev_point, (cx, cy),
                                     drawing_color, brush_size)
                            current_stroke.append((cx, cy))
                        else:
                            current_stroke = [(cx, cy)]
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

    # ─────────────────────────────────────────
    #  POST-STROKE CORRECTION
    #  Triggers on pen-up: was_drawing → not drawing
    # ─────────────────────────────────────────
    if was_drawing and not is_drawing_now:
        if len(current_stroke) >= 6:
            # Erase the raw shaky stroke from canvas
            for i in range(1, len(current_stroke)):
                cv2.line(canvas, current_stroke[i-1], current_stroke[i],
                         (0, 0, 0), brush_size + 4)   # slightly wider erase

            # Draw the corrected version
            corrected, label = correct_stroke(
                current_stroke, drawing_color, brush_size, canvas)

            if corrected:
                flash_frames = FLASH_DUR
                notif_text   = f"✦ {label}"
                notif_frames = NOTIF_DUR

        current_stroke = []

    was_drawing = is_drawing_now

    # ── Flash countdown ──
    if flash_frames > 0:
        flash_frames -= 1

    # ── BLEND CANVAS ──
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    canvas_fg = cv2.bitwise_and(canvas, canvas, mask=mask)
    combined = cv2.add(frame_bg, canvas_fg)

    # ── COLOR WHEEL ──
    if wheel_open:
        overlay = combined.copy()
        draw_color_wheel(overlay, WHEEL_CENTER, WHEEL_R, WHEEL_INNER_R, wheel_hover_angle)
        cv2.addWeighted(overlay, 0.92, combined, 0.08, 0, combined)

    # ── PALETTE BAR ──
    combined = draw_palette_bar(combined, palette, selected_index, hover_index, hover_frames)

    # ── ACTIVE COLOR DOT ──
    swatch_color = (40, 40, 40) if drawing_color == (0, 0, 0) else drawing_color
    cv2.circle(combined, (w - 36, h - 56), 18, swatch_color, cv2.FILLED)
    cv2.circle(combined, (w - 36, h - 56), 18, (90, 90, 90), 2)

    # ── HUD ──
    if notif_frames > 0:
        notif_frames -= 1
    combined = draw_hud(combined, gesture_label, hand_count,
                        notif_text, notif_frames, clear_frames, flash_frames)

    cv2.imshow("AR Drawing", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
