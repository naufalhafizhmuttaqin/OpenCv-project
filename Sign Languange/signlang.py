import cv2
import mediapipe as mp
from collections import deque

# ── MediaPipe setup ──────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ── Landmark indices ─────────────────────────────────────────────
TIP = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky
DIP = [3, 7, 11, 15, 19]
PIP = [2, 6, 10, 14, 18]
MCP = [1, 5,  9, 13, 17]

# ── Helper functions ─────────────────────────────────────────────
def lxy(lm_list, idx):
    return lm_list[idx][1], lm_list[idx][2]

def dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def finger_up(lm_list, f):
    """Finger f (1–4) is extended upward."""
    return lm_list[TIP[f]][2] < lm_list[PIP[f]][2] - 8

def finger_curled(lm_list, f):
    """Finger f (1–4) tip is below its MCP — tightly curled."""
    return lm_list[TIP[f]][2] > lm_list[MCP[f]][2]

def finger_half(lm_list, f):
    """Finger f is bent but not fully curled (tip between PIP and MCP)."""
    tip_y = lm_list[TIP[f]][2]
    pip_y = lm_list[PIP[f]][2]
    mcp_y = lm_list[MCP[f]][2]
    return mcp_y > tip_y >= pip_y  # tip between MCP and PIP level

def thumb_open(lm_list, hand_label):
    """Thumb tip clearly away from palm (sideways)."""
    tx, ty = lxy(lm_list, TIP[0])
    px, py = lxy(lm_list, DIP[0])   # thumb IP joint
    if hand_label == "Right":
        return tx > px + 12
    else:
        return tx < px - 12

def thumb_up_vertical(lm_list):
    """Thumb tip is above its IP joint (thumb pointing up)."""
    return lm_list[TIP[0]][2] < lm_list[DIP[0]][2] - 10

def hand_scale(lm_list):
    """Reference distance: wrist to index MCP."""
    w = lxy(lm_list, 0)
    m = lxy(lm_list, 5)
    return max(dist(w, m), 1)

# ── J trajectory buffer (needs motion) ───────────────────────────
j_traj = deque(maxlen=12)

# ── Main classifier ───────────────────────────────────────────────
def classify_letter(lm_list, hand_label):
    # Per-finger state
    th = thumb_open(lm_list, hand_label)
    th_v = thumb_up_vertical(lm_list)
    i  = finger_up(lm_list, 1)    # index
    m  = finger_up(lm_list, 2)    # middle
    r  = finger_up(lm_list, 3)    # ring
    p  = finger_up(lm_list, 4)    # pinky

    ic = finger_curled(lm_list, 1)
    mc = finger_curled(lm_list, 2)
    rc = finger_curled(lm_list, 3)
    pc = finger_curled(lm_list, 4)

    ih = finger_half(lm_list, 1)   # index half-bent
    mh = finger_half(lm_list, 2)

    # Key coordinates
    tt  = lxy(lm_list, 4)   # thumb tip
    it  = lxy(lm_list, 8)   # index tip
    mt  = lxy(lm_list, 12)  # middle tip
    rt  = lxy(lm_list, 16)  # ring tip
    pt  = lxy(lm_list, 20)  # pinky tip
    i6  = lxy(lm_list, 6)   # index PIP
    wrist = lxy(lm_list, 0)
    imcp  = lxy(lm_list, 5)  # index MCP

    sc = hand_scale(lm_list)

    # Normalised distances
    nti = dist(tt, it)  / sc   # thumb–index
    ntm = dist(tt, mt)  / sc   # thumb–middle
    ntr = dist(tt, rt)  / sc   # thumb–ring
    ntp = dist(tt, pt)  / sc   # thumb–pinky
    nim = dist(it, mt)  / sc   # index–middle
    nir = dist(it, rt)  / sc   # index–ring
    nip = dist(it, pt)  / sc   # index–pinky
    nmp = dist(mt, pt)  / sc   # middle–pinky

    # ── A: fist, thumb rests on side of index — NOT over fingers ──
    # Thumb is beside the fist, not tucked under
    if ic and mc and rc and pc and not th:
        return 'A'

    # ── B: all 4 fingers straight up, thumb tucked across palm ────
    if i and m and r and p and not th:
        return 'B'

    # ── C: curved C-shape, thumb and fingers form open curve ──────
    # All fingers partially bent, thumb out forming a C
    if not i and not m and not r and not p:
        if 0.5 < nti < 1.1 and ntm < 1.0:
            return 'C'

    # ── D: index up, thumb touches middle tip, ring+pinky curl ────
    if i and not m and not r and not p:
        if ntm < 0.5:
            return 'D'

    # ── E: all fingers bent down, touching tips, thumb tucked ─────
    # Tips all near thumb, all curled (not fully closed like A)
    if not i and not m and not r and not p and not th:
        all_tips_low = all(
            lm_list[TIP[f]][2] > lm_list[PIP[f]][2]
            for f in range(1, 5)
        )
        if all_tips_low and nti > 0.45:
            return 'E'

    # ── F: index+thumb touch (OK sign), middle/ring/pinky up ──────
    if m and r and p and not i and nti < 0.4:
        return 'F'

    # ── G: index + thumb point sideways (like a gun sideways) ─────
    # Index extends horizontally, thumb parallel
    if i and not m and not r and not p and th:
        ix, iy = it
        wx, wy = wrist
        pointing_sideways = abs(ix - wx) > abs(iy - wy) * 0.8
        if pointing_sideways:
            return 'G'

    # ── H: index + middle extend horizontally side by side ────────
    if i and m and not r and not p and not th:
        ix, iy = it
        mx2, my = mt
        wx, wy = wrist
        both_sideways = (abs(ix - wx) > abs(iy - wy) * 0.7 and
                         abs(mx2 - wx) > abs(my - wy) * 0.7)
        if both_sideways:
            return 'H'

    # ── I: only pinky up, rest curled ─────────────────────────────
    if p and not i and not m and not r and not th:
        return 'I'

    # ── J: pinky up + motion (draw J shape) ───────────────────────
    # Detect as I-shape but with downward-then-curl motion
    if p and not i and not m and not r:
        px2, py2 = pt
        j_traj.append(py2)
        if len(j_traj) == j_traj.maxlen:
            motion = j_traj[-1] - j_traj[0]   # positive = moved down
            if motion > sc * 0.3:
                return 'J'

    # ── K: index + middle up, thumb between them pointing up ───────
    if i and m and not r and not p and th:
        if nti < 0.7 and nim > 0.25:
            return 'K'

    # ── L: index up + thumb out (right angle) ─────────────────────
    if i and th and not m and not r and not p:
        return 'L'

    # ── M: three fingers folded over thumb ────────────────────────
    # Index + middle + ring curl over thumb, pinky curl too
    if not i and not m and not r and not p:
        if nti < 0.5 and ntm < 0.6 and ntr < 0.7:
            return 'M'

    # ── N: two fingers (index+middle) over thumb ──────────────────
    if not i and not m and not r and not p:
        if nti < 0.5 and ntm < 0.6 and ntr >= 0.65:
            return 'N'

    # ── O: all fingertips pinched toward thumb, making circle ─────
    if not i and not m and not r and not p:
        if nti < 0.5 and ntm < 0.6 and ntp < 0.7:
            return 'O'

    # ── P: like K but hand points downward ────────────────────────
    if i and m and not r and not p and th:
        ix, iy = it
        wx, wy = wrist
        pointing_down = iy > wy      # tip below wrist
        if pointing_down:
            return 'P'

    # ── Q: like G but pointing downward ───────────────────────────
    if i and th and not m and not r and not p:
        ix, iy = it
        wx, wy = wrist
        if iy > wy + 15:
            return 'Q'

    # ── R: index + middle crossed / close together ─────────────────
    if i and m and not r and not p and not th:
        if nim < 0.18:
            return 'R'

    # ── S: fist with thumb wrapped OVER fingers ───────────────────
    if ic and mc and rc and pc:
        # Thumb tip is in front of (lower than) index/middle tips
        if lm_list[TIP[0]][2] < lm_list[TIP[1]][2] and nti < 0.6:
            return 'S'

    # ── T: thumb inserted between index and middle finger ─────────
    if not i and not m and not r and not p:
        # Thumb tip near index MCP area
        imcp_pt = lxy(lm_list, 5)
        if dist(tt, imcp_pt) / sc < 0.55:
            return 'T'

    # ── U: index + middle up and together (parallel) ──────────────
    if i and m and not r and not p and not th:
        if nim < 0.22:
            return 'U'

    # ── V: index + middle up, spread apart (peace/victory sign) ───
    if i and m and not r and not p and not th:
        if nim >= 0.22:
            return 'V'

    # ── W: index + middle + ring up, spread ───────────────────────
    if i and m and r and not p and not th:
        return 'W'

    # ── X: index finger hooked/crooked, rest curled ───────────────
    if not m and not r and not p and not th:
        tip_y  = lm_list[TIP[1]][2]
        dip_y  = lm_list[DIP[1]][2]
        pip_y  = lm_list[PIP[1]][2]
        mcp_y  = lm_list[MCP[1]][2]
        hooked = pip_y > tip_y > dip_y and mcp_y > pip_y
        if hooked:
            return 'X'

    # ── Y: thumb + pinky extended, rest curled ────────────────────
    if th and p and not i and not m and not r:
        return 'Y'

    # ── Z: index extended, traces Z (simplified: index up, others curl)
    # Detected as Z if index is up and moving in Z pattern
    if i and not m and not r and not p and not th:
        return 'Z'

    return ''

# ── Smoothing buffer ──────────────────────────────────────────────
BUFFER_SIZE = 12
pred_buffer = deque(maxlen=BUFFER_SIZE)

def stable_prediction(buf):
    if len(buf) < BUFFER_SIZE:
        return ''
    counts = {}
    for p in buf:
        if p:
            counts[p] = counts.get(p, 0) + 1
    if not counts:
        return ''
    best = max(counts, key=counts.get)
    return best if counts[best] >= BUFFER_SIZE * 0.65 else ''

# ── Main loop ─────────────────────────────────────────────────────
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.7
) as hands:

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        raw_letter = ''

        if results.multi_hand_landmarks and results.multi_handedness:
            hl  = results.multi_hand_landmarks[0]
            hand_label = results.multi_handedness[0].classification[0].label

            mp_draw.draw_landmarks(img, hl, mp_hands.HAND_CONNECTIONS)

            lm_list = []
            for idx, lm in enumerate(hl.landmark):
                lm_list.append((idx, int(lm.x * w), int(lm.y * h)))

            raw_letter = classify_letter(lm_list, hand_label)

        pred_buffer.append(raw_letter)
        letter = stable_prediction(pred_buffer)

        # ── UI ────────────────────────────────────────────────────
        # Letter box
        box_color = (0, 200, 80) if letter else (40, 40, 40)
        cv2.rectangle(img, (10, 10), (130, 140), (0, 0, 0), -1)
        cv2.rectangle(img, (10, 10), (130, 140), box_color, 2)

        disp = letter if letter else '?'
        txt_color = (0, 255, 100) if letter else (70, 70, 70)
        cv2.putText(img, disp, (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.5, txt_color, 9)

        # Raw prediction (smaller, for debug)
        cv2.putText(img, f"raw: {raw_letter if raw_letter else '-'}",
                    (145, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 80), 1)

        # Hand label
        if results.multi_handedness:
            hl_label = results.multi_handedness[0].classification[0].label
            cv2.putText(img, hl_label, (145, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        # Bottom status
        status = f"Detected: {letter}" if letter else "Tunjukkan pose tangan"
        cv2.putText(img, status, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

        # Hint bar
        cv2.rectangle(img, (0, h - 12), (w, h), (20, 20, 20), -1)
        cv2.putText(img, "Q = keluar", (10, h - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

        cv2.imshow("ASL Sign Language Detector", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()