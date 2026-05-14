"""
Manual Offside Tool — Vanishing Point Edition
==============================================
Usa linhas REAIS do campo (baliza + area) para calcular o ponto de fuga.
As linhas de offside sao desenhadas com perspetiva perfeita.

CALIBRACAO (uma vez por cena):
  C           Entrar no modo calibracao
  Clica 2 pts na LINHA DE BALIZA (ou 6 metros)
  Clica 2 pts na LINHA DA AREA (16.5m)
  → Ponto de fuga calculado automaticamente

ANALISE (2 cliques por jogador):
  1           Modo ATK
  Clique 1    Parte do corpo mais avancada (cabeca, peito, joelho...)
  Clique 2    Relvado diretamente abaixo → LINHA desenhada aqui
  2           Modo DEF — mesmo processo para o 2o ultimo defensor
  SPACE       Veredito (FORA DE JOGO / ONSIDE)
  R           Reset marcacoes (mantem calibracao)

NAVEGACAO:
  D / →       +1 frame
  A / ←       -1 frame
  W / ↑       +25 frames (1 seg)
  S / ↓       -25 frames (1 seg)
  E           Exportar frame anotado
  Q / ESC     Sair
"""

import sys
import cv2
import numpy as np
from utils import read_video


# ── Geometry helpers ──────────────────────────────────────────

def line_from_2pts(p1, p2):
    """Homogeneous line coefficients from two pixel points."""
    return np.cross(
        np.array([p1[0], p1[1], 1.0]),
        np.array([p2[0], p2[1], 1.0])
    )


def intersect_lines(l1, l2):
    """Intersection of two homogeneous lines → (x, y) or None if parallel."""
    p = np.cross(l1, l2)
    if abs(p[2]) < 1e-10:
        return None
    return (p[0] / p[2], p[1] / p[2])


def side_of_line(lp1, lp2, pt):
    """Signed area: which side of directed line lp1→lp2 is pt on?"""
    return ((lp2[0] - lp1[0]) * (pt[1] - lp1[1]) -
            (lp2[1] - lp1[1]) * (pt[0] - lp1[0]))


# ── Colors ────────────────────────────────────────────────────
MAGENTA = (255, 0, 255)
CYAN    = (255, 255, 0)
YELLOW  = (0, 255, 255)
WHITE   = (255, 255, 255)
GRAY    = (180, 180, 180)
BLACK   = (0, 0, 0)


# ── State ─────────────────────────────────────────────────────

class State:
    def __init__(self, frames):
        self.frames = frames
        self.n = len(frames)
        self.idx = 0

        # Calibration
        self.calibrating = False
        self.calib_pts = []        # 4 points: [goal1, goal2, area1, area2]
        self.vanishing_pt = None   # (vx, vy) or None
        self.goal_ref_pt = None    # one point on goal line (for direction)
        self.line_direction = None # fallback if VP at infinity

        # Analysis — single click auto-projects to ground
        self.mode = 'defender'
        self.body_part = 'head'    # 'head', 'chest', 'knee', 'foot'
        self.body_heights = {
            'head':  1.75,
            'chest': 1.30,
            'knee':  0.50,
            'foot':  0.05,
        }

        self.atk_body_pt = None    # body part (dot)
        self.atk_ground_pt = None  # ground below (line anchor)
        self.def_body_pt = None
        self.def_ground_pt = None
        self.verdict = None
        self.distance_cm = None    # real-world distance between lines

        # Mouse for magnifier
        self.mouse_x = 0
        self.mouse_y = 0


# ── Drawing ───────────────────────────────────────────────────

LENS_RADIUS = 75
LENS_ZOOM = 3


def draw_magnifier(frame, state):
    """Small magnifier lens in top-right corner, smooth zoom."""
    h, w = frame.shape[:2]
    # Mouse coords are in original space; scale to upscaled frame
    mx, my = state.mouse_x * 2, state.mouse_y * 2

    src_r = LENS_RADIUS // LENS_ZOOM
    x1, y1 = max(mx - src_r, 0), max(my - src_r, 0)
    x2, y2 = min(mx + src_r, w), min(my + src_r, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return

    crop = frame[y1:y2, x1:x2]
    side = LENS_RADIUS * 2
    zoomed = cv2.resize(crop, (side, side), interpolation=cv2.INTER_CUBIC)

    # Position: top-right
    cx = w - LENS_RADIUS - 10
    cy = LENS_RADIUS + 10
    dx1, dy1 = cx - LENS_RADIUS, cy - LENS_RADIUS
    dx2, dy2 = cx + LENS_RADIUS, cy + LENS_RADIUS
    if dx1 < 0 or dy1 < 0 or dx2 > w or dy2 > h:
        return

    # Circular mask
    mask = np.zeros((side, side), dtype=np.uint8)
    cv2.circle(mask, (LENS_RADIUS, LENS_RADIUS), LENS_RADIUS, 255, cv2.FILLED)

    roi = frame[dy1:dy2, dx1:dx2]
    np.copyto(roi, zoomed, where=(mask[:, :, None] > 0))

    # Thin crosshair
    cv2.line(frame, (cx - 6, cy), (cx + 6, cy), WHITE, 1)
    cv2.line(frame, (cx, cy - 6), (cx, cy + 6), WHITE, 1)

    # Subtle border — color matches current mode
    if state.calibrating:
        border = YELLOW
    elif state.mode == 'attacker':
        border = MAGENTA
    else:
        border = CYAN
    cv2.circle(frame, (cx, cy), LENS_RADIUS, border, 1, cv2.LINE_AA)


def draw_perspective_line(frame, point, state, color, thickness=2, scale=1):
    """Draw a line through `point` converging at the vanishing point."""
    h, w = frame.shape[:2]

    if state.vanishing_pt is not None:
        vp = state.vanishing_pt
        # VP is in original coords, scale it
        vpx, vpy = vp[0] * scale, vp[1] * scale
        dx = float(point[0] - vpx)
        dy = float(point[1] - vpy)
    elif state.line_direction is not None:
        dx, dy = state.line_direction
    else:
        return

    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return
    dx, dy = dx / length, dy / length

    ext = max(w, h) * 5
    p1 = (int(point[0] - ext * dx), int(point[1] - ext * dy))
    p2 = (int(point[0] + ext * dx), int(point[1] + ext * dy))
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def draw_dashed_line(frame, pt1, pt2, color, thickness=1, gap=4):
    """Draw a dashed line from pt1 to pt2."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = np.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return
    steps = int(dist / gap)
    for i in range(0, steps, 2):
        t1 = i / steps
        t2 = min((i + 1) / steps, 1.0)
        a = (int(pt1[0] + dx * t1), int(pt1[1] + dy * t1))
        b = (int(pt1[0] + dx * t2), int(pt1[1] + dy * t2))
        cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)


def _draw_extended_line(frame, pt1, pt2, color, thickness=2):
    """Draw a line through pt1 and pt2, extended across the full frame."""
    h, w = frame.shape[:2]
    dx = float(pt2[0] - pt1[0])
    dy = float(pt2[1] - pt1[1])
    length = np.sqrt(dx * dx + dy * dy)
    if length < 1e-6:
        return
    dx, dy = dx / length, dy / length
    ext = max(w, h) * 3
    p1 = (int(pt1[0] - ext * dx), int(pt1[1] - ext * dy))
    p2 = (int(pt1[0] + ext * dx), int(pt1[1] + ext * dy))
    cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)


def auto_ground(state, body_pt):
    """Estimate ground position below a body part using perspective scale.
    Uses goal-to-area distance (16.5m) as reference."""
    if len(state.calib_pts) < 4 or state.vanishing_pt is None:
        # Fallback: drop 20 pixels
        return (body_pt[0], body_pt[1] + 20)

    g1 = state.calib_pts[0]
    a1 = state.calib_pts[2]
    vp_y = state.vanishing_pt[1]

    # Reference: pixel distance G1→A1 represents 16.5m on the ground
    ref_gap = np.sqrt((a1[0] - g1[0]) ** 2 + (a1[1] - g1[1]) ** 2)
    ref_y = (g1[1] + a1[1]) / 2.0
    ref_dist = ref_y - vp_y

    # Scale at clicked depth (proportional to distance from horizon)
    click_dist = body_pt[1] - vp_y
    if ref_dist < 1 or click_dist < 1:
        return (body_pt[0], body_pt[1] + 20)

    ppm = (ref_gap / 16.5) * (click_dist / ref_dist)

    height_m = state.body_heights.get(state.body_part, 1.75)
    drop_px = int(height_m * ppm)
    return (body_pt[0], body_pt[1] + drop_px)


def render(state):
    """Render the current frame with all annotations."""
    # Upscale for better quality (640x360 → 1280x720)
    raw = state.frames[state.idx]
    frame = cv2.resize(raw, (raw.shape[1] * 2, raw.shape[0] * 2),
                       interpolation=cv2.INTER_LANCZOS4)
    # Sharpen
    blurred = cv2.GaussianBlur(frame, (0, 0), 2)
    frame = cv2.addWeighted(frame, 1.4, blurred, -0.4, 0)
    h, w = frame.shape[:2]

    # Scale helper for points (original coords → upscaled)
    def s(pt):
        return (pt[0] * 2, pt[1] * 2)

    # Calibration reference lines — extended across full frame
    if len(state.calib_pts) >= 2:
        _draw_extended_line(frame, s(state.calib_pts[0]), s(state.calib_pts[1]), YELLOW, 2)
    if len(state.calib_pts) >= 4:
        _draw_extended_line(frame, s(state.calib_pts[2]), s(state.calib_pts[3]), (0, 200, 200), 2)

    # Calibration dots
    for i, pt in enumerate(state.calib_pts):
        sp = s(pt)
        cv2.circle(frame, sp, 6, YELLOW, cv2.FILLED)
        cv2.circle(frame, sp, 6, BLACK, 1)
        label = ["G1", "G2", "A1", "A2"][i] if i < 4 else ""
        if label:
            cv2.putText(frame, label, (sp[0] + 8, sp[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)

    calibrated = state.vanishing_pt is not None or state.line_direction is not None

    # ── Offside lines through GROUND points ──
    if calibrated:
        if state.def_ground_pt is not None:
            draw_perspective_line(frame, s(state.def_ground_pt), state, CYAN, 1, scale=2)
        if state.atk_ground_pt is not None:
            draw_perspective_line(frame, s(state.atk_ground_pt), state, MAGENTA, 1, scale=2)

    # ── DEF markers (tiny) ──
    if state.def_body_pt is not None:
        sp = s(state.def_body_pt)
        cv2.circle(frame, sp, 3, CYAN, cv2.FILLED)
        cv2.putText(frame, "DEF", (sp[0] + 6, sp[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)
        if state.def_ground_pt is not None:
            sg = s(state.def_ground_pt)
            draw_dashed_line(frame, sp, sg, CYAN, 1)
            cv2.circle(frame, sg, 2, CYAN, cv2.FILLED)

    # ── ATK markers (tiny) ──
    if state.atk_body_pt is not None:
        sp = s(state.atk_body_pt)
        cv2.circle(frame, sp, 3, MAGENTA, cv2.FILLED)
        cv2.putText(frame, "ATK", (sp[0] + 6, sp[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, MAGENTA, 1)
        if state.atk_ground_pt is not None:
            sg = s(state.atk_ground_pt)
            draw_dashed_line(frame, sp, sg, MAGENTA, 1)
            cv2.circle(frame, sg, 2, MAGENTA, cv2.FILLED)

    # ── Verdict banner ──
    if state.verdict is not None:
        overlay = frame.copy()
        dist_str = ""
        if state.distance_cm is not None:
            dist_str = f" — {state.distance_cm:.0f} cm"
        if state.verdict == 'OFFSIDE':
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 180), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"FORA DE JOGO!{dist_str}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        else:
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 140, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"EM JOGO{dist_str}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    # ── Prompts ──
    if state.calibrating:
        prompts = [
            "CALIB: clica INICIO da LINHA DE BALIZA",
            "CALIB: clica FIM da LINHA DE BALIZA",
            "CALIB: clica INICIO da LINHA DA AREA",
            "CALIB: clica FIM da LINHA DA AREA",
        ]
        step = len(state.calib_pts)
        if step < 4:
            cv2.putText(frame, prompts[step], (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 2)
    elif not calibrated:
        cv2.putText(frame, "Prima C para calibrar (marcar linhas do campo)",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)

    # ── HUD ──
    mode_str = 'ATK [1]' if state.mode == 'attacker' else 'DEF [2]'
    part_labels = {'head': 'Cabeca', 'chest': 'Peito', 'knee': 'Joelho', 'foot': 'Bota'}
    part_str = part_labels.get(state.body_part, state.body_part)
    height_m = state.body_heights.get(state.body_part, 0)
    calib_str = 'OK' if calibrated else '--'
    cv2.putText(frame,
                f"Frame {state.idx}/{state.n - 1} ({state.idx / 25:.1f}s)"
                f"  |  {mode_str}  |  {part_str} ({height_m:.2f}m)"
                f"  |  Calib:{calib_str}",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 1)
    cv2.putText(frame,
                "C=calib  1=ATK  2=DEF  H/P/J/B=parte  SPACE=veredito  R=reset  RClick=chao",
                (10, h - 26), cv2.FONT_HERSHEY_SIMPLEX, 0.28, GRAY, 1)

    # ── Magnifier (always active) ──
    draw_magnifier(frame, state)

    return frame


# ── Calibration + Verdict logic ───────────────────────────────

def finish_calibration(state):
    """Compute vanishing point from 2 parallel field lines (4 clicks)."""
    g1, g2, a1, a2 = state.calib_pts[:4]

    l_goal = line_from_2pts(g1, g2)
    l_area = line_from_2pts(a1, a2)

    vp = intersect_lines(l_goal, l_area)

    if vp is not None:
        state.vanishing_pt = vp
        state.line_direction = None
        print(f"  Ponto de fuga: ({vp[0]:.0f}, {vp[1]:.0f})")
    else:
        d1 = (g2[0] - g1[0], g2[1] - g1[1])
        d2 = (a2[0] - a1[0], a2[1] - a1[1])
        state.line_direction = ((d1[0] + d2[0]) / 2, (d1[1] + d2[1]) / 2)
        state.vanishing_pt = None
        print("  Linhas paralelas — direcao media")

    state.goal_ref_pt = g1
    state.calibrating = False
    print("  Calibracao OK! Marca ATK (1) e DEF (2)")
    print("  Para cada: 1o clique=corpo, 2o clique=relvado abaixo")


def compute_real_distance(state):
    """Compute real distance in cm between ATK and DEF lines using cross-ratio.
    Goal line → area line = 16.5m is the reference."""
    if (state.atk_ground_pt is None or state.def_ground_pt is None
            or len(state.calib_pts) < 4):
        return None

    g1 = (float(state.calib_pts[0][0]), float(state.calib_pts[0][1]))
    g2 = (float(state.calib_pts[1][0]), float(state.calib_pts[1][1]))
    a1 = (float(state.calib_pts[2][0]), float(state.calib_pts[2][1]))
    a2 = (float(state.calib_pts[3][0]), float(state.calib_pts[3][1]))

    # VP_depth: where depth lines (G1→A1 and G2→A2) converge
    l_d1 = line_from_2pts(g1, a1)
    l_d2 = line_from_2pts(g2, a2)
    vp_depth = intersect_lines(l_d1, l_d2)
    if vp_depth is None:
        return None

    # Measurement line M = G1→A1 (depth direction on field)
    mx = a1[0] - g1[0]
    my = a1[1] - g1[1]
    m_len = np.sqrt(mx * mx + my * my)
    if m_len < 1e-6:
        return None
    mx, my = mx / m_len, my / m_len

    def t_on_m(pt):
        return (pt[0] - g1[0]) * mx + (pt[1] - g1[1]) * my

    t_a1 = t_on_m(a1)
    t_vp = t_on_m(vp_depth)

    # Build offside lines through ground points toward VP_horiz
    if state.vanishing_pt is not None:
        vp = state.vanishing_pt
        vf = (float(vp[0]), float(vp[1]))
    elif state.line_direction is not None:
        dx, dy = state.line_direction
        # Use a far point as proxy VP
        dp = state.def_ground_pt
        vf = (float(dp[0]) + dx * 100000, float(dp[1]) + dy * 100000)
    else:
        return None

    df = (float(state.def_ground_pt[0]), float(state.def_ground_pt[1]))
    af = (float(state.atk_ground_pt[0]), float(state.atk_ground_pt[1]))
    l_def = line_from_2pts(df, vf)
    l_atk = line_from_2pts(af, vf)

    # Intersect offside lines with measurement line M
    l_m = l_d1
    d_pt = intersect_lines(l_m, l_def)
    k_pt = intersect_lines(l_m, l_atk)
    if d_pt is None or k_pt is None:
        return None

    t_d = t_on_m(d_pt)
    t_k = t_on_m(k_pt)

    # Möbius transform: x(t) = coeff * t / (t - t_vp)
    # Maps G1→0m, A1→16.5m, VP_depth→∞
    if abs(t_a1) < 1e-6 or abs(t_d - t_vp) < 1e-6 or abs(t_k - t_vp) < 1e-6:
        return None

    coeff = 16.5 * (t_a1 - t_vp) / t_a1
    x_def = coeff * t_d / (t_d - t_vp)
    x_atk = coeff * t_k / (t_k - t_vp)

    return abs(x_atk - x_def) * 100  # cm


def compute_verdict(state):
    """Offside if ATK ground pt is on the goal side of DEF's line."""
    if state.atk_ground_pt is None or state.def_ground_pt is None:
        missing = []
        if state.atk_ground_pt is None:
            missing.append("ATK")
        if state.def_ground_pt is None:
            missing.append("DEF")
        print(f"  Falta marcar: {', '.join(missing)}")
        return
    if state.vanishing_pt is None and state.line_direction is None:
        print("  Calibra primeiro (C)")
        return

    # DEF offside line: through def_ground_pt toward VP
    if state.vanishing_pt is not None:
        lp1 = (float(state.def_ground_pt[0]), float(state.def_ground_pt[1]))
        lp2 = (float(state.vanishing_pt[0]), float(state.vanishing_pt[1]))
    else:
        dx, dy = state.line_direction
        lp1 = (float(state.def_ground_pt[0]), float(state.def_ground_pt[1]))
        lp2 = (lp1[0] + dx * 10000, lp1[1] + dy * 10000)

    goal_side = side_of_line(lp1, lp2, state.goal_ref_pt)
    atk_side = side_of_line(lp1, lp2,
                            (float(state.atk_ground_pt[0]),
                             float(state.atk_ground_pt[1])))

    if goal_side * atk_side > 0:
        state.verdict = 'OFFSIDE'
    else:
        state.verdict = 'ONSIDE'

    state.distance_cm = compute_real_distance(state)
    dist_str = f" ({state.distance_cm:.0f} cm)" if state.distance_cm is not None else ""
    print(f"  → {state.verdict}{dist_str}")


# ── Mouse callback ────────────────────────────────────────────

def on_mouse(event, x, y, flags, state):
    # Mouse coords come in upscaled (2x) space → convert to original
    ox, oy = x // 2, y // 2

    if event == cv2.EVENT_MOUSEMOVE:
        state.mouse_x = ox
        state.mouse_y = oy
        return

    # Calibration mode: left click only
    if state.calibrating and event == cv2.EVENT_LBUTTONDOWN:
        state.calib_pts.append((ox, oy))
        labels = ["G1 (inicio baliza)", "G2 (fim baliza)",
                  "A1 (inicio area)", "A2 (fim area)"]
        step = len(state.calib_pts) - 1
        if step < 4:
            print(f"  {labels[step]}: ({ox}, {oy})")
        if len(state.calib_pts) == 4:
            finish_calibration(state)
        return

    # Left click: set body part + auto-project to ground
    if event == cv2.EVENT_LBUTTONDOWN:
        ground = auto_ground(state, (ox, oy))
        part = state.body_part
        h_m = state.body_heights.get(part, 0)
        if state.mode == 'attacker':
            state.atk_body_pt = (ox, oy)
            state.atk_ground_pt = ground
            state.verdict = None
            print(f"  ATK [{part} {h_m}m]: ({ox},{oy}) → chao ({ground[0]},{ground[1]})")
        elif state.mode == 'defender':
            state.def_body_pt = (ox, oy)
            state.def_ground_pt = ground
            state.verdict = None
            print(f"  DEF [{part} {h_m}m]: ({ox},{oy}) → chao ({ground[0]},{ground[1]})")

    # Right click: manually adjust ground position
    elif event == cv2.EVENT_RBUTTONDOWN:
        if state.mode == 'attacker' and state.atk_body_pt is not None:
            state.atk_ground_pt = (ox, oy)
            state.verdict = None
            print(f"  ATK chao ajustado: ({ox},{oy})")
        elif state.mode == 'defender' and state.def_body_pt is not None:
            state.def_ground_pt = (ox, oy)
            state.verdict = None
            print(f"  DEF chao ajustado: ({ox},{oy})")


# ── Main loop ─────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Uso: python offside_tool.py <video> [frame_inicial]")
        sys.exit(1)

    video_path = sys.argv[1]
    start_frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print("A carregar video...")
    frames = read_video(video_path)
    print(f"  {len(frames)} frames ({len(frames) / 25:.1f}s a 25fps)")

    state = State(frames)
    state.idx = min(start_frame, state.n - 1)

    window = 'Offside Tool'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    cv2.setMouseCallback(window, on_mouse, state)

    print()
    print("  C = calibrar (2 linhas paralelas do campo)")
    print("  2 = modo DEF (comecar por aqui)    1 = modo ATK")
    print("  H = cabeca   P = peito   J = joelho   B = bota")
    print("  Clique esq = marca jogador (auto-projeta ao chao)")
    print("  Clique dir = ajustar posicao do chao manualmente")
    print("  SPACE = veredito   R = reset   ESC = desfazer")
    print()

    while True:
        frame = render(state)
        cv2.imshow(window, frame)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break
        elif key == 27:  # ESC = undo last click
            if state.calibrating:
                if state.calib_pts:
                    state.calib_pts.pop()
                    print(f"  Ultimo ponto removido ({len(state.calib_pts)}/4)")
                else:
                    state.calibrating = False
                    print("  Calibracao cancelada")
            else:
                # Clear current mode's marks
                if state.mode == 'attacker' and (state.atk_body_pt or state.atk_ground_pt):
                    state.atk_body_pt = None
                    state.atk_ground_pt = None
                    state.verdict = None
                    print("  ATK limpo")
                elif state.mode == 'defender' and (state.def_body_pt or state.def_ground_pt):
                    state.def_body_pt = None
                    state.def_ground_pt = None
                    state.verdict = None
                    print("  DEF limpo")
        elif key == ord('d') or key == 83:
            state.idx = min(state.idx + 1, state.n - 1)
        elif key == ord('a') or key == 81:
            state.idx = max(state.idx - 1, 0)
        elif key == ord('w') or key == 82:
            state.idx = min(state.idx + 25, state.n - 1)
        elif key == ord('s') or key == 84:
            state.idx = max(state.idx - 25, 0)
        elif key == ord('c'):
            state.calibrating = True
            state.calib_pts = []
            state.vanishing_pt = None
            state.line_direction = None
            print("  CALIBRACAO: 2 pts BALIZA + 2 pts AREA")
        elif key == ord('1'):
            state.mode = 'attacker'
            print("  Modo: ATK")
        elif key == ord('2'):
            state.mode = 'defender'
            print("  Modo: DEF")
        elif key == ord('h'):
            state.body_part = 'head'
            print("  Parte: Cabeca (1.75m)")
        elif key == ord('p'):
            state.body_part = 'chest'
            print("  Parte: Peito (1.30m)")
        elif key == ord('j'):
            state.body_part = 'knee'
            print("  Parte: Joelho (0.50m)")
        elif key == ord('b'):
            state.body_part = 'foot'
            print("  Parte: Bota (0.05m)")
        elif key == ord(' '):
            compute_verdict(state)
        elif key == ord('r'):
            state.atk_body_pt = None
            state.atk_ground_pt = None
            state.def_body_pt = None
            state.def_ground_pt = None
            state.verdict = None
            print("  Reset (calibracao mantida)")
        elif key == ord('e'):
            out = f'output_videos/offside_frame_{state.idx}.jpg'
            cv2.imwrite(out, render(state))
            print(f"  Exportado: {out}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
