"""
Tu és o VAR! — Web Frontend (multi-user)
=============================================
Each browser gets an isolated session (state + video folder).
Sessions expire after 30 min of inactivity.

Run:  python app.py
Open: http://localhost:5000
"""

import os
import shutil
import time
import uuid
import cv2
import numpy as np
from flask import (Flask, render_template, request, jsonify,
                   Response, make_response)
from werkzeug.utils import secure_filename

from offside_tool import (
    State, finish_calibration, auto_ground, compute_verdict,
    draw_perspective_line, draw_dashed_line, _draw_extended_line,
    MAGENTA, CYAN, YELLOW, WHITE, BLACK,
)
from utils import read_video, LazyVideo

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB

INPUT_DIR = 'input_videos'
OUTPUT_DIR = 'output_videos'
MAX_VIDEO_MB = 100
SESSION_TIMEOUT = 30 * 60  # 30 min
MAX_SESSIONS = 20
IS_PROD = os.environ.get('RENDER') or os.environ.get('FLY_APP_NAME') or os.environ.get('ONRENDER')

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Session store ─────────────────────────────────────────────
_sessions = {}   # sid → {state, video_path, last_access}
_last_gc = 0.0


def _sdir(sid):
    return os.path.join(INPUT_DIR, sid)


def _gc():
    """Evict sessions idle > SESSION_TIMEOUT (runs at most every 30 s)."""
    global _last_gc
    now = time.time()
    if now - _last_gc < 30:
        return
    _last_gc = now
    stale = [s for s, v in _sessions.items()
             if now - v['last_access'] > SESSION_TIMEOUT]
    for sid in stale:
        d = _sdir(sid)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        _sessions.pop(sid, None)
        print(f"  [gc] sessao {sid} expirou")


def _get_sess():
    """Return (sid, session_dict).  Creates a new session if needed."""
    _gc()
    sid = request.cookies.get('sid')
    if sid and sid in _sessions:
        _sessions[sid]['last_access'] = time.time()
        return sid, _sessions[sid]
    # Evict oldest if at capacity
    if len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k]['last_access'])
        d = _sdir(oldest)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
        _sessions.pop(oldest, None)
    if not sid:
        sid = uuid.uuid4().hex[:10]
    _sessions[sid] = {'state': None, 'video_path': None,
                      'last_access': time.time()}
    os.makedirs(_sdir(sid), exist_ok=True)
    # Try to auto-recover: reload video from cookie hint
    vp = request.cookies.get('vp')
    if vp and os.path.isfile(vp):
        _try_reload(sid, vp)
    return sid, _sessions[sid]


def _try_reload(sid, video_path):
    """Try to reload a video into the session (recovery after restart)."""
    try:
        lv = LazyVideo(video_path)
        _sessions[sid]['state'] = State(lv)
        _sessions[sid]['video_path'] = video_path
        print(f"  [recover] {sid} reloaded {video_path} ({lv.n} frames)")
    except Exception as e:
        print(f"  [recover] failed: {e}")


def _set_cookie(resp, sid):
    """Set session cookie on response."""
    resp.set_cookie('sid', sid, max_age=SESSION_TIMEOUT,
                    httponly=True, samesite='Lax',
                    secure=bool(IS_PROD))
    return resp


def _resp(data, sid):
    """JSON response with session cookie."""
    r = make_response(jsonify(data))
    return _set_cookie(r, sid)


def _resp_bytes(raw, mime, sid):
    r = make_response(Response(raw, mimetype=mime))
    return _set_cookie(r, sid)
    return r


def _cleanup_sdir(sid, keep=None):
    d = _sdir(sid)
    if not os.path.isdir(d):
        return
    exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.part', '.temp')
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if fp == keep:
            continue
        if f.lower().endswith(exts) and os.path.isfile(fp):
            try:
                os.remove(fp)
            except OSError:
                pass


def _check_size(path):
    mb = os.path.getsize(path) / (1024 * 1024)
    if mb > MAX_VIDEO_MB:
        try:
            os.remove(path)
        except OSError:
            pass
        return f'Video demasiado grande ({mb:.0f} MB). Max: {MAX_VIDEO_MB} MB'
    return None


# ── Web-specific render (no magnifier, no HUD) ───────────────

def render_web(st):
    """Render annotated frame for the web UI."""
    raw = st.frames[st.idx]
    frame = cv2.resize(raw, (raw.shape[1] * 2, raw.shape[0] * 2),
                       interpolation=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(frame, (0, 0), 2)
    frame = cv2.addWeighted(frame, 1.4, blurred, -0.4, 0)
    h, w = frame.shape[:2]

    def s(pt):
        return (pt[0] * 2, pt[1] * 2)

    # Calibration lines
    if len(st.calib_pts) >= 2:
        _draw_extended_line(frame, s(st.calib_pts[0]), s(st.calib_pts[1]),
                            YELLOW, 2)
    if len(st.calib_pts) >= 4:
        _draw_extended_line(frame, s(st.calib_pts[2]), s(st.calib_pts[3]),
                            (0, 200, 200), 2)

    # Calibration dots
    for i, pt in enumerate(st.calib_pts):
        sp = s(pt)
        cv2.circle(frame, sp, 6, YELLOW, cv2.FILLED)
        cv2.circle(frame, sp, 6, BLACK, 1)
        label = ["G1", "G2", "A1", "A2"][i] if i < 4 else ""
        if label:
            cv2.putText(frame, label, (sp[0] + 8, sp[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)

    calibrated = st.vanishing_pt is not None or st.line_direction is not None

    # Offside lines
    if calibrated:
        if st.def_ground_pt is not None:
            draw_perspective_line(frame, s(st.def_ground_pt), st,
                                  CYAN, 1, scale=2)
        if st.atk_ground_pt is not None:
            draw_perspective_line(frame, s(st.atk_ground_pt), st,
                                  MAGENTA, 1, scale=2)

    # DEF markers
    if st.def_body_pt is not None:
        sp = s(st.def_body_pt)
        cv2.circle(frame, sp, 3, CYAN, cv2.FILLED)
        cv2.putText(frame, "DEF", (sp[0] + 6, sp[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1)
        if st.def_ground_pt is not None:
            sg = s(st.def_ground_pt)
            draw_dashed_line(frame, sp, sg, CYAN, 1)
            cv2.circle(frame, sg, 2, CYAN, cv2.FILLED)

    # ATK markers
    if st.atk_body_pt is not None:
        sp = s(st.atk_body_pt)
        cv2.circle(frame, sp, 3, MAGENTA, cv2.FILLED)
        cv2.putText(frame, "ATK", (sp[0] + 6, sp[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, MAGENTA, 1)
        if st.atk_ground_pt is not None:
            sg = s(st.atk_ground_pt)
            draw_dashed_line(frame, sp, sg, MAGENTA, 1)
            cv2.circle(frame, sg, 2, MAGENTA, cv2.FILLED)

    # Verdict banner
    if st.verdict is not None:
        overlay = frame.copy()
        dist_str = ""
        if st.distance_cm is not None:
            dist_str = f" — {st.distance_cm:.0f} cm"
        if st.verdict == 'OFFSIDE':
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 180), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"FORA DE JOGO!{dist_str}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
        else:
            cv2.rectangle(overlay, (0, 0), (w, 40), (0, 140, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"EM JOGO{dist_str}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)

    # Calibration prompt
    if st.calibrating:
        prompts = [
            "CALIB: clica INICIO da LINHA DE BALIZA",
            "CALIB: clica FIM da LINHA DE BALIZA",
            "CALIB: clica INICIO da LINHA DA AREA",
            "CALIB: clica FIM da LINHA DA AREA",
        ]
        step = len(st.calib_pts)
        if step < 4:
            cv2.putText(frame, prompts[step], (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 2)

    return frame


def frame_to_jpeg(frame, quality=92):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _state_dict(st):
    """Build JSON-friendly state dict from a State object."""
    if st is None:
        return {'loaded': False}
    calibrated = (st.vanishing_pt is not None
                  or st.line_direction is not None)
    return {
        'loaded': True,
        'n_frames': st.n,
        'current_frame': st.idx,
        'fps': 25,
        'frame_w': st.frames[0].shape[1] * 2,
        'frame_h': st.frames[0].shape[0] * 2,
        'calibrating': st.calibrating,
        'calib_points': len(st.calib_pts),
        'calibrated': calibrated,
        'mode': st.mode,
        'body_part': st.body_part,
        'body_heights': st.body_heights,
        'def_marked': st.def_body_pt is not None,
        'atk_marked': st.atk_body_pt is not None,
        'verdict': st.verdict,
        'distance_cm': (round(st.distance_cm, 1)
                        if st.distance_cm is not None else None),
    }


# ── Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    sid, _ = _get_sess()
    r = make_response(render_template('index.html'))
    return _set_cookie(r, sid)


@app.route('/videos')
def list_videos():
    sid, sess = _get_sess()
    videos = []
    # Show shared videos from INPUT_DIR root
    if os.path.isdir(INPUT_DIR):
        for f in sorted(os.listdir(INPUT_DIR)):
            fp = os.path.join(INPUT_DIR, f)
            if os.path.isfile(fp) and f.lower().endswith(
                    ('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                videos.append(f)
    # Also show session-local videos
    sd = _sdir(sid)
    if os.path.isdir(sd):
        for f in sorted(os.listdir(sd)):
            if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                videos.append(f'{sid}/{f}')
    return _resp(videos, sid)


@app.route('/download_url', methods=['POST'])
def download_url():
    sid, sess = _get_sess()
    data = request.json
    url = data.get('url', '').strip()
    if not url:
        return _resp({'error': 'URL vazio'}, sid), 400

    sd = _sdir(sid)
    os.makedirs(sd, exist_ok=True)

    try:
        import yt_dlp
    except ImportError:
        return _resp({'error': 'yt-dlp nao instalado'}, sid), 500

    # Use a safe ASCII filename to avoid OpenCV issues on Windows
    safe_name = f"dl_{int(time.time())}"
    outtmpl = os.path.join(sd, f'{safe_name}.%(ext)s')
    ydl_opts = {
        'outtmpl': outtmpl,
        'format': 'best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'quiet': False,
        'no_warnings': True,
        'overwrites': True,
        'max_filesize': MAX_VIDEO_MB * 1024 * 1024,
    }
    # Use cookies if available (needed for YouTube)
    cpath = _cookies_path(sid)
    if os.path.isfile(cpath):
        ydl_opts['cookiefile'] = cpath

    display_name = ''
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            display_name = info.get('title', safe_name)
            path = None
            if info.get('requested_downloads'):
                path = info['requested_downloads'][0]['filepath']
            if not path or not os.path.exists(path):
                # Fallback: most recent file in session dir
                files = []
                for f in os.listdir(sd):
                    fp = os.path.join(sd, f)
                    if os.path.isfile(fp) and f.lower().endswith(
                            ('.mp4', '.mkv', '.webm', '.avi')):
                        files.append((os.path.getmtime(fp), fp))
                if files:
                    files.sort(reverse=True)
                    path = files[0][1]
    except Exception as e:
        msg = str(e)
        if 'Unsupported URL' in msg:
            return _resp({
                'error': 'Site nao suportado. Tenta YouTube, Vimeo, '
                         'Dailymotion, TVI, vsports.pt, ou faz upload direto.'
            }, sid), 400
        return _resp({'error': f'Falha no download: {msg[:200]}'}, sid), 500

    if not path or not os.path.exists(path):
        return _resp({'error': 'Ficheiro nao encontrado apos download'}, sid), 500

    # Rename to guaranteed-safe ASCII path if needed
    safe_path = os.path.join(sd, f'{safe_name}.mp4')
    if path != safe_path and os.path.exists(path):
        try:
            os.rename(path, safe_path)
            path = safe_path
        except OSError:
            pass  # keep original path

    sz = os.path.getsize(path) / (1024 * 1024)
    print(f"  [{sid}] download: {path} ({sz:.1f} MB)")

    err = _check_size(path)
    if err:
        return _resp({'error': err}, sid), 400

    _cleanup_sdir(sid, keep=path)

    try:
        lv = LazyVideo(path)
    except Exception as e:
        return _resp({'error': f'Falha ao carregar video: {e}'}, sid), 500

    sess['state'] = State(lv)
    sess['video_path'] = path
    result = _state_dict(sess['state'])
    result['filename'] = display_name or os.path.basename(path)
    r = _resp(result, sid)
    r.set_cookie('vp', path, max_age=SESSION_TIMEOUT,
                 httponly=True, samesite='Lax', secure=bool(IS_PROD))
    return r


@app.route('/load', methods=['POST'])
def load_video():
    sid, sess = _get_sess()

    if 'file' in request.files:
        f = request.files['file']
        if f.filename:
            fname = secure_filename(f.filename)
            sd = _sdir(sid)
            os.makedirs(sd, exist_ok=True)
            path = os.path.join(sd, fname)
            f.save(path)
            err = _check_size(path)
            if err:
                return _resp({'error': err}, sid), 400
        else:
            return _resp({'error': 'Ficheiro vazio'}, sid), 400
    elif request.is_json and 'name' in request.json:
        fname = request.json['name']
        path = os.path.normpath(os.path.join(INPUT_DIR, fname))
    else:
        return _resp({'error': 'Nenhum video especificado'}, sid), 400

    if not os.path.exists(path):
        return _resp({'error': f'Video nao encontrado: {fname}'}, sid), 404

    _cleanup_sdir(sid, keep=path)

    try:
        lv = LazyVideo(path)
    except Exception as e:
        return _resp({'error': f'Falha ao carregar video: {e}'}, sid), 500

    sess['state'] = State(lv)
    sess['video_path'] = path
    result = _state_dict(sess['state'])
    r = _resp(result, sid)
    r.set_cookie('vp', path, max_age=SESSION_TIMEOUT,
                 httponly=True, samesite='Lax', secure=bool(IS_PROD))
    return r


@app.route('/frame')
def get_frame():
    sid, sess = _get_sess()
    st = sess['state']
    if st is None:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(blank, "Carrega um video para comecar",
                    (320, 360), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (80, 80, 80), 2)
        return _resp_bytes(frame_to_jpeg(blank), 'image/jpeg', sid)
    frame = render_web(st)
    return _resp_bytes(frame_to_jpeg(frame), 'image/jpeg', sid)


@app.route('/state')
def get_state():
    sid, sess = _get_sess()
    return _resp(_state_dict(sess['state']), sid)


@app.route('/action', methods=['POST'])
def action():
    sid, sess = _get_sess()
    st = sess['state']
    if st is None:
        print(f"  [action] sid={sid} state=None sessions={list(_sessions.keys())}")
        return _resp({'error': 'Nenhum video carregado'}, sid), 400

    data = request.json
    act = data.get('action')

    if act == 'set_frame':
        idx = int(data['idx'])
        st.idx = max(0, min(idx, st.n - 1))

    elif act == 'calibrate':
        st.calibrating = True
        st.calib_pts = []
        st.vanishing_pt = None
        st.line_direction = None

    elif act == 'click':
        ox = int(data['x'])
        oy = int(data['y'])
        button = data.get('button', 'left')

        if st.calibrating and button == 'left':
            st.calib_pts.append((ox, oy))
            if len(st.calib_pts) == 4:
                finish_calibration(st)

        elif button == 'left':
            ground = auto_ground(st, (ox, oy))
            if st.mode == 'attacker':
                st.atk_body_pt = (ox, oy)
                st.atk_ground_pt = ground
                st.verdict = None
            elif st.mode == 'defender':
                st.def_body_pt = (ox, oy)
                st.def_ground_pt = ground
                st.verdict = None

        elif button == 'right':
            if st.mode == 'attacker' and st.atk_body_pt is not None:
                st.atk_ground_pt = (ox, oy)
                st.verdict = None
            elif st.mode == 'defender' and st.def_body_pt is not None:
                st.def_ground_pt = (ox, oy)
                st.verdict = None

    elif act == 'set_mode':
        st.mode = data['mode']

    elif act == 'set_body_part':
        st.body_part = data['part']

    elif act == 'set_heights':
        for k, v in data['heights'].items():
            if k in st.body_heights:
                try:
                    st.body_heights[k] = float(v)
                except (ValueError, TypeError):
                    pass

    elif act == 'verdict':
        compute_verdict(st)

    elif act == 'reset':
        st.atk_body_pt = None
        st.atk_ground_pt = None
        st.def_body_pt = None
        st.def_ground_pt = None
        st.verdict = None
        st.distance_cm = None

    elif act == 'undo':
        if st.calibrating:
            if st.calib_pts:
                st.calib_pts.pop()
            else:
                st.calibrating = False
        else:
            if st.mode == 'attacker' and (st.atk_body_pt or st.atk_ground_pt):
                st.atk_body_pt = None
                st.atk_ground_pt = None
                st.verdict = None
            elif st.mode == 'defender' and (st.def_body_pt or st.def_ground_pt):
                st.def_body_pt = None
                st.def_ground_pt = None
                st.verdict = None

    elif act == 'export':
        frame = render_web(st)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f'offside_{sid}_{st.idx}.jpg')
        cv2.imwrite(path, frame)

    return _resp(_state_dict(st), sid)


# ── Cookies for YouTube ──────────────────────────────────────

def _cookies_path(sid):
    return os.path.join(_sdir(sid), 'cookies.txt')


@app.route('/has_cookies')
def has_cookies():
    sid, _ = _get_sess()
    ok = os.path.isfile(_cookies_path(sid))
    return _resp({'ok': ok}, sid)


@app.route('/upload_cookies', methods=['POST'])
def upload_cookies():
    sid, _ = _get_sess()
    if 'file' not in request.files:
        return _resp({'error': 'Sem ficheiro'}, sid), 400
    f = request.files['file']
    sd = _sdir(sid)
    os.makedirs(sd, exist_ok=True)
    f.save(_cookies_path(sid))
    return _resp({'ok': True}, sid)


if __name__ == '__main__':
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('RENDER') is None  # debug only locally
    print(f"\n  ⚽ Tu és o VAR! — http://localhost:{port}\n")
    app.run(debug=debug, host='0.0.0.0', port=port)
