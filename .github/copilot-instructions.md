# Copilot Instructions — Football Offside Detector

## Project Goal
Semi-automatic offside detection from football clips or live streams.
Based on **Soccer-Analytics** (YOLOv8 + ByteTrack) with an offside module
inspired by **SoccerOffsideTracker**.

## Architecture
| File | Role |
|---|---|
| `main.py` | Full pipeline orchestrator |
| `tracker.py` | YOLOv8 detection + ByteTrack multi-object tracking |
| `team_assigner.py` | KMeans jersey-colour clustering → team separation |
| `view_transformer.py` | OpenCV homography: frame pixels ↔ field metres |
| `offside_detector.py` | Second-last-defender offside rule, per frame |
| `field_visualizer.py` | Bird's-eye 2D mini-map overlay |
| `utils.py` | Video I/O helpers |

## Data Conventions
- **`tracks`** is a dict with keys `'players'`, `'referees'`, `'ball'`.
  Each value is a list (one entry per frame) of dicts keyed by `track_id`.
  Each player entry: `{'bbox': [x1,y1,x2,y2], 'team': int, 'team_color': tuple}`.
- **`team_ball_control`** is a `np.ndarray` of shape `(n_frames,)` with values 1 or 2.
- **`offside_results`** is a dict keyed by `frame_num`:
  ```python
  {
    'offside_line_x': float,          # field X in metres
    'offside_players': [track_id, …],
    'attackers': [(id, [x,y]), …],
    'defenders': [(id, [x,y]), …],
  }
  ```
- Field coordinates: **X** = along the pitch (0–105 m), **Y** = across (0–68 m).
- `view_transformer.transform_point(pts)` takes pixel `(x,y)` → field `(x,y)`.
- `view_transformer.inverse_transform_x(field_x)` returns pixel X for an offside line.

## Key Extension Points
- **Pose estimation**: add MediaPipe Pose in `tracker.py` to use exact body-part positions.
- **Automatic line detection**: detect pitch lines in `view_transformer.py` using Hough transform to auto-calibrate `PIXEL_VERTICES`.
- **Pass detection**: analyse ball trajectory change in `tracker.py`; freeze offside check at the pass frame.
- **Web UI**: wrap `main.py` with FastAPI + a React frontend; stream frames via WebSocket.

## Model
- Place `best.pt` (YOLOv8, trained on football) in `models/`.
- Recommended source: Roboflow Universe → search "football players detection".

## Camera Calibration
Edit `PIXEL_VERTICES` in `view_transformer.py` to match your video:
```
python -c "from view_transformer import pick_perspective_points; pick_perspective_points('input_videos/clip.mp4')"
```
Click the 4 visible field corners in order: bottom-left, top-left, top-right, bottom-right.
