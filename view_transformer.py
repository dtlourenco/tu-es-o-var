import numpy as np
import cv2

# Standard pitch dimensions (meters)
COURT_LENGTH = 105
COURT_WIDTH = 68

# Pixel coordinates of 4 visible field reference points in the video frame.
# Calibrated from sec 20 of offside_clip.mp4 (SLB vs BRA, Estádio da Luz).
# Points: penalty box edge + halfway line intersections.
# ⚠️ ADJUST THESE to match your specific video using calibrate.py
PIXEL_VERTICES = np.array([
    [105, 348],    # penalty box line meets touchline (bottom-left)
    [215, 170],    # penalty box top corner (top-left)
    [530, 155],    # halfway line near circle (top-right)
    [640, 285],    # halfway line meets touchline (bottom-right)
], dtype=np.float32)

# Corresponding field coordinates in metres (bird's eye view)
# Penalty box edge is at x=16.5m from goal line
# Halfway line is at x=52.5m
# Touchline bottom = y=68, top = y=0
# Using visible lateral span: ~y=20 (far side) to y=68 (near touchline)
TARGET_VERTICES = np.array([
    [16.5, 68],    # penalty box edge at near touchline
    [16.5, 20],    # penalty box edge at far visible side
    [52.5, 20],    # halfway line at far visible side
    [52.5, 68],    # halfway line at near touchline
], dtype=np.float32)


class ViewTransformer:
    def __init__(self, pixel_vertices=PIXEL_VERTICES, target_vertices=TARGET_VERTICES):
        self.pixel_vertices = pixel_vertices
        self.target_vertices = target_vertices
        self.H, _ = cv2.findHomography(pixel_vertices, target_vertices)
        self.H_inv, _ = cv2.findHomography(target_vertices, pixel_vertices)

    def transform_point(self, points):
        """Transform pixel coordinates → field 2D coordinates (meters)."""
        if points is None or len(points) == 0:
            return None
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H)
        return transformed.reshape(-1, 2)

    def inverse_transform_point(self, points):
        """Transform field 2D coordinates (meters) → pixel coordinates."""
        if points is None or len(points) == 0:
            return None
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.H_inv)
        return transformed.reshape(-1, 2)

    def inverse_transform_x(self, field_x):
        """
        Convert a field X coordinate (meters) to frame pixel X.
        Uses the vertical midline (Y = COURT_WIDTH/2) as reference row.
        """
        if field_x is None:
            return None
        mid_y = COURT_WIDTH / 2
        field_point = np.array([[[float(field_x), mid_y]]], dtype=np.float32)
        pixel_point = cv2.perspectiveTransform(field_point, self.H_inv)
        return float(pixel_point[0][0][0])

    def add_transformed_positions(self, tracks):
        """Enrich all player/ball tracks with their field 2D position."""
        for obj, obj_tracks in tracks.items():
            for frame_num, track in enumerate(obj_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info.get('bbox')
                    if bbox is None:
                        continue
                    foot_x = (bbox[0] + bbox[2]) / 2
                    foot_y = bbox[3]
                    pos = self.transform_point(np.array([[foot_x, foot_y]]))
                    if pos is not None:
                        tracks[obj][frame_num][track_id]['position_transformed'] = pos[0].tolist()
        return tracks


# ---------------------------------------------------------------------------
# Helper: pick perspective points interactively from your video
# ---------------------------------------------------------------------------
def pick_perspective_points(video_path, n_points=4):
    """
    Opens the first frame of a video so you can click 4 corner points.
    Run once to get PIXEL_VERTICES for your specific camera angle.

    Usage:
        python -c "from view_transformer import pick_perspective_points; \
                   pick_perspective_points('input_videos/clip.mp4')"
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read video.")
        return

    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n_points:
            points.append((x, y))
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Pick 4 Field Corners", frame)
            if len(points) == n_points:
                print("\nPIXEL_VERTICES = np.array([")
                for p in points:
                    print(f"    {list(p)},")
                print("], dtype=np.float32)")

    cv2.imshow("Pick 4 Field Corners", frame)
    cv2.setMouseCallback("Pick 4 Field Corners", on_click)
    print("Click the 4 field corners in order: bottom-left, top-left, top-right, bottom-right")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return points
