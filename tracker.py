import os
import pickle
import numpy as np
import cv2
import pandas as pd
from ultralytics import YOLO
import supervision as sv


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect_frames(self, frames, batch_size=20):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections.extend(batch)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)
        tracks = {'players': [], 'referees': [], 'ball': []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            det_sv = sv.Detections.from_ultralytics(detection)

            # Treat goalkeepers as players
            for i, cls_id in enumerate(det_sv.class_id):
                if cls_names[cls_id] == 'goalkeeper':
                    det_sv.class_id[i] = cls_names_inv.get('player', cls_id)

            det_with_tracks = self.tracker.update_with_detections(det_sv)

            tracks['players'].append({})
            tracks['referees'].append({})
            tracks['ball'].append({})

            for fd in det_with_tracks:
                bbox = fd[0].tolist()
                cls_id = fd[3]
                track_id = fd[4]
                label = cls_names[cls_id]
                if label == 'player':
                    tracks['players'][frame_num][track_id] = {'bbox': bbox}
                elif label == 'referee':
                    tracks['referees'][frame_num][track_id] = {'bbox': bbox}

            # Ball has no persistent ID — use detection directly
            for fd in det_sv:
                bbox = fd[0].tolist()
                cls_id = fd[3]
                if cls_names[cls_id] == 'ball':
                    tracks['ball'][frame_num][1] = {'bbox': bbox}

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    # ------------------------------------------------------------------
    # Ball interpolation
    # ------------------------------------------------------------------

    def interpolate_ball_positions(self, ball_positions):
        raw = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df = pd.DataFrame(raw, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill()
        return [{1: {'bbox': row}} for row in df.to_numpy().tolist()]

    # ------------------------------------------------------------------
    # Ball possession
    # ------------------------------------------------------------------

    def assign_ball_to_player(self, players, ball_bbox):
        if ball_bbox is None:
            return -1
        bx, by = (ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2
        ball_pos = np.array([bx, by])
        min_dist = float('inf')
        assigned = -1
        for player_id, player in players.items():
            pb = player['bbox']
            px, py = (pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2
            dist = np.linalg.norm(ball_pos - np.array([px, py]))
            if dist < 70 and dist < min_dist:
                min_dist = dist
                assigned = player_id
        return assigned

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center = int((bbox[0] + bbox[2]) / 2)
        width = int(bbox[2] - bbox[0])
        cv2.ellipse(frame, (x_center, y2), (width // 2, int(0.35 * width)),
                    0.0, -45, 235, color, 2, cv2.LINE_4)
        if track_id is not None:
            rw, rh = 40, 20
            x1 = x_center - rw // 2
            y1_r = y2 + 5
            cv2.rectangle(frame, (x1, y1_r), (x1 + rw, y1_r + rh), color, cv2.FILLED)
            cv2.putText(frame, str(track_id), (x1 + 8, y1_r + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame

    def _draw_triangle(self, frame, bbox, color):
        y1 = int(bbox[1])
        x_c = int((bbox[0] + bbox[2]) / 2)
        pts = np.array([[x_c, y1], [x_c - 10, y1 - 20], [x_c + 10, y1 - 20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame

    def _draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        n = frame_num + 1
        t1 = np.sum(team_ball_control[:n] == 1) / n
        t2 = np.sum(team_ball_control[:n] == 2) / n
        cv2.putText(frame, f"Team 1 Ball: {t1 * 100:.0f}%", (1360, 900),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball: {t2 * 100:.0f}%", (1360, 950),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 3)
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            # Only coloured ellipses — no IDs (offside overlay handles key players)
            for track_id, player in tracks['players'][frame_num].items():
                color = player.get('team_color', (0, 0, 255))
                frame = self._draw_ellipse(frame, player['bbox'], color)
            for _, ball in tracks['ball'][frame_num].items():
                frame = self._draw_triangle(frame, ball['bbox'], (0, 255, 0))
            output_frames.append(frame)
        return output_frames
