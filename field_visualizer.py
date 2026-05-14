import cv2
import numpy as np

# Mini-map dimensions
FIELD_W = 800
FIELD_H = 500
MARGIN = 40
PLAYER_R = 8


class FieldVisualizer:
    def __init__(self, court_length=105, court_width=68):
        self.court_length = court_length
        self.court_width = court_width
        self.scale_x = (FIELD_W - 2 * MARGIN) / court_length
        self.scale_y = (FIELD_H - 2 * MARGIN) / court_width

    def _to_pixel(self, fx, fy):
        px = int(MARGIN + fx * self.scale_x)
        py = int(MARGIN + fy * self.scale_y)
        return px, py

    def _draw_pitch(self, img):
        # Background
        cv2.rectangle(img, (0, 0), (FIELD_W, FIELD_H), (34, 139, 34), cv2.FILLED)
        # Border
        cv2.rectangle(img, (MARGIN, MARGIN), (FIELD_W - MARGIN, FIELD_H - MARGIN), (255, 255, 255), 2)
        # Centre line
        cx = FIELD_W // 2
        cv2.line(img, (cx, MARGIN), (cx, FIELD_H - MARGIN), (255, 255, 255), 2)
        # Centre circle (~9.15 m radius)
        cv2.circle(img, (cx, FIELD_H // 2), int(9.15 * self.scale_x), (255, 255, 255), 2)
        # Penalty boxes
        pb_w = int(40.32 * self.scale_x)
        pb_h = int(16.5 * self.scale_y)
        cy = FIELD_H // 2
        cv2.rectangle(img, (MARGIN, cy - pb_h // 2), (MARGIN + pb_w, cy + pb_h // 2), (255, 255, 255), 2)
        cv2.rectangle(img,
                      (FIELD_W - MARGIN - pb_w, cy - pb_h // 2),
                      (FIELD_W - MARGIN, cy + pb_h // 2), (255, 255, 255), 2)
        return img

    def draw_field(self, frame, offside_data, tracks, frame_num):
        img = np.zeros((FIELD_H, FIELD_W, 3), dtype=np.uint8)
        img = self._draw_pitch(img)

        if offside_data:
            offside_line_x = offside_data.get('offside_line_x')
            is_offside = offside_data.get('is_offside', False)
            receiver_id = offside_data.get('receiver')
            defender_id = offside_data.get('key_defender_id')

            # Offside line
            if offside_line_x is not None:
                lx, _ = self._to_pixel(offside_line_x, 0)
                line_color = (0, 0, 255) if is_offside else (0, 255, 255)
                cv2.line(img, (lx, MARGIN), (lx, FIELD_H - MARGIN), line_color, 2)

            # All players as small dots for context
            for player_id, pos in offside_data.get('attackers', []):
                px, py = self._to_pixel(float(pos[0]), float(pos[1]))
                cv2.circle(img, (px, py), 4, (0, 120, 200), cv2.FILLED)

            for player_id, pos in offside_data.get('defenders', []):
                px, py = self._to_pixel(float(pos[0]), float(pos[1]))
                cv2.circle(img, (px, py), 4, (200, 80, 80), cv2.FILLED)

            # Highlight receiver (magenta ring)
            for player_id, pos in offside_data.get('attackers', []):
                if player_id == receiver_id:
                    px, py = self._to_pixel(float(pos[0]), float(pos[1]))
                    cv2.circle(img, (px, py), PLAYER_R + 4, (255, 0, 255), 3)
                    inner = (0, 0, 255) if is_offside else (0, 255, 0)
                    cv2.circle(img, (px, py), PLAYER_R, inner, cv2.FILLED)
                    label = "OFF" if is_offside else "ON"
                    cv2.putText(img, label, (px - 10, py - PLAYER_R - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Highlight key defender (cyan ring)
            for player_id, pos in offside_data.get('defenders', []):
                if player_id == defender_id:
                    px, py = self._to_pixel(float(pos[0]), float(pos[1]))
                    cv2.circle(img, (px, py), PLAYER_R + 4, (255, 255, 0), 3)
                    cv2.circle(img, (px, py), PLAYER_R, (255, 255, 0), cv2.FILLED)
                    cv2.putText(img, "DEF", (px - 12, py - PLAYER_R - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Overlay mini-map on top-right (below verdict banner)
        fh, fw = frame.shape[:2]
        new_w = fw // 3
        new_h = int(FIELD_H * new_w / FIELD_W)
        resized = cv2.resize(img, (new_w, new_h))
        x_off = fw - new_w - 10
        y_off = 50  # below the 40px banner
        frame[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return frame
