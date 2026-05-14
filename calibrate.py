"""
Calibration Helper — Pick perspective points from a video
==========================================================
Opens a frame from the video and lets you click 4 field corners.
Prints the PIXEL_VERTICES array to paste into view_transformer.py.

Usage:
    python calibrate.py                                  # frame 0 of default video
    python calibrate.py path/to/video.mp4                # frame 0 of custom video
    python calibrate.py path/to/video.mp4 20             # frame at second 20
"""

import sys
import cv2
import numpy as np

DEFAULT_VIDEO = 'input_videos/clip.mp4'


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO
    seek_sec = float(sys.argv[2]) if len(sys.argv) > 2 else 0

    cap = cv2.VideoCapture(video_path)
    if seek_sec > 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(seek_sec * fps))
        print(f"ℹ️  Seeking to second {seek_sec} (frame {int(seek_sec * fps)})")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Could not open video: {video_path}")
        print("   Make sure the file exists in input_videos/")
        sys.exit(1)

    # Resize for display if too large
    h, w = frame.shape[:2]
    scale = 1.0
    if w > 1600:
        scale = 1600 / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        print(f"ℹ️  Frame resized by {scale:.2f}x for display. Coordinates will be scaled back.")

    points = []
    labels = ['Bottom-Left', 'Top-Left', 'Top-Right', 'Bottom-Right']
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            idx = len(points) - 1
            cv2.circle(frame, (x, y), 8, colors[idx], -1)
            cv2.putText(frame, f"{labels[idx]} ({x},{y})", (x + 12, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[idx], 2)

            # Draw connecting lines
            if len(points) > 1:
                cv2.line(frame, points[-2], points[-1], (255, 255, 255), 1)
            if len(points) == 4:
                cv2.line(frame, points[3], points[0], (255, 255, 255), 1)

            cv2.imshow("Calibration — Click 4 Field Corners", frame)

            if len(points) == 4:
                print("\n" + "=" * 50)
                print("✅ 4 points selected! Copy this into view_transformer.py:\n")
                print("PIXEL_VERTICES = np.array([")
                for i, (px, py) in enumerate(points):
                    # Scale back to original resolution
                    ox = int(px / scale)
                    oy = int(py / scale)
                    print(f"    [{ox}, {oy}],   # {labels[i]}")
                print("], dtype=np.float32)")
                print("\n" + "=" * 50)
                print("\nPress any key to close the window.")

    window_name = "Calibration — Click 4 Field Corners"
    cv2.imshow(window_name, frame)
    cv2.setMouseCallback(window_name, on_click)

    print("\n🎯 CALIBRATION MODE")
    print("=" * 50)
    print("Click 4 visible field corners in this order:")
    print("  1. Bottom-Left  (RED)")
    print("  2. Top-Left     (GREEN)")
    print("  3. Top-Right    (BLUE)")
    print("  4. Bottom-Right (YELLOW)")
    print("=" * 50)
    print("Waiting for clicks...\n")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
