import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [read_video] ERRO: nao conseguiu abrir {video_path}")
        return []
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(f"  [read_video] ERRO: 0 frames lidos de {video_path}")
    else:
        print(f"  [read_video] OK: {len(frames)} frames de {video_path}")
    return frames


def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        return
    h, w = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (w, h))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
