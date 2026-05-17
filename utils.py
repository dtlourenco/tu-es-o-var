import cv2


class LazyVideo:
    """Read frames on demand instead of loading all into RAM."""

    def __init__(self, path):
        self.path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"Cannot open {path}")
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self._cache_idx = -1
        self._cache_frame = None
        print(f"  [LazyVideo] {path}: {self.n} frames")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if idx == self._cache_idx and self._cache_frame is not None:
            return self._cache_frame
        cap = cv2.VideoCapture(self.path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise IndexError(f"Frame {idx} not readable")
        self._cache_idx = idx
        self._cache_frame = frame
        return frame


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
