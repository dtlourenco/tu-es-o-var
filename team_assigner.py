import numpy as np
import cv2
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def _get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1, random_state=42)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        top_half = image[:image.shape[0] // 2, :]
        kmeans = self._get_clustering_model(top_half)
        labels = kmeans.labels_.reshape(top_half.shape[:2])
        corner_labels = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corner_labels), key=corner_labels.count)
        player_cluster = 1 - non_player_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frames, all_player_tracks):
        """
        Calibrate team colors using multiple frames (picks the one with
        the most detected players for better KMeans clustering).
        """
        # Find the frame with the most players for calibration
        best_frame_idx = 0
        best_count = 0
        for i, pt in enumerate(all_player_tracks):
            if len(pt) > best_count:
                best_count = len(pt)
                best_frame_idx = i

        print(f"  [teams] calibrating on frame {best_frame_idx} ({best_count} players)")
        frame = frames[best_frame_idx]
        player_detections = all_player_tracks[best_frame_idx]

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            # Skip tiny bboxes
            if (bbox[2] - bbox[0]) < 10 or (bbox[3] - bbox[1]) < 15:
                continue
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        if len(player_colors) < 4:
            print("  [teams] WARNING: too few valid player colors, falling back")
            self.team_colors[1] = (0, 0, 200)   # red (BGR)
            self.team_colors[2] = (200, 100, 50) # blue (BGR)
            self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=1)
            self.kmeans.fit([[0,0,200],[200,100,50]])
            return

        self.kmeans = KMeans(n_clusters=2, init='k-means++', n_init=10, random_state=42)
        self.kmeans.fit(player_colors)
        self.team_colors[1] = tuple(self.kmeans.cluster_centers_[0].astype(int).tolist())
        self.team_colors[2] = tuple(self.kmeans.cluster_centers_[1].astype(int).tolist())
        print(f"  [teams] team 1 color (BGR): {self.team_colors[1]}")
        print(f"  [teams] team 2 color (BGR): {self.team_colors[2]}")

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        player_color = self.get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
        self.player_team_dict[player_id] = team_id
        return team_id
