"""
Football Offside Detector — Main Pipeline
==========================================
Semi-automatic offside analysis:
  - Automatic: tracks players, assigns teams, detects ball
  - Manual: set MANUAL_PASS_FRAMES to specify exact pass moments
  - The system analyses offside at those frames automatically
"""

import numpy as np
from utils import read_video, save_video
from tracker import Tracker
from team_assigner import TeamAssigner
from view_transformer import ViewTransformer
from offside_detector import OffsideDetector
from field_visualizer import FieldVisualizer

INPUT_VIDEO  = 'input_videos/offside_clip.mp4'
OUTPUT_VIDEO = 'output_videos/output.avi'
MODEL_PATH   = 'models/best.pt'
STUB_PATH    = 'stubs/track_stubs.pkl'

# ── Semi-automatic: specify pass frames manually ──────────────
# Set to a list of frame numbers where passes happen.
# Leave empty [] for fully automatic pass detection.
# Tip: at 25fps, second 20 = frame 500. Use calibrate.py to find frames.
MANUAL_PASS_FRAMES = list(range(475, 576, 12))


def main():
    print("[1/6] Reading video…")
    video_frames = read_video(INPUT_VIDEO)
    print(f"      Loaded {len(video_frames)} frames.")

    print("[2/6] Tracking players, referees and ball…")
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path=STUB_PATH,
    )
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print("[3/6] Assigning teams by jersey colour…")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames, tracks['players'])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, data in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num], data['bbox'], player_id
            )
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    print("[4/6] Computing ball possession per frame…")
    team_ball_control = []
    ball_holders = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num].get(1, {}).get('bbox')
        assigned_player = tracker.assign_ball_to_player(player_track, ball_bbox)
        ball_holders.append(assigned_player)
        if assigned_player != -1:
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    team_ball_control = np.array(team_ball_control)

    print("[5/6] Detecting offside…")
    view_transformer = ViewTransformer()
    offside_detector = OffsideDetector()

    if MANUAL_PASS_FRAMES:
        print(f"  [mode] SEMI-AUTOMATIC — {len(MANUAL_PASS_FRAMES)} manual pass frame(s)")
        offside_results, pass_events = offside_detector.detect_offside_at_frames(
            tracks, team_ball_control, view_transformer,
            ball_holders, MANUAL_PASS_FRAMES
        )
    else:
        print("  [mode] AUTOMATIC pass detection")
        offside_results, pass_events = offside_detector.detect_offside(
            tracks, team_ball_control, view_transformer, ball_holders
        )

    for i, pe in enumerate(pass_events):
        verdict = "OFFSIDE" if pe['is_offside'] else "ONSIDE"
        print(f"      Pass #{i+1} @ frame {pe['kick_frame']}: "
              f"player {pe['passer']}→{pe['receiver']} [{verdict}]")
    if not pass_events:
        print("      No passes detected — offside overlay will be minimal.")

    print("[6/6] Drawing annotations and saving output…")
    field_visualizer = FieldVisualizer()
    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    for frame_num, frame in enumerate(output_frames):
        frame = offside_detector.draw_offside_overlay(
            frame, offside_results.get(frame_num), view_transformer,
            tracks=tracks, frame_num=frame_num
        )
        output_frames[frame_num] = field_visualizer.draw_field(
            frame, offside_results.get(frame_num), tracks, frame_num
        )

    save_video(output_frames, OUTPUT_VIDEO)
    print(f"\n✅ Done! Output saved to {OUTPUT_VIDEO}")


if __name__ == '__main__':
    main()
