import numpy as np
import cv2


# Show overlay for N frames AFTER the pass only
PASS_DISPLAY_AFTER = 20   # ~0.8s at 25fps
PASS_DISPLAY_BEFORE = 5   # tiny window before for context

# Minimum frames a player must hold the ball to count as stable possession.
MIN_HOLD_FRAMES = 3

# Players with field coordinates outside these bounds are ignored (bad transform).
FIELD_X_MIN, FIELD_X_MAX = -2, 107
FIELD_Y_MIN, FIELD_Y_MAX = -2, 70


class OffsideDetector:
    def __init__(self):
        self.offside_line_color = (0, 255, 255)   # yellow
        self.offside_player_color = (0, 0, 255)   # red (offside)
        self.safe_player_color = (0, 255, 0)       # green (onside)
        self.receiver_highlight = (255, 0, 255)    # magenta — receiver ring
        self.defender_highlight = (255, 255, 0)    # cyan — defender ring

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _foot_point(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, y2)

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    # ------------------------------------------------------------------
    # Stabilised ball possession
    # ------------------------------------------------------------------

    @staticmethod
    def _stabilise_possession(ball_holders, min_hold=MIN_HOLD_FRAMES):
        """
        Smooth out noisy frame-by-frame ball assignments.
        A player only 'owns' the ball if they hold it for >= min_hold
        consecutive frames.  Short flickers are replaced by -1 (no holder).
        """
        n = len(ball_holders)
        stable = [-1] * n
        i = 0
        while i < n:
            pid = ball_holders[i]
            if pid == -1:
                i += 1
                continue
            j = i
            while j < n and ball_holders[j] == pid:
                j += 1
            run_len = j - i
            if run_len >= min_hold:
                for k in range(i, j):
                    stable[k] = pid
            i = j
        return stable

    # ------------------------------------------------------------------
    # Pass detection — stabilised, uses KICK frame not reception frame
    # ------------------------------------------------------------------

    def detect_passes(self, tracks, ball_holders):
        """
        Detect pass events using stabilised possession.
        A pass = ball goes from player A → player B on the SAME team.
        - kick_frame: last frame A had the ball
        - analysis_frame: best frame where BOTH A and B are visible
          (searched near kick_frame ±10 for offside analysis)
        Filters out camera-cut gaps (>MAX_FLIGHT) and backward/lateral passes.
        """
        stable = self._stabilise_possession(ball_holders)

        # Build segments: (player_id, start_frame, end_frame_inclusive)
        segments = []
        i = 0
        n = len(stable)
        while i < n:
            if stable[i] == -1:
                i += 1
                continue
            pid = stable[i]
            j = i
            while j < n and stable[j] == pid:
                j += 1
            segments.append((pid, i, j - 1))
            i = j

        MAX_FLIGHT = 50  # >2 seconds gap = camera cut, not a pass

        passes = []
        for idx in range(1, len(segments)):
            passer_id, p_start, p_end = segments[idx - 1]
            receiver_id, r_start, r_end = segments[idx]

            if passer_id == receiver_id:
                continue

            flight_gap = r_start - p_end - 1
            if flight_gap > MAX_FLIGHT:
                continue

            kick_frame = p_end
            receive_frame = r_start

            # Same team check
            prev_team = self._get_player_team(tracks, kick_frame, passer_id)
            curr_team = self._get_player_team(tracks, receive_frame, receiver_id)
            if prev_team is None or curr_team is None or prev_team != curr_team:
                continue

            # Find the best frame near the kick where BOTH players are visible
            analysis_frame = self._find_analysis_frame(
                tracks, kick_frame, passer_id, receiver_id, search_range=10
            )

            passes.append({
                'kick_frame': kick_frame,
                'receive_frame': receive_frame,
                'analysis_frame': analysis_frame,
                'passer': passer_id,
                'receiver': receiver_id,
                'team': prev_team,
                'flight_gap': flight_gap,
            })

        return passes

    def _find_analysis_frame(self, tracks, kick_frame, passer_id, receiver_id,
                             search_range=10):
        """
        Find the best frame near kick_frame where both passer and receiver
        are tracked. Searches kick_frame first, then nearby frames.
        Returns kick_frame if no better frame found.
        """
        best = kick_frame
        for offset in range(0, search_range + 1):
            for f in [kick_frame + offset, kick_frame - offset]:
                if f < 0 or f >= len(tracks['players']):
                    continue
                pt = tracks['players'][f]
                if passer_id in pt and receiver_id in pt:
                    return f
        return best

    @staticmethod
    def _get_bbox(tracks, frame_num, player_id):
        if frame_num < 0 or frame_num >= len(tracks['players']):
            return None
        player = tracks['players'][frame_num].get(player_id)
        if player:
            return player.get('bbox')
        return None

    @staticmethod
    def _get_player_team(tracks, frame_num, player_id):
        if frame_num < 0 or frame_num >= len(tracks['players']):
            return None
        player = tracks['players'][frame_num].get(player_id)
        if player:
            return player.get('team')
        return None

    # ------------------------------------------------------------------
    # Core offside analysis — evaluated at the KICK frame
    # ------------------------------------------------------------------

    def detect_offside(self, tracks, team_ball_control, view_transformer,
                       ball_holders):
        """
        Focused offside detection at pass moments.
        Uses the analysis_frame (best frame where both passer & receiver
        are visible, near the kick) for position evaluation.

        Returns:
            offside_results: dict keyed by frame_num
            pass_events: list of enriched pass dicts
        """
        passes = self.detect_passes(tracks, ball_holders)
        attack_dir = self._detect_attack_direction(tracks, view_transformer)
        offside_results = {}
        enriched_passes = []

        print(f"  [offside] {len(passes)} passes detected")

        for p in passes:
            kick_frame = p['kick_frame']
            analysis_frame = p.get('analysis_frame', kick_frame)
            receiver_id = p['receiver']
            passer_id = p['passer']
            attacking_team = p['team']

            direction = attack_dir.get(attacking_team, 1)

            # Analyse positions at the ANALYSIS frame (both players visible)
            player_track = tracks['players'][analysis_frame]
            attackers = []
            defenders = []

            for pid, pdata in player_track.items():
                bbox = pdata.get('bbox')
                team = pdata.get('team')
                if bbox is None or team is None:
                    continue
                foot = self._foot_point(bbox)
                transformed = view_transformer.transform_point(np.array([foot]))
                if transformed is None:
                    continue
                pos_2d = transformed[0]
                # Skip players whose transformed position is outside the field
                if (pos_2d[0] < FIELD_X_MIN or pos_2d[0] > FIELD_X_MAX or
                        pos_2d[1] < FIELD_Y_MIN or pos_2d[1] > FIELD_Y_MAX):
                    continue
                if team == attacking_team:
                    attackers.append((pid, pos_2d, bbox))
                else:
                    defenders.append((pid, pos_2d, bbox))

            if len(defenders) < 2:
                print(f"    pass @ frame {kick_frame}: SKIP (only {len(defenders)} defenders)")
                continue

            # Second-last defender (offside line)
            if direction > 0:
                sorted_defs = sorted(defenders, key=lambda x: x[1][0], reverse=True)
            else:
                sorted_defs = sorted(defenders, key=lambda x: x[1][0], reverse=False)
            key_defender = sorted_defs[1]
            offside_line_x = key_defender[1][0]

            # Find RECEIVER position
            receiver_entry = None
            for a in attackers:
                if a[0] == receiver_id:
                    receiver_entry = a
                    break
            if receiver_entry is None:
                print(f"    pass @ frame {kick_frame}: SKIP (receiver {receiver_id} not found)")
                continue

            # Offside check
            if direction > 0:
                is_offside = receiver_entry[1][0] > offside_line_x
            else:
                is_offside = receiver_entry[1][0] < offside_line_x

            pass_info = {
                **p,
                'is_offside': is_offside,
                'offside_line_x': offside_line_x,
                'receiver_pos_2d': receiver_entry[1],
                'key_defender_id': key_defender[0],
                'key_defender_pos_2d': key_defender[1],
                'attackers': [(a[0], a[1]) for a in attackers],
                'defenders': [(d[0], d[1]) for d in defenders],
                'attack_direction': direction,
            }
            enriched_passes.append(pass_info)

            verdict = "OFFSIDE" if is_offside else "onside"
            print(f"    pass @ frame {kick_frame} (analysis={analysis_frame}): "
                  f"{passer_id}→{receiver_id}  flight={p.get('flight_gap',0)}f  [{verdict}]")

            # Populate overlay window (mostly AFTER the kick)
            start = max(0, kick_frame - PASS_DISPLAY_BEFORE)
            end = min(len(tracks['players']), kick_frame + PASS_DISPLAY_AFTER)
            for f in range(start, end):
                if f not in offside_results or abs(f - kick_frame) < abs(f - offside_results[f]['kick_frame']):
                    offside_results[f] = {
                        'kick_frame': kick_frame,
                        'is_kick_frame': (f == kick_frame),
                        **pass_info,
                    }

        return offside_results, enriched_passes

    # ------------------------------------------------------------------
    # Semi-automatic: analyse offside at user-specified frames
    # ------------------------------------------------------------------

    def detect_offside_at_frames(self, tracks, team_ball_control, view_transformer,
                                  ball_holders, pass_frames):
        """
        Semi-automatic mode: the user specifies which frames are pass moments.
        At each frame:
          1. Find who has the ball = passer
          2. Find the most forward teammate in attack direction = receiver
          3. Find the 2nd-last defender = offside line
          4. Check if receiver is offside
        """
        attack_dir = self._detect_attack_direction(tracks, view_transformer)
        offside_results = {}
        enriched_passes = []

        print(f"  [offside] Analysing {len(pass_frames)} manual pass frame(s)")

        for kick_frame in pass_frames:
            if kick_frame < 0 or kick_frame >= len(tracks['players']):
                print(f"    frame {kick_frame}: SKIP (out of range)")
                continue

            # Find passer = ball holder at this frame
            passer_id = ball_holders[kick_frame] if kick_frame < len(ball_holders) else -1
            if passer_id == -1:
                # Search nearby frames for ball holder
                for offset in range(1, 10):
                    for f in [kick_frame - offset, kick_frame + offset]:
                        if 0 <= f < len(ball_holders) and ball_holders[f] != -1:
                            passer_id = ball_holders[f]
                            break
                    if passer_id != -1:
                        break
            if passer_id == -1:
                print(f"    frame {kick_frame}: SKIP (no ball holder found)")
                continue

            passer_team = self._get_player_team(tracks, kick_frame, passer_id)
            if passer_team is None:
                print(f"    frame {kick_frame}: SKIP (passer has no team)")
                continue

            direction = attack_dir.get(passer_team, 1)

            # Collect all players with valid field positions
            player_track = tracks['players'][kick_frame]
            attackers = []
            defenders = []

            for pid, pdata in player_track.items():
                bbox = pdata.get('bbox')
                team = pdata.get('team')
                if bbox is None or team is None:
                    continue
                foot = self._foot_point(bbox)
                transformed = view_transformer.transform_point(np.array([foot]))
                if transformed is None:
                    continue
                pos_2d = transformed[0]
                if (pos_2d[0] < FIELD_X_MIN or pos_2d[0] > FIELD_X_MAX or
                        pos_2d[1] < FIELD_Y_MIN or pos_2d[1] > FIELD_Y_MAX):
                    continue
                if team == passer_team:
                    attackers.append((pid, pos_2d, bbox))
                else:
                    defenders.append((pid, pos_2d, bbox))

            if len(defenders) < 2:
                print(f"    frame {kick_frame}: SKIP (only {len(defenders)} defenders)")
                continue

            # Offside line = 2nd-last defender
            if direction > 0:
                sorted_defs = sorted(defenders, key=lambda x: x[1][0], reverse=True)
            else:
                sorted_defs = sorted(defenders, key=lambda x: x[1][0], reverse=False)
            key_defender = sorted_defs[1]
            offside_line_x = key_defender[1][0]

            # Receiver = most forward teammate (excluding passer)
            candidates = [a for a in attackers if a[0] != passer_id]
            if not candidates:
                print(f"    frame {kick_frame}: SKIP (no receiver candidates)")
                continue

            if direction > 0:
                receiver_entry = max(candidates, key=lambda x: x[1][0])
            else:
                receiver_entry = min(candidates, key=lambda x: x[1][0])

            receiver_id = receiver_entry[0]

            # Offside check
            if direction > 0:
                is_offside = receiver_entry[1][0] > offside_line_x
            else:
                is_offside = receiver_entry[1][0] < offside_line_x

            pass_info = {
                'kick_frame': kick_frame,
                'analysis_frame': kick_frame,
                'receive_frame': kick_frame,
                'passer': passer_id,
                'receiver': receiver_id,
                'team': passer_team,
                'is_offside': is_offside,
                'offside_line_x': offside_line_x,
                'receiver_pos_2d': receiver_entry[1],
                'key_defender_id': key_defender[0],
                'key_defender_pos_2d': key_defender[1],
                'attackers': [(a[0], a[1]) for a in attackers],
                'defenders': [(d[0], d[1]) for d in defenders],
                'attack_direction': direction,
            }
            enriched_passes.append(pass_info)

            verdict = "OFFSIDE" if is_offside else "onside"
            print(f"    frame {kick_frame}: passer={passer_id} (team {passer_team}) → "
                  f"receiver={receiver_id} | DEF={key_defender[0]} | [{verdict}]")
            print(f"      receiver field_x={receiver_entry[1][0]:.1f}  "
                  f"offside_line={offside_line_x:.1f}  dir={direction}")

            # Populate overlay window
            start = max(0, kick_frame - PASS_DISPLAY_BEFORE)
            end = min(len(tracks['players']), kick_frame + PASS_DISPLAY_AFTER)
            for f in range(start, end):
                if f not in offside_results or abs(f - kick_frame) < abs(f - offside_results[f]['kick_frame']):
                    offside_results[f] = {
                        'kick_frame': kick_frame,
                        'is_kick_frame': (f == kick_frame),
                        **pass_info,
                    }

        return offside_results, enriched_passes

    def _detect_attack_direction(self, tracks, view_transformer):
        """
        Auto-detect which direction each team attacks.
        Team with lower average X attacks toward higher X (+1).
        """
        team_x = {1: [], 2: []}
        sample_frames = range(0, len(tracks['players']), 10)
        for frame_num in sample_frames:
            for pid, pdata in tracks['players'][frame_num].items():
                bbox = pdata.get('bbox')
                team = pdata.get('team')
                if bbox is None or team is None or team not in (1, 2):
                    continue
                foot = self._foot_point(bbox)
                transformed = view_transformer.transform_point(np.array([foot]))
                if transformed is not None:
                    team_x[team].append(transformed[0][0])

        avg1 = np.mean(team_x[1]) if team_x[1] else 52.5
        avg2 = np.mean(team_x[2]) if team_x[2] else 52.5

        if avg1 < avg2:
            return {1: 1, 2: -1}
        else:
            return {1: -1, 2: 1}

    # ------------------------------------------------------------------
    # Drawing — only use LIVE bboxes, never stale fallbacks
    # ------------------------------------------------------------------

    def draw_offside_overlay(self, frame, offside_data, view_transformer,
                             tracks=None, frame_num=None):
        """
        Clean overlay:
        - Yellow offside line
        - Magenta ring on RECEIVER  (with OFFSIDE/ONSIDE label)
        - Cyan ring on KEY DEFENDER (with DEF label)
        - Verdict banner at top
        Only draws player markers when the player is tracked in THIS frame.
        """
        if offside_data is None:
            return frame

        is_offside = offside_data.get('is_offside', False)
        offside_line_x = offside_data.get('offside_line_x')
        receiver_id = offside_data.get('receiver')
        defender_id = offside_data.get('key_defender_id')
        is_kick_frame = offside_data.get('is_kick_frame', False)
        kick_frame = offside_data.get('kick_frame', 0)

        h, w = frame.shape[:2]

        # ONLY use live bboxes — no stale fallbacks
        # Also reject bboxes near the frame edges (likely phantom detections)
        receiver_bbox = None
        defender_bbox = None
        edge_margin = 25
        if tracks and frame_num is not None:
            pt = tracks['players'][frame_num]
            if receiver_id in pt:
                rb = pt[receiver_id].get('bbox')
                if rb is not None:
                    rcx = (rb[0] + rb[2]) / 2
                    if edge_margin < rcx < w - edge_margin:
                        receiver_bbox = rb
            if defender_id in pt:
                db = pt[defender_id].get('bbox')
                if db is not None:
                    dcx = (db[0] + db[2]) / 2
                    if edge_margin < dcx < w - edge_margin:
                        defender_bbox = db

        # --- Offside line (skip if off-screen) ---
        if offside_line_x is not None:
            lx = view_transformer.inverse_transform_x(offside_line_x)
            if lx is not None:
                lx = int(lx)
                if 10 < lx < w - 10:
                    overlay = frame.copy()
                    cv2.line(overlay, (lx, 0), (lx, h), self.offside_line_color, 2)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # --- Highlight receiver (magenta ring) ---
        if receiver_bbox is not None:
            verdict_color = self.offside_player_color if is_offside else self.safe_player_color
            rx = int((receiver_bbox[0] + receiver_bbox[2]) / 2)
            ry = int(receiver_bbox[1])
            bw = max(int(receiver_bbox[2] - receiver_bbox[0]), 20)
            cv2.circle(frame, (rx, ry - 10), bw // 2 + 8, self.receiver_highlight, 4)
            cv2.circle(frame, (rx, ry - 10), 8, verdict_color, cv2.FILLED)
            label = "OFFSIDE" if is_offside else "ONSIDE"
            cv2.putText(frame, label, (rx - 30, ry - bw // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.receiver_highlight, 2)
            cv2.putText(frame, "REC", (rx - 12, ry + bw // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.receiver_highlight, 2)

        # --- Highlight key defender (cyan ring) ---
        if defender_bbox is not None:
            dx = int((defender_bbox[0] + defender_bbox[2]) / 2)
            dy = int(defender_bbox[1])
            dw = max(int(defender_bbox[2] - defender_bbox[0]), 20)
            cv2.circle(frame, (dx, dy - 10), dw // 2 + 8, self.defender_highlight, 4)
            cv2.circle(frame, (dx, dy - 10), 8, self.defender_highlight, cv2.FILLED)
            cv2.putText(frame, "DEF", (dx - 12, dy - dw // 2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.defender_highlight, 2)

        # --- Verdict banner ---
        banner_h = 40
        overlay = frame.copy()
        if is_offside:
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 180), cv2.FILLED)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            txt = "FORA DE JOGO!" if is_kick_frame else "FORA DE JOGO"
            cv2.putText(frame, txt, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 120, 0), cv2.FILLED)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, "ONSIDE", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        diff = frame_num - kick_frame if frame_num is not None else 0
        sign = "+" if diff >= 0 else ""
        cv2.putText(frame, f"Kick {sign}{diff}", (w - 120, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame
