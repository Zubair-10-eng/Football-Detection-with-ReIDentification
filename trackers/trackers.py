import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width
import torch
import os

try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    FeatureExtractor = None

class Tracker:
    def __init__(self, model_path, device=None):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.reid = FeatureExtractor('osnet_x1_0', device=self.device) if FeatureExtractor else None
        self.player_features = {}
        self.player_histograms = {}  # Store color histograms for each player
        self.player_positions = {}
        self.next_id = 1001
        self.sim_thresh = 0.75
        self.max_feat_hist = 15
        self.max_pos_hist = 10
        self.hist_thresh = 0.7  # Tune this for your data (1.0 is perfect match)

    def extract_features(self, frame, bbox):
        if not self.reid:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        return self.reid(crop).cpu().numpy().flatten().tolist()

    def extract_color_histogram(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def compare_histograms(self, hist1, hist2):
        if hist1 is None or hist2 is None:
            return 0.0
        # Correlation: 1.0 is perfect match, -1.0 is opposite
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    def center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def similarity(self, f1, f2):
        if f1 is None or f2 is None:
            return 0.0
        f1, f2 = np.array(f1), np.array(f2)
        n1, n2 = np.linalg.norm(f1), np.linalg.norm(f2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(f1, f2) / (n1 * n2))

    def close_enough(self, pos1, pos2, thresh=100):
        return np.linalg.norm(np.array(pos1) - np.array(pos2)) < thresh

    def assign_id(self, features, bbox, color_hist=None):
        # First, filter candidates by color histogram similarity
        best_id, best_sim = None, 0
        best_hist_sim = 0
        candidate_ids = []
        for pid, hists in self.player_histograms.items():
            if not hists:
                continue
            hist_sim = self.compare_histograms(color_hist, hists[-1])
            if hist_sim > self.hist_thresh:
                candidate_ids.append(pid)
            if hist_sim > best_hist_sim:
                best_hist_sim = hist_sim
        # Now, among candidates, use OSNet feature similarity
        for pid in candidate_ids:
            feats = self.player_features.get(pid, [])
            for old_feat in feats[-5:]:
                sim = self.similarity(features, old_feat)
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid
        if best_sim > self.sim_thresh:
            self.player_positions[best_id].append(self.center(bbox))
            if len(self.player_positions[best_id]) > self.max_pos_hist:
                self.player_positions[best_id] = self.player_positions[best_id][-self.max_pos_hist:]
            return best_id
        # Try spatial match if feature match fails
        for pid, positions in self.player_positions.items():
            if positions and self.close_enough(self.center(bbox), positions[-1]):
                self.player_positions[pid].append(self.center(bbox))
                if len(self.player_positions[pid]) > self.max_pos_hist:
                    self.player_positions[pid] = self.player_positions[pid][-self.max_pos_hist:]
                return pid
        # New player
        pid = self.next_id
        self.next_id += 1
        self.player_features[pid] = []
        self.player_histograms[pid] = []
        self.player_positions[pid] = [self.center(bbox)]
        return pid

    def detect(self, frames):
        batch = 20
        out = []
        for i in range(0, len(frames), batch):
            out += self.model.predict(frames[i:i+batch], conf=0.1)
        return out

    def track(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        detections = self.detect(frames)
        tracks = {"players": [], "referees": [], "ball": []}
        for i, det in enumerate(detections):
            cls_names = det.names
            cls_inv = {v: k for k, v in cls_names.items()}
            det_sv = sv.Detections.from_ultralytics(det)
            for idx, cid in enumerate(det_sv.class_id):
                if cls_names[cid] == "goalkeeper":
                    det_sv.class_id[idx] = cls_inv["player"]
            tracked = self.tracker.update_with_detections(det_sv)
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            frame = frames[i]
            for t in tracked:
                bbox = t[0].tolist()
                cid = t[3]
                tid = t[4]
                if cid == cls_inv['player']:
                    feats = self.extract_features(frame, bbox)
                    color_hist = self.extract_color_histogram(frame, bbox)
                    pid = self.assign_id(feats, bbox, color_hist)
                    self.player_features[pid].append(feats)
                    if len(self.player_features[pid]) > self.max_feat_hist:
                        self.player_features[pid] = self.player_features[pid][-self.max_feat_hist:]
                    self.player_histograms[pid].append(color_hist)
                    if len(self.player_histograms[pid]) > self.max_feat_hist:
                        self.player_histograms[pid] = self.player_histograms[pid][-self.max_feat_hist:]
                    tracks["players"][i][pid] = {"bbox": bbox, "features": feats, "color_hist": color_hist, "track_id": tid}
                if cid == cls_inv['referee']:
                    feats = self.extract_features(frame, bbox)
                    tracks["referees"][i][tid] = {"bbox": bbox, "features": feats}
            for t in det_sv:
                bbox = t[0].tolist()
                cid = t[3]
                if cid == cls_inv['ball']:
                    tracks["ball"][i][1] = {"bbox": bbox}
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        return tracks

    def draw_player(self, frame, bbox, color, pid=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(frame, (x_center, y2), (int(width), int(0.35 * width)), 0.0, -45, 235, color, 2, cv2.LINE_4)
        rect_w, rect_h = 40, 20
        x1_rect = x_center - rect_w // 2
        x2_rect = x_center + rect_w // 2
        y1_rect = (y2 - rect_h // 2) + 15
        y2_rect = (y2 + rect_h // 2) + 15
        if pid is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)), color, cv2.FILLED)
            x1_text = x1_rect + 12
            if pid > 99:
                x1_text -= 10
            cv2.putText(frame, f"ID:{pid}", (int(x1_text), int(y1_rect + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        return frame

    def draw_ball(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0,0,0), 2)
        return frame

    def draw(self, video_frames, tracks):
        out = []
        for i, frame in enumerate(video_frames):
            f = frame.copy()
            for pid, player in tracks["players"][i].items():
                f = self.draw_player(f, player["bbox"], (0,0,255), pid)
            for _, ref in tracks["referees"][i].items():
                f = self.draw_player(f, ref["bbox"], (0,255,255))
            for _, ball in tracks["ball"][i].items():
                f = self.draw_ball(f, ball["bbox"], (0,255,0))
            out.append(f)
        return out

    # Aliases for compatibility
    get_object_tracks = track
    draw_annotations = draw
    extract_osnet_features = extract_features
    draw_ellipse = draw_player
    draw_traingle = draw_ball
    get_track_features = lambda self, tracks, track_type="players": {pid: [d["features"] for d in frame.values() if "features" in d and d["features"] is not None] for frame in tracks[track_type] for pid, d in frame.items()}
    get_player_statistics = lambda self: {"total_players": len(self.player_features), "feature_history_lengths": {pid: len(f) for pid, f in self.player_features.items()}}