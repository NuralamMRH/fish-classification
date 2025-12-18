#!/usr/bin/env python3
"""
Enhanced Fish Identification Web App with Visual Results
=======================================================

A web application with drag & drop and visual bounding boxes
showing detected fish with species names overlaid.
"""

import os
import sys
import cv2
import numpy as np
import json
import subprocess
import tempfile
import uuid
from datetime import datetime
from flask import Flask, request, render_template_string, redirect, flash, send_file
from werkzeug.utils import secure_filename
from math import pi

app = Flask(__name__, static_folder=None)
app.secret_key = 'fish_secret_2024'

# Configuration (absolute paths)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
STATIC_FOLDER = os.path.join(PROJECT_ROOT, 'static')
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
RESULTS_FOLDER = os.path.join(STATIC_FOLDER, 'results')
TEMP_FOLDER = os.path.join(STATIC_FOLDER, 'tmp')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
DEFAULT_PX_PER_CM = 37.7952755906
CLASS_INFO_PATH = os.path.join(PROJECT_ROOT, 'models', 'classification', 'info.json')
SPECIES_RULES = [
    {"kw": ["bluefin tuna", "thunnus thynnus"], "len": (160, 180), "wt": (140, 180), "sci": "Thunnus thynnus", "common": "Bluefin tuna"},
    {"kw": ["yellowfin tuna", "thunnus albacares"], "len": (150, 170), "wt": (110, 150), "sci": "Thunnus albacares", "common": "Yellowfin tuna"},
    {"kw": ["giant grouper", "epinephelus lanceolatus", "grouper"], "len": (150, 170), "wt": (90, 120), "sci": "Epinephelus lanceolatus", "common": "Giant grouper"},
    {"kw": ["marlin", "istiophoridae"], "len": (170, 200), "wt": (120, 160), "sci": "Istiophoridae", "common": "Marlin"},
    {"kw": ["swordfish", "xiphias gladius"], "len": (160, 180), "wt": (100, 140), "sci": "Xiphias gladius", "common": "Swordfish"},
    {"kw": ["shark", "selachimorpha"], "len": (160, 190), "wt": (90, 130), "sci": "Selachimorpha", "common": "Shark"},
    {"kw": ["mekong catfish", "giant catfish", "catfish", "pangasianodon gigas"], "len": (140, 160), "wt": (80, 110), "sci": "Pangasianodon gigas", "common": "Mekong giant catfish"},
    {"kw": ["common carp", "cyprinus carpio", "carp"], "len": (140, 150), "wt": (55, 70), "sci": "Cyprinus carpio", "common": "Common carp"},
    {"kw": ["grass carp", "ctenopharyngodon idella"], "len": (150, 160), "wt": (60, 85), "sci": "Ctenopharyngodon idella", "common": "Grass carp"},
    {"kw": ["barramundi", "lates calcarifer"], "len": (140, 150), "wt": (45, 60), "sci": "Lates calcarifer", "common": "Barramundi"},
    {"kw": ["hilsa", "tenualosa ilisha"], "len": (120, 130), "wt": (25, 35), "sci": "Tenualosa ilisha", "common": "Hilsa"},
]
def _match_species_rule(name):
    n = (name or "").lower()
    for r in SPECIES_RULES:
        if any(k in n for k in r["kw"]):
            return r
    return None

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def test_models():
    """Test if models are working."""
    try:
        # Test YOLO detection
        pr = PROJECT_ROOT.replace("\\", "/")
        yolo_script = f"""
import sys, os, cv2
sys.path.insert(0, os.path.join('{pr}', 'models', 'detection'))
from inference import YOLOInference
detector = YOLOInference(os.path.join('{pr}', 'models', 'detection', 'model.ts'), yolo_ver='v8')
image = cv2.imread(os.path.join('{pr}', 'vietnamese-catfish.jpg'))
detections = detector.predict(image)
if image is not None and detections and len(detections) > 0 and len(detections[0]) > 0:
    print('YOLO_SUCCESS')
else:
    print('YOLO_FAIL')
"""
        yolo_test = subprocess.run([sys.executable, "-c", yolo_script], capture_output=True, text=True, timeout=30)
        
        # Test classification
        class_script = f"""
import sys, os, cv2, json
sys.path.insert(0, os.path.join('{pr}', 'models', 'classification'))
from inference import EmbeddingClassifier
config = {{
    'model': {{
        'path': os.path.join('{pr}', 'models', 'classification', 'model.ckpt'),
        'device': 'cpu'
    }},
    'dataset': {{
        'path': os.path.join('{pr}', 'models', 'classification', 'database.pt')
    }},
    'log_level': 'INFO'
}}
classifier = EmbeddingClassifier(config)
image = cv2.imread(os.path.join('{pr}', 'vietnamese-catfish.jpg'))
results = classifier.inference_numpy(image)
if image is not None and results and len(results) > 0:
    items = results[0] if isinstance(results[0], list) else results
    if items:
        items = sorted(items, key=lambda rr: rr.accuracy if hasattr(rr, 'accuracy') else rr.get('accuracy', 0.0), reverse=True)
        r = items[0]
        species = r.name if hasattr(r, 'name') else r.get('name', 'Unknown')
        acc = float(r.accuracy if hasattr(r, 'accuracy') else r.get('accuracy', 0.0))
        print('CLASS_SUCCESS:' + json.dumps({{'species': species, 'accuracy': acc}}))
else:
    print('CLASS_FAIL')
"""
        class_test = subprocess.run([sys.executable, "-c", class_script], capture_output=True, text=True, timeout=30)
        
        seg_test = subprocess.run([
            sys.executable, "-c", """
import sys, os, cv2
sys.path.insert(0, './models/segmentation')
from inference import Inference
segmentator = Inference('./models/segmentation/model.ts')
image = cv2.imread('vietnamese-catfish.jpg')
polys = segmentator.predict(image)
if polys and len(polys) > 0 and len(polys[0].points) > 0:
    print('SEG_SUCCESS')
else:
    print('SEG_FAIL')
"""
        ], capture_output=True, text=True, timeout=30)
        
        face_test = subprocess.run([
            sys.executable, "-c", """
import sys, os, cv2
sys.path.insert(0, './models/face_detector')
from inference import YOLOInference
detector = YOLOInference('./models/face_detector/model.ts', yolo_ver='v8')
image = cv2.imread('faces.jpeg')
detections = detector.predict(image)
if detections and detections[0]:
    print('FACE_SUCCESS')
else:
    print('FACE_FAIL')
"""
        ], capture_output=True, text=True, timeout=30)
        
        yolo_ok = "YOLO_SUCCESS" in yolo_test.stdout
        class_ok = "CLASS_SUCCESS" in class_test.stdout
        seg_ok = "SEG_SUCCESS" in seg_test.stdout
        face_ok = "FACE_SUCCESS" in face_test.stdout
        
        return yolo_ok, class_ok, seg_ok, face_ok
        
    except Exception as e:
        print(f"Error testing models: {e}")
        return False, False, False, False

def _pairwise_max_distance(pts):
    try:
        from scipy.spatial.distance import pdist, squareform
        D = pdist(pts, metric='euclidean')
        M = squareform(D)
        idx = np.unravel_index(np.argmax(M), M.shape)
        return float(M[idx]), (pts[idx[0]], pts[idx[1]])
    except Exception:
        A = pts.astype(np.float32)
        diff = A[:, None, :] - A[None, :, :]
        D = np.sqrt((diff ** 2).sum(-1))
        idx = np.unravel_index(np.argmax(D), D.shape)
        return float(D[idx]), (pts[idx[0]], pts[idx[1]])

def _rotate_points(pts, angle_deg, center):
    theta = np.deg2rad(angle_deg)
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    shifted = pts - center
    rotated = shifted @ R.T
    return rotated + center

def _estimate_lengths_and_width(pts):
    p = np.array(pts, dtype=np.float32)
    rect = cv2.minAreaRect(p)
    box = cv2.boxPoints(rect).astype(np.float32)
    d01 = float(np.linalg.norm(box[0] - box[1]))
    d12 = float(np.linalg.norm(box[1] - box[2]))
    length_rect = max(d01, d12)
    width_rect = min(d01, d12)
    return length_rect, width_rect, rect

def _compute_FL_and_girth_width(pts, rect):
    p = np.array(pts, dtype=np.float32)
    center = np.array(rect[0], dtype=np.float32)
    angle = rect[2]
    rotated = _rotate_points(p, angle, center)
    xs = rotated[:, 0]
    ys = rotated[:, 1]
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    bins = 60
    edges = np.linspace(x_min, x_max, bins + 1)
    widths = []
    centers = []
    for k in range(bins):
        mask = (xs >= edges[k]) & (xs < edges[k + 1])
        if np.any(mask):
            w = float(np.max(ys[mask]) - np.min(ys[mask]))
        else:
            w = 0.0
        widths.append(w)
        centers.append(float((edges[k] + edges[k + 1]) / 2.0))
    widths = np.array(widths, dtype=np.float32)
    centers = np.array(centers, dtype=np.float32)
    tail_region = int(max(1, 0.25 * len(centers)))
    fork_idx = int(np.argmin(widths[:tail_region]))
    fork_x = float(centers[fork_idx])
    nose_x = float(np.max(xs))
    fl_px = max(0.0, nose_x - fork_x)
    mid_x = float((nose_x + fork_x) / 2.0)
    mid_idx = int(np.argmin(np.abs(centers - mid_x)))
    mid_idx = max(0, min(mid_idx, bins - 1))
    eps = max(1e-3, 0.01 * (x_max - x_min))
    nose_mask = np.abs(xs - nose_x) <= eps
    fork_mask = np.abs(xs - fork_x) <= eps
    nose_y = float(np.mean(ys[nose_mask])) if np.any(nose_mask) else float(np.mean(ys))
    fork_y = float(np.mean(ys[fork_mask])) if np.any(fork_mask) else float(np.mean(ys))
    nose_pt_rot = np.array([[nose_x, nose_y]], dtype=np.float32)
    fork_pt_rot = np.array([[fork_x, fork_y]], dtype=np.float32)
    nose_pt = _rotate_points(nose_pt_rot, -angle, center)[0]
    fork_pt = _rotate_points(fork_pt_rot, -angle, center)[0]
    mid_pt = (nose_pt + fork_pt) / 2.0
    v_len = nose_pt - fork_pt
    v_norm = np.linalg.norm(v_len)
    if v_norm < 1e-6:
        girth_width_px = float(np.max(widths))
        gx = float(centers[mid_idx])
        g_mask = (xs >= edges[mid_idx]) & (xs < edges[mid_idx + 1])
        if np.any(g_mask):
            gy_top = float(np.max(ys[g_mask]))
            gy_bottom = float(np.min(ys[g_mask]))
        else:
            gy_top = float(np.max(ys))
            gy_bottom = float(np.min(ys))
        g_top_rot = np.array([[gx, gy_top]], dtype=np.float32)
        g_bottom_rot = np.array([[gx, gy_bottom]], dtype=np.float32)
        g_top = _rotate_points(g_top_rot, -angle, center)[0]
        g_bottom = _rotate_points(g_bottom_rot, -angle, center)[0]
        return fl_px, girth_width_px, (nose_pt, fork_pt), (g_top, g_bottom)
    perp = np.array([-v_len[1], v_len[0]], dtype=np.float32) / v_norm
    t_pos = None
    t_neg = None
    pos_pt = None
    neg_pt = None
    P = p.astype(np.float32)
    M = mid_pt.astype(np.float32)
    n_pts = len(P)
    for i in range(n_pts):
        A = P[i]
        B = P[(i + 1) % n_pts]
        E = B - A
        A_mat = np.array([[perp[0], -E[0]], [perp[1], -E[1]]], dtype=np.float32)
        b_vec = np.array([A[0] - M[0], A[1] - M[1]], dtype=np.float32)
        det = np.linalg.det(A_mat)
        if abs(det) < 1e-8:
            continue
        sol = np.linalg.solve(A_mat, b_vec)
        t, u = float(sol[0]), float(sol[1])
        if 0.0 <= u <= 1.0:
            I = M + t * perp
            if t >= 0:
                if t_pos is None or t < t_pos:
                    t_pos = t
                    pos_pt = I
            else:
                if t_neg is None or t > t_neg:
                    t_neg = t
                    neg_pt = I
    if pos_pt is None or neg_pt is None:
        girth_width_px = float(widths[mid_idx])
        gx = float(centers[mid_idx])
        g_mask = (xs >= edges[mid_idx]) & (xs < edges[mid_idx + 1])
        if np.any(g_mask):
            gy_top = float(np.max(ys[g_mask]))
            gy_bottom = float(np.min(ys[g_mask]))
        else:
            gy_top = float(np.max(ys))
            gy_bottom = float(np.min(ys))
        g_top_rot = np.array([[gx, gy_top]], dtype=np.float32)
        g_bottom_rot = np.array([[gx, gy_bottom]], dtype=np.float32)
        g_top = _rotate_points(g_top_rot, -angle, center)[0]
        g_bottom = _rotate_points(g_bottom_rot, -angle, center)[0]
        return fl_px, girth_width_px, (nose_pt, fork_pt), (g_top, g_bottom)
    girth_width_px = float(np.linalg.norm(pos_pt - neg_pt))
    return fl_px, girth_width_px, (nose_pt, fork_pt), (pos_pt, neg_pt)

def create_annotated_image(image_path, detection_results):
    """Create an annotated image with bounding boxes and species labels."""
    try:
        # Load original image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Define colors for different fish (similar to the reference image)
        colors = [
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta/Pink
            (0, 255, 0),      # Green
            (255, 255, 0),    # Cyan
            (0, 165, 255),    # Orange
            (128, 0, 128),    # Purple
            (255, 0, 0),      # Blue
            (0, 128, 255),    # Red-Orange
        ]
        
        # Draw bounding boxes and labels for each fish
        for i, fish in enumerate(detection_results['fish']):
            box = fish['box']
            species = fish['species']
            accuracy = fish['accuracy']
            confidence = fish['confidence']
            
            # Get color for this fish (cycle through colors)
            color = colors[i % len(colors)]
            
            x1, y1, x2, y2 = box
            if fish.get('segmentation'):
                pts = np.array(fish['segmentation']['points'], np.int32)
                if len(pts) > 0:
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)
            if fish.get('face_box'):
                fx1, fy1, fx2, fy2 = fish['face_box']
                cv2.rectangle(image, (fx1, fy1), (fx2, fy2), color, 2)
            label = f"{fish.get('common_name') or species}"
            confidence_text = f"Acc: {accuracy:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            extras = []
            m_for_label = fish.get('measurements')
            if m_for_label:
                try:
                    if m_for_label.get('weight_kg_lgg_over_800'):
                        extras.append(f"Weight L*G^2/800: {m_for_label.get('weight_kg_lgg_over_800')} kg")
                except Exception:
                    pass
            (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            (conf_width, conf_height), _ = cv2.getTextSize(confidence_text, font, font_scale, thickness)
            extra_sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t in extras]
            extra_widths = [w for (w, h) in extra_sizes]
            extra_heights = [cv2.getTextSize(t, font, font_scale, thickness)[0][1] for t in extras]
            bg_width = max([label_width, conf_width] + extra_widths) + 10
            bg_height = label_height + conf_height + sum(extra_heights) + 15
            if fish.get('segmentation') and fish['segmentation'] and fish['segmentation'].get('points'):
                spts = np.array(fish['segmentation']['points'], dtype=np.int32)
                cx = int(np.mean(spts[:, 0]))
                cy = int(np.mean(spts[:, 1]))
            else:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
            tlx = max(0, cx - bg_width // 2)
            tly = max(0, cy - bg_height - 5)
            brx = min(image.shape[1] - 1, cx + bg_width // 2)
            bry = min(image.shape[0] - 1, cy)
            cv2.rectangle(image, (tlx, tly), (brx, bry), color, -1)
            y_cursor = tly + 5 + label_height
            cv2.putText(image, label, (tlx + 5, y_cursor), font, font_scale, (0, 0, 0), thickness)
            y_cursor += conf_height + 5
            cv2.putText(image, confidence_text, (tlx + 5, y_cursor), font, font_scale, (0, 0, 0), thickness)
            for t in extras:
                th = cv2.getTextSize(t, font, font_scale, thickness)[0][1]
                y_cursor += th + 5
                cv2.putText(image, t, (tlx + 5, y_cursor), font, font_scale, (0, 0, 0), thickness)
            
            if fish.get('measurements'):
                m = fish['measurements']
                try:
                    if m.get('length_line'):
                        lp1 = tuple(int(v) for v in m['length_line'][0])
                        lp2 = tuple(int(v) for v in m['length_line'][1])
                        cv2.line(image, lp1, lp2, (255, 255, 255), 3)
                        lmid = (int((lp1[0] + lp2[0]) / 2), int((lp1[1] + lp2[1]) / 2))
                        lcm = m.get('length_cm') if m.get('length_cm') else (m.get('length_px') or 0.0) / DEFAULT_PX_PER_CM
                        ltext = f"{m.get('length_type', 'TL')}: {lcm:.2f} cm"
                        cv2.putText(image, ltext, lmid, font, 0.6, (0, 0, 0), 2)
                    if m.get('girth_line'):
                        gp1 = tuple(int(v) for v in m['girth_line'][0])
                        gp2 = tuple(int(v) for v in m['girth_line'][1])
                        cv2.line(image, gp1, gp2, (176, 171, 255), 3)
                        gmid = (int((gp1[0] + gp2[0]) / 2), int((gp1[1] + gp2[1]) / 2))
                        gcm = m.get('girth_cm') if m.get('girth_cm') else (m.get('girth_px') or 0.0) / DEFAULT_PX_PER_CM
                        gtext = f"Girth: {gcm:.2f} cm"
                        cv2.putText(image, gtext, gmid, font, 0.6, (0, 0, 0), 2)
                except Exception:
                    pass
        
        # Save annotated image under static/results
        result_filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, image)
        
        return os.path.join('results', result_filename)
        
    except Exception as e:
        print(f"Error creating annotated image: {e}")
        return None

def process_image_external(image_path, pixels_per_cm=None, length_type='AUTO', girth_factor=pi):
    """Process image using external scripts to avoid import conflicts."""
    try:
        # First, detect fish with YOLO
        temp_dir = TEMP_FOLDER.replace("\\\\", "/")
        pr = PROJECT_ROOT.replace("\\", "/")
        yolo_script = f"""
import sys, os, cv2, json
sys.path.insert(0, os.path.join('{pr}', 'models', 'detection'))
from inference import YOLOInference
detector = YOLOInference(os.path.join('{pr}', 'models', 'detection', 'model.ts'), yolo_ver='v8')
image = cv2.imread('{image_path}')
detections = detector.predict(image)

if image is not None and detections and len(detections) > 0 and len(detections[0]) > 0:
    fish_data = []
    for i, fish in enumerate(detections[0]):
        box = fish.get_box()
        confidence = fish.get_score()
        
        # Save fish crop
        fish_crop = fish.get_mask_BGR()
        crop_path = f'{temp_dir}/temp_fish_{{i}}.jpg'
        cv2.imwrite(crop_path, fish_crop)
        
        fish_data.append({{
            'fish_id': i + 1,
            'box': [int(x) for x in box],
            'confidence': float(confidence),
            'crop_path': crop_path
        }})
    
    result = {{'success': True, 'fish': fish_data}}
    print('YOLO_RESULT:' + json.dumps(result))
else:
    print('YOLO_RESULT:' + json.dumps({{'success': False, 'error': 'No fish detected'}}))
"""
        
        yolo_result = subprocess.run([sys.executable, "-c", yolo_script], 
                                   capture_output=True, text=True, timeout=30)
        
        # Parse YOLO result
        yolo_output = None
        for line in yolo_result.stdout.split('\n'):
            if line.startswith('YOLO_RESULT:'):
                yolo_output = json.loads(line.replace('YOLO_RESULT:', ''))
                break
        
        if not yolo_output or not yolo_output.get('success'):
            return {"error": "No fish detected in image"}
        
        # Now classify each detected fish
        final_results = []
        for fish in yolo_output['fish']:
            crop_path = fish['crop_path']
            
            # Classify this fish crop
            class_script = f"""
import sys, os, cv2, json, torch
from PIL import Image
sys.path.insert(0, os.path.join('{pr}', 'models', 'classification'))
from inference import EmbeddingClassifier
config = {{
    'model': {{
        'path': os.path.join('{pr}', 'models', 'classification', 'model.ckpt'),
        'device': 'cpu'
    }},
    'dataset': {{
        'path': os.path.join('{pr}', 'models', 'classification', 'database.pt')
    }},
    'log_level': 'INFO'
}}
classifier = EmbeddingClassifier(config)
image = cv2.imread('{crop_path}')
try:
    results = classifier.inference_numpy(image)
    items = results[0] if isinstance(results[0], list) else results
    if items and len(items) > 0:
        items = sorted(items, key=lambda rr: rr.accuracy if hasattr(rr, 'accuracy') else rr.get('accuracy', 0.0), reverse=True)
        r = items[0]
        species = r.name if hasattr(r, 'name') else r.get('name', 'Unknown')
        acc = float(r.accuracy if hasattr(r, 'accuracy') else r.get('accuracy', 0.0))
        sid = int(r.species_id) if hasattr(r, 'species_id') else int(r.get('species_id', -1))
        print('CLASS_RESULT:' + json.dumps({{'species': species, 'accuracy': acc, 'species_id': sid}}))
    else:
        # Fallback: use ArcFace probabilities to pick top class
        tensor = classifier.transform(Image.fromarray(image)).unsqueeze(0).to(classifier.device)
        emb, probabilities, _ = classifier.model(tensor)
        top_prob, top_idx = torch.topk(probabilities, 1)
        label = classifier.id_to_label[top_idx[0][0].item()]
        acc = float(top_prob[0][0].item())
        print('CLASS_RESULT:' + json.dumps({{'species': label, 'accuracy': acc}}))
except Exception:
    try:
        tensor = classifier.transform(Image.fromarray(image)).unsqueeze(0).to(classifier.device)
        emb, probabilities, _ = classifier.model(tensor)
        top_prob, top_idx = torch.topk(probabilities, 1)
        label = classifier.id_to_label[top_idx[0][0].item()]
        acc = float(top_prob[0][0].item())
        print('CLASS_RESULT:' + json.dumps({{'species': label, 'accuracy': acc}}))
    except Exception:
        print('CLASS_RESULT:' + json.dumps({{'species': 'Unknown', 'accuracy': 0.0}}))
"""
            
            class_result = subprocess.run([sys.executable, "-c", class_script], 
                                        capture_output=True, text=True, timeout=30)
            
            # Parse classification result
            class_output = None
            for line in class_result.stdout.split('\n'):
                if line.startswith('CLASS_RESULT:'):
                    class_output = json.loads(line.replace('CLASS_RESULT:', ''))
                    break
            
            if not class_output:
                class_output = {'species': 'Unknown', 'accuracy': 0.0}
            
            seg_script = f"""
import sys, os, cv2, json
sys.path.insert(0, os.path.join('{pr}', 'models', 'segmentation'))
from inference import Inference
segmentator = Inference(os.path.join('{pr}', 'models', 'segmentation', 'model.ts'))
image = cv2.imread('{crop_path}')
polys = segmentator.predict(image)
if image is not None and polys and len(polys) > 0 and len(polys[0].points) > 0:
    pts = [(int(p[0]), int(p[1])) for p in polys[0].points]
    print('SEG_RESULT:' + json.dumps({{'points': pts}}))
else:
    print('SEG_RESULT:' + json.dumps({{'points': []}}))
"""
            seg_result = subprocess.run([sys.executable, "-c", seg_script],
                                        capture_output=True, text=True, timeout=30)
            seg_points = []
            for line in seg_result.stdout.split('\n'):
                if line.startswith('SEG_RESULT:'):
                    seg_output = json.loads(line.replace('SEG_RESULT:', ''))
                    seg_points = seg_output.get('points', [])
                    break
            
            face_script = f"""
import sys, os, cv2, json
sys.path.insert(0, os.path.join('{pr}', 'models', 'face_detector'))
from inference import YOLOInference
detector = YOLOInference(os.path.join('{pr}', 'models', 'face_detector', 'model.ts'), yolo_ver='v8')
image = cv2.imread('{crop_path}')
detections = detector.predict(image)
if image is not None and detections and len(detections) > 0 and len(detections[0]) > 0:
    box = detections[0][0].get_box()
    print('FACE_RESULT:' + json.dumps({{'box': [int(box[0]), int(box[1]), int(box[2]), int(box[3])] }}))
else:
    print('FACE_RESULT:' + json.dumps({{'box': []}}))
"""
            face_result = subprocess.run([sys.executable, "-c", face_script],
                                         capture_output=True, text=True, timeout=30)
            face_box = []
            for line in face_result.stdout.split('\n'):
                if line.startswith('FACE_RESULT:'):
                    face_output = json.loads(line.replace('FACE_RESULT:', ''))
                    face_box = face_output.get('box', [])
                    break
            
            x1, y1, x2, y2 = fish['box']
            global_seg_points = [(p[0] + x1, p[1] + y1) for p in seg_points]
            global_face_box = [face_box[0] + x1, face_box[1] + y1, face_box[2] + x1, face_box[3] + y1] if face_box else []
            
            final_item = {
                "fish_id": fish['fish_id'],
                "species": class_output['species'],
                "accuracy": round(class_output['accuracy'], 3),
                "confidence": round(fish['confidence'], 3),
                "box": fish['box'],
                "segmentation": {"points": global_seg_points} if global_seg_points else None,
                "face_box": global_face_box if global_face_box else None
            }
            
            try:
                img = cv2.imread(image_path)
                h, w = img.shape[:2]
                seg_pts = final_item["segmentation"]["points"] if final_item.get("segmentation") else None
                if seg_pts and len(seg_pts) >= 2:
                    tl_px, width_px, rect = _estimate_lengths_and_width(seg_pts)
                    dist_px, tl_endpoints = _pairwise_max_distance(np.array(seg_pts, dtype=np.float32))
                    fl_px, girth_width_px, fl_line, girth_line = _compute_FL_and_girth_width(seg_pts, rect)
                else:
                    bx1, by1, bx2, by2 = final_item["box"]
                    tl_px = float(np.hypot(bx2 - bx1, by2 - by1))
                    width_px = float(min(bx2 - bx1, by2 - by1))
                    dist_px = tl_px
                    fl_px = tl_px
                    girth_width_px = width_px
                    tl_endpoints = [(bx1, by1), (bx2, by2)]
                    midx = (bx1 + bx2) / 2.0
                    girth_line = ((midx, by1), (midx, by2))
                    fl_line = tl_endpoints
                use_type = length_type
                species_lower = (final_item["species"] or "").lower()
                auto_fl_species = ["mackerel", "amberjack", "permit", "pompano", "bluefish", "mullet", "cobia", "rudderfish", "mahi", "hogfish", "tuna"]
                if use_type == 'AUTO':
                    use_type = 'FL' if any(s in species_lower for s in auto_fl_species) else 'TL'
                chosen_length_px = fl_px if use_type == 'FL' else dist_px
                girth_px = float(girth_factor) * float(girth_width_px)
                length_cm = None
                girth_cm = None
                weight_lbs_formula1 = None
                weight_lbs_formula2 = None
                weight_kg_formula1 = None
                weight_kg_formula2 = None
                scale = None
                if pixels_per_cm and float(pixels_per_cm) > 0:
                    scale = float(pixels_per_cm)
                else:
                    scale = DEFAULT_PX_PER_CM
                if scale and float(scale) > 0:
                    length_cm = chosen_length_px / float(scale)
                    girth_cm = girth_px / float(scale)
                    length_in = length_cm / 2.54
                    girth_in = girth_cm / 2.54
                    weight_lbs_formula1 = (length_in * girth_in * girth_in) / 800.0
                    weight_lbs_formula2 = (length_in * length_in * length_in) / 1200.0
                    weight_kg_formula1 = weight_lbs_formula1 * 0.45359237
                    weight_kg_formula2 = weight_lbs_formula2 * 0.45359237
                frame_dists = None
                frame_dists_cm = None
                try:
                    pts = np.array(seg_pts, dtype=np.float32) if seg_pts else np.array([[final_item["box"][0], final_item["box"][1]],
                                                                                        [final_item["box"][2], final_item["box"][3]]], dtype=np.float32)
                    xs = pts[:, 0]
                    ys = pts[:, 1]
                    frame_dists = {
                        "left": float(np.min(xs)),
                        "right": float(w - np.max(xs)),
                        "top": float(np.min(ys)),
                        "bottom": float(h - np.max(ys))
                    }
                    if scale and float(scale) > 0:
                        frame_dists_cm = {
                            "left": round(frame_dists["left"] / float(scale), 2),
                            "right": round(frame_dists["right"] / float(scale), 2),
                            "top": round(frame_dists["top"] / float(scale), 2),
                            "bottom": round(frame_dists["bottom"] / float(scale), 2),
                        }
                except Exception:
                    frame_dists = None
                    frame_dists_cm = None
                final_item["measurements"] = {
                    "length_type": use_type,
                    "length_px": round(chosen_length_px, 2),
                    "girth_px": round(girth_px, 2),
                    "width_px": round(girth_width_px, 2),
                    "length_cm": round(length_cm, 2) if length_cm else None,
                    "girth_cm": round(girth_cm, 2) if girth_cm else None,
                    "weight_kg_lgg_over_800": round(weight_kg_formula1, 2) if weight_lbs_formula1 else None,
                    "weight_kg_l3_over_1200": round(weight_kg_formula2, 2) if weight_lbs_formula2 else None,
                    "frame_distance_px": frame_dists,
                    "frame_distance_cm": frame_dists_cm,
                    "length_line": [list(map(lambda v: float(v), fl_line[1])) , list(map(lambda v: float(v), fl_line[0]))] if use_type == 'FL' else [list(map(lambda v: float(v), tl_endpoints[0])) , list(map(lambda v: float(v), tl_endpoints[1]))],
                    "girth_line": [list(map(lambda v: float(v), girth_line[0])) , list(map(lambda v: float(v), girth_line[1]))]
                }
                rule = _match_species_rule(final_item["species"])
                if rule:
                    final_item["scientific_name"] = rule["sci"]
                    final_item["common_name"] = rule.get("common", final_item["species"])
                    final_item["measurements"]["rule_weight_kg_min"] = rule["wt"][0]
                    final_item["measurements"]["rule_weight_kg_max"] = rule["wt"][1]
                    final_item["measurements"]["rule_length_cm_min"] = rule["len"][0]
                    final_item["measurements"]["rule_length_cm_max"] = rule["len"][1]
                else:
                    final_item["scientific_name"] = None
                    final_item["common_name"] = final_item["species"]
            except Exception:
                pass
            
            final_results.append(final_item)
            
            # Clean up temp file
            try:
                os.remove(crop_path)
            except:
                pass
        
        # Create the final result structure
        result = {"success": True, "fish_count": len(final_results), "fish": final_results}
        
        # Create annotated image with bounding boxes
        annotated_image = create_annotated_image(image_path, result)
        if annotated_image:
            result["annotated_image"] = annotated_image
        
        return result
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Enhanced HTML Template with Drag & Drop and Visual Results
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🐟 Enhanced Fish Identification System</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
        }
        .container { 
            background: white; 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.1); 
        }
        .status { 
            padding: 15px; 
            margin: 15px 0; 
            border-radius: 8px; 
            background: #e8f5e8; 
            border-left: 4px solid #4caf50;
        }
        .upload-zone { 
            margin: 20px 0; 
            padding: 40px; 
            border: 3px dashed #ddd; 
            border-radius: 12px; 
            text-align: center; 
            transition: all 0.3s ease;
            cursor: pointer;
            background: #fafafa;
        }
        .upload-zone:hover, .upload-zone.dragover { 
            border-color: #2196F3; 
            background: #f0f8ff;
            transform: translateY(-2px);
        }
        .upload-zone.dragover {
            border-color: #4caf50;
            background: #f0fff0;
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 15px;
            color: #666;
        }
        .result-container { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 20px; 
            margin: 20px 0; 
        }
        .result-image {
            text-align: center;
        }
        .result-image img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .result-details {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #2196F3;
        }
        .fish-item { 
            border: 1px solid #ddd; 
            margin: 10px 0; 
            padding: 15px; 
            border-radius: 8px; 
            background: white;
            border-left: 4px solid #4caf50;
        }
        .fish-header {
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 10px;
        }
        .species-name {
            font-size: 18px;
            font-weight: bold;
            color: #333;
            margin: 5px 0;
        }
        .accuracy-high { color: #4caf50; }
        .accuracy-medium { color: #ff9800; }
        .accuracy-low { color: #f44336; }
        
        button { 
            background: linear-gradient(45deg, #2196F3, #21cbf3);
            color: white; 
            padding: 12px 25px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
        }
        .error { 
            background: #ffebee; 
            color: #c62828; 
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f44336;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #2196F3;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .summary-card {
            background: linear-gradient(45deg, #4caf50, #8bc34a);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        @media (max-width: 768px) {
            .result-container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🐟 Enhanced Fish Identification System</h1>
        <p>Upload fish images with drag & drop to identify species and see visual bounding boxes</p>
        
        <div class="status">
            <strong>✅ System Status:</strong> Enhanced Visual Detection Ready<br>
            <strong>🔍 YOLOv10 Detection:</strong> {{ yolo_status }}<br>
            <strong>🔬 Species Classification:</strong> {{ class_status }}<br>
            <strong>🎨 Segmentation:</strong> {{ seg_status }}<br>
            <strong>🙂 Face Detection:</strong> {{ face_status }}<br>
            <strong>📊 Database:</strong> {{ db_count or 'unknown' }}+ fish species with visual bounding boxes
        </div>
        
        {% with messages = get_flashed_messages() %}
            {% if messages %}
                {% for message in messages %}
                    <div class="error">{{ message }}</div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        
        <div class="upload-zone" id="uploadZone" onclick="document.getElementById('fileInput').click()">
            <div class="upload-icon">📸</div>
            <h3>Drop fish image here or click to select</h3>
            <p>Supports PNG, JPG, JPEG, GIF, BMP (max 16MB)</p>
            <form id="uploadForm" method="POST" enctype="multipart/form-data" style="display: none;">
                <input type="file" id="fileInput" name="file" accept=".png,.jpg,.jpeg,.gif,.bmp" onchange="handleFileSelect(event)">
                <input type="hidden" name="pixels_per_cm" id="ppcField" value="{{ pixels_per_cm or '' }}">
                <input type="hidden" name="length_type" id="ltField" value="{{ length_type or 'AUTO' }}">
                <input type="hidden" name="girth_factor" id="gfField" value="{{ girth_factor or 3.1416 }}">
            </form>
        </div>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <h3>🔍 Analyzing fish...</h3>
            <p>Detecting fish and identifying species...</p>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #fffde7; border-radius: 8px; border-left: 4px solid #ff9800;">
            <h3>📏 Measurement Settings</h3>
            <form method="POST" enctype="multipart/form-data" id="settingsForm">
                <label>Pixels per cm: <input type="number" step="0.0001" min="0" name="pixels_per_cm" value="{{ pixels_per_cm or '' }}" style="width:150px;"></label>
                <label style="margin-left: 15px;">Length Type:
                    <select name="length_type">
                        <option value="AUTO" {% if length_type == 'AUTO' %}selected{% endif %}>AUTO</option>
                        <option value="TL" {% if length_type == 'TL' %}selected{% endif %}>TL</option>
                        <option value="FL" {% if length_type == 'FL' %}selected{% endif %}>FL</option>
                        <option value="LJFL" {% if length_type == 'LJFL' %}selected{% endif %}>LJFL</option>
                    </select>
                </label>
                <label style="margin-left: 15px;">Girth Factor:
                    <input type="number" step="0.0001" name="girth_factor" value="{{ girth_factor or 3.1416 }}" style="width:150px;">
                </label>
            </form>
        </div>
        
        {% if results %}
            {% if results.success %}
                <div class="summary-card">
                    <h2>🎯 Detection Complete!</h2>
                    <h3>{{ results.fish_count }} Fish Detected and Identified</h3>
                </div>
                
                <div class="result-container">
                    {% if results.annotated_image %}
                        <div class="result-image">
                            <h3>📷 Visual Detection Results</h3>
                            <img src="/static/{{ results.annotated_image }}" alt="Fish Detection Results">
                            <p><small>Fish are outlined with bounding boxes, segmentation polygons, labels, and face markers</small></p>
                        </div>
                    {% endif %}
                    
                    <div class="result-details">
                        <h3>📊 Detailed Identification Results</h3>
                        
                        {% for fish in results.fish %}
                            <div class="fish-item">
                                <div class="fish-header">🐠 Fish #{{ fish.fish_id }}</div>
                                <div class="species-name">{{ fish.common_name or fish.species }}{% if fish.scientific_name %} ({{ fish.scientific_name }}){% endif %}</div>
                                <p><strong>Classification Accuracy:</strong> 
                                    <span class="{% if fish.accuracy >= 0.8 %}accuracy-high{% elif fish.accuracy >= 0.6 %}accuracy-medium{% else %}accuracy-low{% endif %}">
                                        {{ (fish.accuracy * 100)|round(1) }}%
                                    </span>
                                </p>
            <p><strong>Detection Confidence:</strong> {{ (fish.confidence * 100)|round(1) }}%</p>
            <p><strong>Location:</strong> Box [{{ fish.box[0] }}, {{ fish.box[1] }}, {{ fish.box[2] }}, {{ fish.box[3] }}]</p>
            {% if fish.segmentation %}
                <p><strong>Segmentation Points:</strong> {{ fish.segmentation.points|length }}</p>
            {% endif %}
            {% if fish.face_box %}
                <p><strong>Face Box:</strong> [{{ fish.face_box[0] }}, {{ fish.face_box[1] }}, {{ fish.face_box[2] }}, {{ fish.face_box[3] }}]</p>
            {% endif %}
            {% if fish.measurements %}
                <p><strong>Length Type:</strong> {{ fish.measurements.length_type }}</p>
                <p><strong>Length:</strong> {{ fish.measurements.length_px }} px{% if fish.measurements.length_cm %} ({{ fish.measurements.length_cm }} cm){% endif %}</p>
                <p><strong>Girth:</strong> {{ fish.measurements.girth_px }} px{% if fish.measurements.girth_cm %} ({{ fish.measurements.girth_cm }} cm){% endif %}</p>
                {% if fish.measurements.weight_kg_lgg_over_800 %}
                    <p><strong>Weight L*G^2/800:</strong> {{ fish.measurements.weight_kg_lgg_over_800 }} kg</p>
                {% endif %}
                {% if fish.measurements.weight_kg_l3_over_1200 %}
                    <p><strong>Weight L^3/1200:</strong> {{ fish.measurements.weight_kg_l3_over_1200 }} kg</p>
                {% endif %}
                {% if fish.measurements.rule_weight_kg_min %}
                    <p><strong>Rule Weight:</strong> {{ fish.measurements.rule_weight_kg_min }}–{{ fish.measurements.rule_weight_kg_max }} kg</p>
                    <p><strong>Typical Length:</strong> {{ fish.measurements.rule_length_cm_min }}–{{ fish.measurements.rule_length_cm_max }} cm</p>
                {% endif %}
                {% if fish.measurements.frame_distance_px %}
                    <p><strong>Frame Distance:</strong> 
                        L {{ fish.measurements.frame_distance_px.left|round(1) }} px, 
                        R {{ fish.measurements.frame_distance_px.right|round(1) }} px, 
                        T {{ fish.measurements.frame_distance_px.top|round(1) }} px, 
                        B {{ fish.measurements.frame_distance_px.bottom|round(1) }} px
                        {% if fish.measurements.frame_distance_cm %}
                            ({{ fish.measurements.frame_distance_cm.left }} / {{ fish.measurements.frame_distance_cm.right }} / {{ fish.measurements.frame_distance_cm.top }} / {{ fish.measurements.frame_distance_cm.bottom }} cm)
                        {% endif %}
                    </p>
                {% endif %}
            {% endif %}
        </div>
        {% endfor %}
    </div>
</div>
            {% else %}
                <div class="error">
                    <h2>❌ No Fish Detected</h2>
                    <p>{{ results.error }}</p>
                    <p>Try uploading a clearer image with visible fish.</p>
                </div>
            {% endif %}
        {% endif %}
        
        <div style="margin-top: 30px; padding: 20px; background: #e3f2fd; border-radius: 8px;">
            <h3>🔗 API Integration</h3>
            <p><strong>POST</strong> to <code>/api</code> with 'file' parameter for programmatic access</p>
            <p><strong>Health Check:</strong> <a href="/health">/health</a></p>
            <p><strong>Example:</strong> <code>curl -X POST -F "file=@fish.jpg" http://localhost:5001/api</code></p>
        </div>
    </div>

    <script>
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');
        const uploadForm = document.getElementById('uploadForm');
        const loading = document.getElementById('loading');

        // Drag and drop functionality
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadZone.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            uploadZone.classList.add('dragover');
        }

        function unhighlight(e) {
            uploadZone.classList.remove('dragover');
        }

        uploadZone.addEventListener('drop', handleDrop, false);

        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelect({ target: { files: files } });
            }
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                // Show loading
                loading.style.display = 'block';
                uploadZone.style.display = 'none';
                
                const ppcInput = document.querySelector('input[name="pixels_per_cm"]');
                const ltInput = document.querySelector('select[name="length_type"]');
                const gfInput = document.querySelector('input[name="girth_factor"]');
                document.getElementById('ppcField').value = ppcInput ? ppcInput.value : '';
                document.getElementById('ltField').value = ltInput ? ltInput.value : 'AUTO';
                document.getElementById('gfField').value = gfInput ? gfInput.value : '3.1416';
                
                // Submit form
                uploadForm.submit();
            }
        }
        
        // Auto-scroll to results if they exist
        {% if results %}
            setTimeout(() => {
                const resultsSection = document.querySelector('.summary-card');
                if (resultsSection) {
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                }
            }, 500);
        {% endif %}
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    yolo_ok, class_ok, seg_ok, face_ok = test_models()
    yolo_status = "✅ Working" if yolo_ok else "❌ Error"
    class_status = "✅ Working" if class_ok else "❌ Error"
    seg_status = "✅ Working" if seg_ok else "❌ Error"
    face_status = "✅ Working" if face_ok else "❌ Error"
    
    results = None
    pixels_per_cm = None
    length_type = 'AUTO'
    girth_factor = pi
    db_count = None
    try:
        with open(CLASS_INFO_PATH, 'r') as f:
            info = json.load(f)
            db_count = info.get('num_of_class')
    except Exception:
        db_count = None
    
    if request.method == 'POST':
        if request.form.get('pixels_per_cm'):
            try:
                pixels_per_cm = float(request.form.get('pixels_per_cm'))
            except Exception:
                pixels_per_cm = None
        if request.form.get('length_type'):
            length_type = request.form.get('length_type') or 'AUTO'
        if request.form.get('girth_factor'):
            try:
                girth_factor = float(request.form.get('girth_factor'))
            except Exception:
                girth_factor = pi
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            if not (yolo_ok and class_ok and seg_ok and face_ok):
                flash('Models not ready. Please check console for errors.')
                return redirect(request.url)
                
            try:
                # Save uploaded file temporarily
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = secure_filename(file.filename)
                temp_filename = f"{timestamp}_{filename}"
                temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
                file.save(temp_path)
                
                # Process the image
                results = process_image_external(temp_path, pixels_per_cm=pixels_per_cm, length_type=length_type, girth_factor=girth_factor)
                
                # Clean up uploaded file
                os.unlink(temp_path)
                
            except Exception as e:
                results = {"error": f"Server error: {str(e)}"}
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, GIF, or BMP files.')
            return redirect(request.url)
    
    return render_template_string(HTML_TEMPLATE, 
                                results=results, 
                                yolo_status=yolo_status, 
                                class_status=class_status,
                                seg_status=seg_status,
                                face_status=face_status,
                                pixels_per_cm=pixels_per_cm,
                                length_type=length_type,
                                girth_factor=girth_factor,
                                db_count=db_count)

@app.route('/static/<path:filename>')
def static_files(filename):
    """Serve static files (annotated images)."""
    return send_file(os.path.join(STATIC_FOLDER, filename))

@app.route('/api', methods=['POST'])
def api():
    """API endpoint for fish identification."""
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400
    
    file = request.files['file']
    if not allowed_file(file.filename):
        return {"error": "Invalid file type"}, 400
    
    yolo_ok, class_ok, seg_ok, face_ok = test_models()
    if not (yolo_ok and class_ok and seg_ok and face_ok):
        return {"error": "Models not ready"}, 503
    
    try:
        pixels_per_cm = request.form.get('pixels_per_cm')
        length_type = request.form.get('length_type') or 'AUTO'
        girth_factor = request.form.get('girth_factor')
        ppc = None
        gf = pi
        try:
            ppc = float(pixels_per_cm) if pixels_per_cm else None
        except Exception:
            ppc = None
        try:
            gf = float(girth_factor) if girth_factor else pi
        except Exception:
            gf = pi
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg', dir=UPLOAD_FOLDER) as tmp:
            file.save(tmp.name)
            results = process_image_external(tmp.name, pixels_per_cm=ppc, length_type=length_type, girth_factor=gf)
            os.unlink(tmp.name)
        return results
    except Exception as e:
        return {"error": f"Server error: {str(e)}"}, 500

@app.route('/health')
def health():
    """Health check endpoint."""
    yolo_ok, class_ok, seg_ok, face_ok = test_models()
    return {
        "status": "healthy" if (yolo_ok and class_ok and seg_ok and face_ok) else "degraded",
        "yolo_detector": "✅ Working" if yolo_ok else "❌ Error",
        "fish_classifier": "✅ Working" if class_ok else "❌ Error",
        "segmentation": "✅ Working" if seg_ok else "❌ Error",
        "face_detector": "✅ Working" if face_ok else "❌ Error",
        "models_ready": yolo_ok and class_ok and seg_ok and face_ok,
        "features": ["visual_bounding_boxes", "drag_drop", "species_labeling", "segmentation_masks", "face_detection"],
        "api_version": "2.0"
    }

if __name__ == '__main__':
    print("🐟 Starting Enhanced Fish Identification Web App...")
    print("🔍 Testing models...")
    
    yolo_ok, class_ok, seg_ok, face_ok = test_models()
    
    print(f"📍 YOLO Detection: {'✅ Working' if yolo_ok else '❌ Error'}")
    print(f"🔬 Classification: {'✅ Working' if class_ok else '❌ Error'}")
    print(f"🎨 Segmentation: {'✅ Working' if seg_ok else '❌ Error'}")
    print(f"🙂 Face Detection: {'✅ Working' if face_ok else '❌ Error'}")
    
    if not (yolo_ok and class_ok and seg_ok and face_ok):
        print("\n⚠️  Some models have issues, but starting server anyway...")
        print("   Check the /health endpoint for status")
    else:
        print("\n🎉 All models working perfectly!")
    
    print("\n🚀 Starting enhanced web server...")
    print("📱 Open your browser and go to: http://localhost:5001")
    print("✨ Features: Drag & Drop + Bounding Boxes + Segmentation + Species Labels + Face Detection")
    print("🔗 API endpoint: POST to http://localhost:5001/api")
    print("🩺 Health check: http://localhost:5001/health")
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    
    app.run(host='0.0.0.0', port=5001, debug=True) 
