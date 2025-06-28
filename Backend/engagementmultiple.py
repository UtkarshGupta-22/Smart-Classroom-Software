import cv2
import numpy as np
import time
import os
from datetime import datetime
from collections import deque
import mediapipe as mp
import matplotlib
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import json


def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    return obj

import argparse

ENGAGEMENT_WINDOW = 30
EYE_AR_THRESHOLD = 0.2
MOUTH_AR_THRESHOLD = 0.5
GAZE_THRESHOLD = 0.2
TALKING_THRESHOLD = 0.08
CONFUSION_THRESHOLD = 0.4
STRESS_THRESHOLD = 0.3
ATTENTION_SPANS = [60, 300, 600]
ENGAGEMENT_THRESHOLD = 60
CALIBRATION_TIME = 5

engagement_history = deque(maxlen=ENGAGEMENT_WINDOW)
attention_spans = {span: deque(maxlen=span) for span in ATTENTION_SPANS}
session_stats = {
    "total_time": 0,
    "engaged_time": 0,
    "disengaged_time": 0,
    "attention_span": 0,
    "max_attention_span": 0,
    "breaks": 0,
    "face_time_percentage": 0,
    "engagement_drops": []
}

user_settings = {
    "user_name": "Student",
    "session_goal": 30,
    "break_interval": 25,
    "calibration": {
        "baseline_eye_ar": 0,
        "baseline_mouth_ar": 0,
        "baseline_head_pose": [0, 0, 0]
    }
}

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
base_dir = f"engagement_session_{timestamp_str}"
os.makedirs(base_dir, exist_ok=True)
log_file = f"{base_dir}/engagement_log.csv"
settings_file = f"{base_dir}/user_settings.json"
stats_file = f"{base_dir}/session_stats.json"
model_file = f"{base_dir}/engagement_model.joblib"
chart_file = f"{base_dir}/engagement_chart.png"


with open(log_file, 'w') as f:
    f.write(
        "timestamp,face_detected,looking_away,eyes_closed,yawning,talking,confused,stressed,engagement_score,attention_level,activity\n")

with open(settings_file, 'w') as f:
    json.dump(user_settings, f, indent=4)

feature_history = []
start_time = time.time()
current_activity = "Learning"

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=20,  # adjustable
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils



def calculate_eye_aspect_ratio(eye_points):
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C) if C > 0 else 0
    return ear


def calculate_mouth_aspect_ratio(mouth_points):
    A = np.linalg.norm(mouth_points[2] - mouth_points[8])
    B = np.linalg.norm(mouth_points[3] - mouth_points[7])
    C = np.linalg.norm(mouth_points[4] - mouth_points[6])
    D = np.linalg.norm(mouth_points[0] - mouth_points[10])
    mar = (A + B + C) / (3.0 * D) if D > 0 else 0
    return mar


def get_engagement_score(behaviors, attention_history=None):
    weights = {
        'face_detected': 100,
        'looking_away': -30,
        'eyes_closed': -40,
        'yawning': -20,
        'talking': 5,
        'confused': -15,
        'stressed': -15
    }

    score = weights['face_detected'] if behaviors['face_detected'] else 0
    for behavior, detected in behaviors.items():
        if behavior != 'face_detected' and detected:
            score += weights[behavior]

    if attention_history and len(attention_history) > 0:
        alpha = 0.7
        prev_score = attention_history[-1]
        score = alpha * score + (1 - alpha) * prev_score

    attention_level = "High" if score > 75 else "Medium" if score > 40 else "Low"

    return max(0, min(100, score)), attention_level


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

NOSE_TIP = 1
LEFT_EYE_LEFT_CORNER = 130
RIGHT_EYE_RIGHT_CORNER = 359
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
CHIN = 152


def get_landmark_coords(landmarks, indexes, image_shape):
    h, w = image_shape
    return np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indexes])


def get_3d_landmark_coords(landmarks, indexes):
    return np.array([(landmarks[i].x, landmarks[i].y, landmarks[i].z) for i in indexes])


def detect_looking_away(landmarks, shape):
    h, w = shape

    left_iris_points = np.mean([(landmarks[i].x, landmarks[i].y) for i in LEFT_IRIS], axis=0)
    right_iris_points = np.mean([(landmarks[i].x, landmarks[i].y) for i in RIGHT_IRIS], axis=0)

    left_eye_left = (landmarks[33].x, landmarks[33].y)
    left_eye_right = (landmarks[133].y, landmarks[133].y)

    right_eye_left = (landmarks[362].x, landmarks[362].y)
    right_eye_right = (landmarks[263].x, landmarks[263].y)

    left_ratio = (left_iris_points[0] - left_eye_left[0]) / (left_eye_right[0] - left_eye_left[0]) if left_eye_right[
                                                                                                          0] != \
                                                                                                      left_eye_left[
                                                                                                          0] else 0.5
    right_ratio = (right_iris_points[0] - right_eye_left[0]) / (right_eye_right[0] - right_eye_left[0]) if \
    right_eye_right[0] != right_eye_left[0] else 0.5

    looking_left = left_ratio < 0.4 and right_ratio < 0.4
    looking_right = left_ratio > 0.6 and right_ratio > 0.6

    return looking_left or looking_right


def detect_confused(landmarks):
    nose = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y, landmarks[NOSE_TIP].z])
    left_eye = np.array(
        [landmarks[LEFT_EYE_LEFT_CORNER].x, landmarks[LEFT_EYE_LEFT_CORNER].y, landmarks[LEFT_EYE_LEFT_CORNER].z])
    right_eye = np.array(
        [landmarks[RIGHT_EYE_RIGHT_CORNER].x, landmarks[RIGHT_EYE_RIGHT_CORNER].y, landmarks[RIGHT_EYE_RIGHT_CORNER].z])

    eye_vector = right_eye - left_eye
    horizontal_vector = np.array([1, 0, 0])

    eye_vector = eye_vector / np.linalg.norm(eye_vector)

    dot_product = np.dot(eye_vector[:2], horizontal_vector[:2])
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_degrees = np.degrees(angle)

    head_tilted = abs(angle_degrees - 90) > CONFUSION_THRESHOLD * 90

    return head_tilted


def detect_talking(current_mar, prev_mar, mouth_movement_history):
    mouth_movement = abs(current_mar - prev_mar)
    mouth_movement_history.append(mouth_movement)

    if len(mouth_movement_history) >= 5:
        recent_movements = list(mouth_movement_history)[-5:]
        movement_variance = np.var(recent_movements)
        return movement_variance > TALKING_THRESHOLD and current_mar < MOUTH_AR_THRESHOLD

    return False


def calibrate_system(capture):
    print("Starting calibration (5 seconds)...")
    calibration_data = {
        "eye_ar_samples": [],
        "mouth_ar_samples": [],
        "head_pose_samples": []
    }

    end_time = time.time() + CALIBRATION_TIME

    while time.time() < end_time:
        ret, frame = capture.read()
        if not ret:
            print("Error reading frame during calibration.")
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            shape = frame.shape[:2]

            left_eye = get_landmark_coords(landmarks, LEFT_EYE_IDX, shape)
            right_eye = get_landmark_coords(landmarks, RIGHT_EYE_IDX, shape)
            mouth = get_landmark_coords(landmarks, MOUTH_IDX, shape)

            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            mouth_ar = calculate_mouth_aspect_ratio(mouth)

            nose = np.array([landmarks[NOSE_TIP].x, landmarks[NOSE_TIP].y, landmarks[NOSE_TIP].z])
            left_eye_point = np.array([landmarks[LEFT_EYE_LEFT_CORNER].x, landmarks[LEFT_EYE_LEFT_CORNER].y,
                                       landmarks[LEFT_EYE_LEFT_CORNER].z])
            right_eye_point = np.array([landmarks[RIGHT_EYE_RIGHT_CORNER].x, landmarks[RIGHT_EYE_RIGHT_CORNER].y,
                                        landmarks[RIGHT_EYE_RIGHT_CORNER].z])

            calibration_data["eye_ar_samples"].append(avg_ear)
            calibration_data["mouth_ar_samples"].append(mouth_ar)
            calibration_data["head_pose_samples"].append([
                nose[0], nose[1], nose[2]
            ])

            remaining = max(0, int(end_time - time.time()))
            cv2.putText(frame, f"Calibration: {remaining}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Look directly at the camera", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        else:
            cv2.putText(frame, "No face detected - please look at the camera", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        cv2.imshow('Calibration', frame)
        cv2.waitKey(1)

    if calibration_data["eye_ar_samples"]:
        for key in ["eye_ar_samples", "mouth_ar_samples"]:
            values = np.array(calibration_data[key])
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered = values[(values >= lower_bound) & (values <= upper_bound)]

            if len(filtered) > 0:
                if key == "eye_ar_samples":
                    user_settings["calibration"]["baseline_eye_ar"] = float(np.mean(filtered))
                elif key == "mouth_ar_samples":
                    user_settings["calibration"]["baseline_mouth_ar"] = float(np.mean(filtered))

        if calibration_data["head_pose_samples"]:
            head_poses = np.array(calibration_data["head_pose_samples"])
            user_settings["calibration"]["baseline_head_pose"] = [
                float(np.mean(head_poses[:, 0])),
                float(np.mean(head_poses[:, 1])),
                float(np.mean(head_poses[:, 2]))
            ]

    with open(settings_file, 'w') as f:
        json.dump(user_settings, f, indent=4)

    print("Calibration complete!")
    return user_settings["calibration"]


def generate_engagement_chart(history_data):
    times = []
    scores = []
    events = []
    event_times = []
    event_scores = []
    event_labels = []

    for entry in history_data:
        times.append(entry["elapsed_time"])
        scores.append(entry["score"])

        if entry.get("event"):
            events.append(entry["event"])
            event_times.append(entry["elapsed_time"])
            event_scores.append(entry["score"])
            event_labels.append(entry["event"])

    plt.figure(figsize=(12, 6))

    plt.plot(times, scores, 'b-', linewidth=2)

    plt.axhline(y=75, color='g', linestyle='--', alpha=0.7, label='High Engagement')
    plt.axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Medium Engagement')

    if event_times:
        plt.scatter(event_times, event_scores, color='red', s=50, zorder=5)

        for i, label in enumerate(event_labels):
            plt.annotate(label, (event_times[i], event_scores[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center')

    plt.xlabel('Time (seconds)')
    plt.ylabel('Engagement Score (%)')
    plt.title(f'Engagement Analysis - {user_settings["user_name"]}')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)

    plt.legend()

    plt.savefig(chart_file, dpi=100, bbox_inches='tight')
    plt.close()


def train_engagement_model(features, labels):
    try:
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(features, labels)


        joblib.dump(model, model_file)
        print("Engagement prediction model trained and saved.")
        return model
    except Exception as e:
        print(f"Error training model: {e}")
        return None


def process_frame(frame, calibration_data=None, ml_model=None):
    behaviors = {key: False for key in
                 ['face_detected', 'looking_away', 'eyes_closed', 'yawning', 'talking', 'confused', 'stressed']}
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if not hasattr(process_frame, 'mouth_movement_history'):
        process_frame.mouth_movement_history = deque(maxlen=10)
    if not hasattr(process_frame, 'feature_history'):
        process_frame.feature_history = []
    if not hasattr(process_frame, 'continuous_attention_time'):
        process_frame.continuous_attention_time = 0
    if not hasattr(process_frame, 'last_activity_time'):
        process_frame.last_activity_time = time.time()
    if not hasattr(process_frame, 'engagement_data_points'):
        process_frame.engagement_data_points = []

    elapsed_time = time.time() - start_time

    current_features = []
    attention_level = "Low"

    if results.multi_face_landmarks:
        behaviors['face_detected'] = True
        landmarks = results.multi_face_landmarks[0].landmark
        shape = frame.shape[:2]

        left_eye = get_landmark_coords(landmarks, LEFT_EYE_IDX, shape)
        right_eye = get_landmark_coords(landmarks, RIGHT_EYE_IDX, shape)
        mouth = get_landmark_coords(landmarks, MOUTH_IDX, shape)

        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        mouth_ar = calculate_mouth_aspect_ratio(mouth)

        if calibration_data:
            eye_threshold = calibration_data["baseline_eye_ar"] * 0.7  # 70% of baseline is closed
            mouth_threshold = calibration_data["baseline_mouth_ar"] * 1.5  # 150% of baseline is yawning
        else:
            eye_threshold = EYE_AR_THRESHOLD
            mouth_threshold = MOUTH_AR_THRESHOLD

        behaviors['eyes_closed'] = avg_ear < eye_threshold
        behaviors['yawning'] = mouth_ar > mouth_threshold
        behaviors['looking_away'] = detect_looking_away(landmarks, shape)
        behaviors['confused'] = detect_confused(landmarks)

        if hasattr(process_frame, 'prev_mouth_ar'):
            behaviors['talking'] = detect_talking(mouth_ar, process_frame.prev_mouth_ar,
                                                  process_frame.mouth_movement_history)
        process_frame.prev_mouth_ar = mouth_ar

        if not hasattr(process_frame, 'blink_history'):
            process_frame.blink_history = []

        if hasattr(process_frame, 'prev_eyes_closed'):
            if not process_frame.prev_eyes_closed and behaviors['eyes_closed']:
                process_frame.blink_history.append(time.time())

        current_time = time.time()
        process_frame.blink_history = [t for t in process_frame.blink_history if current_time - t < 10]

        blink_rate = len(process_frame.blink_history) * 6  # Multiply by 6 to get per minute rate
        behaviors['stressed'] = blink_rate > 25  # High blink rate can indicate stress

        process_frame.prev_eyes_closed = behaviors['eyes_closed']

        current_features = [
            avg_ear,
            mouth_ar,
            int(behaviors['looking_away']),
            int(behaviors['eyes_closed']),
            int(behaviors['yawning']),
            int(behaviors['talking']),
            int(behaviors['confused']),
            int(behaviors['stressed']),
            blink_rate
        ]

        for pt in np.concatenate([left_eye, right_eye, mouth]):
            cv2.circle(frame, tuple(pt), 2, (0, 255, 0), -1)

    score, attention_level = get_engagement_score(behaviors, engagement_history)
    engagement_history.append(score)
    avg_score = np.mean(engagement_history)

    ml_prediction = None
    if ml_model is not None and current_features:
        try:
            ml_prediction = ml_model.predict([current_features])[0]
            attention_level = ml_prediction  # Use ML prediction instead
        except Exception as e:
            print(f"Error making prediction: {e}")

    for span in ATTENTION_SPANS:
        attention_spans[span].append(score)

    if score >= ENGAGEMENT_THRESHOLD:
        process_frame.continuous_attention_time += 1
    else:
        if process_frame.continuous_attention_time > session_stats["max_attention_span"]:
            session_stats["max_attention_span"] = process_frame.continuous_attention_time
        process_frame.continuous_attention_time = 0

    session_stats["total_time"] = int(elapsed_time)
    if behaviors['face_detected']:
        session_stats["engaged_time"] += 1 if score >= ENGAGEMENT_THRESHOLD else 0
        session_stats["disengaged_time"] += 1 if score < ENGAGEMENT_THRESHOLD else 0
    session_stats["attention_span"] = process_frame.continuous_attention_time

    if session_stats["total_time"] > 0:
        session_stats["face_time_percentage"] = (session_stats["engaged_time"] + session_stats["disengaged_time"]) / \
                                                session_stats["total_time"] * 100

    if hasattr(process_frame, 'prev_score') and process_frame.prev_score - score > 25:  # Sharp drop
        drop_event = {
            "time": elapsed_time,
            "from_score": process_frame.prev_score,
            "to_score": score,
            "behaviors": {k: v for k, v in behaviors.items()}
        }
        session_stats["engagement_drops"].append(drop_event)

    process_frame.prev_score = score

    data_point = {
        "elapsed_time": elapsed_time,
        "score": score,
        "behaviors": {k: v for k, v in behaviors.items()},
        "attention_level": attention_level
    }

    current_activity_time = time.time()
    if current_activity_time - process_frame.last_activity_time > 60:  # Activity change detection (placeholder)
        data_point["event"] = f"Activity: {current_activity}"
        process_frame.last_activity_time = current_activity_time

    process_frame.engagement_data_points.append(data_point)

    if behaviors['face_detected'] and current_features:
        feature_history.append({
            "features": current_features,
            "score": score,
            "attention_level": attention_level
        })

    y_pos = 30
    cv2.putText(frame, f"Activity: {current_activity} ({int(elapsed_time // 60)}:{int(elapsed_time % 60):02d})",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_pos += 30

    attention_colors = {"High": (0, 255, 0), "Medium": (0, 165, 255), "Low": (0, 0, 255)}
    cv2.putText(frame, f"Attention: {attention_level}",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, attention_colors.get(attention_level, (255, 255, 255)), 2)
    y_pos += 30

    cv2.putText(frame,
                f"Focus Streak: {process_frame.continuous_attention_time}s (Max: {session_stats['max_attention_span']}s)",
                (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_pos += 30

    for k, v in behaviors.items():
        if k == 'face_detected': continue
        status = "YES" if v else "NO"
        color = (0, 0, 255) if v and k != 'talking' else (0, 255, 0)
        # For talking, we use green for YES (it's positive) and red for NO
        if k == 'talking':
            color = (0, 255, 0) if v else (0, 0, 255)
        cv2.putText(frame, f"{k.replace('_', ' ').title()}: {status}", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 2)
        y_pos += 30

    cv2.putText(frame, f"Engagement: {score:.1f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    y_pos += 30
    cv2.putText(frame, f"Avg Engagement: {avg_score:.1f}%", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    meter_width = int(frame.shape[1] * 0.8)
    meter_height = 20
    meter_x = int((frame.shape[1] - meter_width) / 2)
    meter_y = frame.shape[0] - 50

    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (50, 50, 50), -1)

    filled_width = int(meter_width * (score / 100))
    if score > 75:  # High engagement
        color = (0, 255, 0)  # Green
    elif score > 40:  # Medium engagement
        color = (0, 165, 255)  # Orange
    else:  # Low engagement
        color = (0, 0, 255)  # Red

    cv2.rectangle(frame, (meter_x, meter_y), (meter_x + filled_width, meter_y + meter_height), color, -1)

    cv2.line(frame, (meter_x + int(0.4 * meter_width), meter_y),
             (meter_x + int(0.4 * meter_width), meter_y + meter_height), (200, 200, 200), 1)
    cv2.line(frame, (meter_x + int(0.75 * meter_width), meter_y),
             (meter_x + int(0.75 * meter_width), meter_y + meter_height), (200, 200, 200), 1)

    cv2.putText(frame, "Engagement Level", (meter_x, meter_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    current_time = time.time()
    if not hasattr(process_frame, 'last_log_time') or current_time - process_frame.last_log_time >= 1.0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_data = [
            timestamp,
            int(behaviors['face_detected']),
            int(behaviors['looking_away']),
            int(behaviors['eyes_closed']),
            int(behaviors['yawning']),
            int(behaviors['talking']),
            int(behaviors['confused']),
            int(behaviors['stressed']),
            f"{score:.1f}",
            attention_level,
            current_activity
        ]
        with open(log_file, 'a') as f:
            f.write(','.join(map(str, log_data)) + '\n')
        process_frame.last_log_time = current_time

    if not hasattr(process_frame, 'last_stats_save') or current_time - process_frame.last_stats_save >= 10.0:
        with open(stats_file, 'w') as f:
            json.dump(convert_np_types(session_stats), f, indent=4)
        process_frame.last_stats_save = current_time

    return frame

def process_frame_multi(frame, calibration_data=None):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    engagement_scores = []
    face_boxes = []

    if results.multi_face_landmarks:
        h, w = frame.shape[:2]

        for idx, landmarks in enumerate(results.multi_face_landmarks):
            left_eye = get_landmark_coords(landmarks.landmark, LEFT_EYE_IDX, (h, w))
            right_eye = get_landmark_coords(landmarks.landmark, RIGHT_EYE_IDX, (h, w))
            mouth = get_landmark_coords(landmarks.landmark, MOUTH_IDX, (h, w))

            left_ear = calculate_eye_aspect_ratio(left_eye)
            right_ear = calculate_eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            mouth_ar = calculate_mouth_aspect_ratio(mouth)

            eye_threshold = calibration_data["baseline_eye_ar"] * 0.7 if calibration_data else EYE_AR_THRESHOLD
            mouth_threshold = calibration_data["baseline_mouth_ar"] * 1.5 if calibration_data else MOUTH_AR_THRESHOLD

            behaviors = {
                'face_detected': True,
                'looking_away': detect_looking_away(landmarks.landmark, (h, w)),
                'eyes_closed': avg_ear < eye_threshold,
                'yawning': mouth_ar > mouth_threshold,
                'talking': False,  # Skip for multi-face for speed
                'confused': detect_confused(landmarks.landmark),
                'stressed': False  # Optional: skip blink detection for demo
            }

            score, _ = get_engagement_score(behaviors)
            engagement_scores.append(score)

            # Draw bounding box
            xs = [int(lm.x * w) for lm in landmarks.landmark]
            ys = [int(lm.y * h) for lm in landmarks.landmark]
            x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)

            face_boxes.append(((x_min, y_min, x_max, y_max), score))

    # Draw all faces with engagement scores
    for i, (box, score) in enumerate(face_boxes):
        x1, y1, x2, y2 = box
        color = (0, 255, 0) if score > 75 else (0, 165, 255) if score > 40 else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"S{i+1}: {int(score)}%", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display cumulative score
    if engagement_scores:
        avg_score = sum(engagement_scores) / len(engagement_scores)
        cv2.putText(frame, f"Class Avg Engagement: {int(avg_score)}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame



def keyboard_controls(key):
    global current_activity

    if key == ord('l'):
        current_activity = "Learning"
        print("Activity changed to: Learning")
    elif key == ord('r'):
        current_activity = "Reading"
        print("Activity changed to: Reading")
    elif key == ord('e'):
        current_activity = "Exercise"
        print("Activity changed to: Exercise")
    elif key == ord('b'):
        current_activity = "Break"
        print("Activity changed to: Break")
    elif key == ord('t'):
        current_activity = "Test"
        print("Activity changed to: Test")
    elif key == ord('g'):
        generate_engagement_chart(process_frame.engagement_data_points)
        print(f"Chart saved to {chart_file}")
    elif key == ord('s'):
        with open(stats_file, 'w') as f:
            json.dump(convert_np_types(session_stats), f, indent=4)
        print(f"Statistics saved to {stats_file}")
    elif key == ord('m'):
        if len(feature_history) > 30:
            X = [entry["features"] for entry in feature_history]
            y = [entry["attention_level"] for entry in feature_history]
            trained_model = train_engagement_model(X, y)
            print("ML model trained with current session data")
            return trained_model

    return None


def analyze_session(data_points, stats):
    analysis = {
        "summary": "",
        "strengths": [],
        "areas_to_improve": [],
        "recommendations": []
    }

    activities = {}
    for point in data_points:
        activity = point.get("activity", "Unknown")
        if activity not in activities:
            activities[activity] = {"scores": [], "durations": []}
        activities[activity]["scores"].append(point["score"])

    best_activity = None
    worst_activity = None
    best_score = -1
    worst_score = 101

    for activity, data in activities.items():
        avg_score = np.mean(data["scores"]) if data["scores"] else 0
        if avg_score > best_score:
            best_score = avg_score
            best_activity = activity
        if avg_score < worst_score and avg_score > 0:
            worst_score = avg_score
            worst_activity = activity

    engaged_percentage = 0
    if stats["total_time"] > 0:
        engaged_percentage = (stats["engaged_time"] / stats["total_time"]) * 100

    total_minutes = stats["total_time"] / 60
    summary = (f"Session lasted {total_minutes:.1f} minutes with {engaged_percentage:.1f}% engagement. "
               f"Maximum continuous attention span was {stats['max_attention_span']} seconds.")
    analysis["summary"] = summary

    if engaged_percentage > 70:
        analysis["strengths"].append("High overall engagement throughout the session")
    if stats["max_attention_span"] > 120:
        analysis["strengths"].append("Excellent ability to maintain focus for extended periods")
    if best_activity:
        analysis["strengths"].append(f"Highest engagement during {best_activity} activities")

    if engaged_percentage < 50:
        analysis["areas_to_improve"].append("Overall engagement level needs improvement")
    if stats["max_attention_span"] < 60:
        analysis["areas_to_improve"].append("Short attention spans - difficulty maintaining continuous focus")
    if len(stats["engagement_drops"]) > 5:
        analysis["areas_to_improve"].append(
            f"Frequent engagement drops ({len(stats['engagement_drops'])} major drops detected)")
    if worst_activity:
        analysis["areas_to_improve"].append(f"Lower engagement during {worst_activity} activities")

    if stats["max_attention_span"] < 120:
        analysis["recommendations"].append(
            "Try the Pomodoro Technique: 25 minutes of focused work followed by a 5-minute break")
    if worst_activity:
        analysis["recommendations"].append(f"Consider different approaches for {worst_activity} to increase engagement")
    if len(stats["engagement_drops"]) > 5:
        analysis["recommendations"].append("Identify common distractions and create a more focused environment")

    return analysis


def show_feedback(feedback, frame=None):
    if frame is None:
        frame = np.zeros((600, 800, 3), dtype=np.uint8)

    overlay = frame.copy()
    cv2.rectangle(overlay, (50, 50), (frame.shape[1] - 50, frame.shape[0] - 50), (30, 30, 30), -1)

    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "Session Feedback", (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    lines = []
    summary = feedback["summary"]
    words = summary.split()
    current_line = ""
    for word in words:
        if len(current_line + " " + word) < 60:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    y_pos = 140
    for line in lines:
        cv2.putText(frame, line, (100, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y_pos += 30

    y_pos += 20
    cv2.putText(frame, "Strengths:", (100, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_pos += 30
    for strength in feedback["strengths"]:
        cv2.putText(frame, "✓ " + strength, (120, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y_pos += 30

    y_pos += 20
    cv2.putText(frame, "Areas to Improve:", (100, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    y_pos += 30
    for area in feedback["areas_to_improve"]:
        cv2.putText(frame, "! " + area, (120, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        y_pos += 30

    y_pos += 20
    cv2.putText(frame, "Recommendations:", (100, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_pos += 30
    for rec in feedback["recommendations"]:
        cv2.putText(frame, "→ " + rec, (120, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 30

    cv2.putText(frame, "Press any key to continue", (frame.shape[1] // 2 - 100, frame.shape[0] - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    return frame


def export_session_data():
    try:
        df = pd.read_csv(log_file)
        export_file = f"{base_dir}/engagement_analysis.xlsx"
        df.to_excel(export_file, index=False)
        print(f"Data exported to {export_file}")
        return export_file
    except Exception as e:
        print(f"Error exporting data: {e}")
        return None

use_multi_mode = False  # Default: single-face mode


def main():
    global use_multi_mode
    parser = argparse.ArgumentParser(description='Enhanced Student Engagement Monitor')
    parser.add_argument('--name', type=str, default="Student", help='Student name')
    parser.add_argument('--goal', type=int, default=30, help='Session goal in minutes')
    parser.add_argument('--break_interval', type=int, default=25, help='Recommended break interval in minutes')
    parser.add_argument('--skip_calibration', action='store_true', help='Skip the calibration step')
    parser.add_argument('--video', type=str, default="", help='Use video file instead of webcam')
    parser.add_argument('--show_guidelines', action='store_true', help='Show engagement guidelines at startup')
    args = parser.parse_args()

    user_settings["user_name"] = args.name
    user_settings["session_goal"] = args.goal
    user_settings["break_interval"] = args.break_interval

    print(f"Enhanced Student Engagement Monitor (MediaPipe Version)")
    print(f"User: {user_settings['user_name']}")
    print(f"Session goal: {user_settings['session_goal']} minutes")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.video}")
            return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not accessible.")
            return

    if args.show_guidelines:
        guidelines = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.putText(guidelines, "Engagement Guidelines", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(guidelines, "1. Sit in a well-lit area facing the camera", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1)
        cv2.putText(guidelines, "2. Try to maintain good posture", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1)
        cv2.putText(guidelines, "3. Take short breaks every 25 minutes", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1)
        cv2.putText(guidelines, "4. The system tracks:", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(guidelines, "   - Eye closure and gaze direction", (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1)
        cv2.putText(guidelines, "   - Mouth movements (talking, yawning)", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1)
        cv2.putText(guidelines, "   - Head position and orientation", (50, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 1)
        cv2.putText(guidelines, "5. Keyboard shortcuts:", (50, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(guidelines, "   - L: Learning activity", (50, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),
                    1)
        cv2.putText(guidelines, "   - R: Reading activity", (50, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200),
                    1)
        cv2.putText(guidelines, "   - B: Break", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(guidelines, "   - Q: Quit", (50, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(guidelines, "Press any key to continue...", (250, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1)

        cv2.imshow('Guidelines', guidelines)
        cv2.waitKey(0)
        cv2.destroyWindow('Guidelines')

    calibration_data = None
    if not args.skip_calibration:
        calibration_data = calibrate_system(cap)

    ml_model = None

    def chart_thread_function():
        while True:
            time.sleep(30)  # Update chart every 30 seconds
            if len(process_frame.engagement_data_points) > 10:
                generate_engagement_chart(process_frame.engagement_data_points)

    chart_thread = threading.Thread(target=chart_thread_function, daemon=True)
    chart_thread.start()

    show_feedback_flag = False
    feedback_data = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame.")
            break

        if show_feedback_flag and feedback_data:
            feedback_frame = show_feedback(feedback_data, frame.copy())
            cv2.imshow('Engagement Monitor', feedback_frame)
            if cv2.waitKey(1) & 0xFF != 255:  # Any key press
                show_feedback_flag = False
        else:
            if use_multi_mode:
                frame = process_frame_multi(frame, calibration_data)
            else:
                frame = process_frame(frame, calibration_data)

            cv2.imshow('Engagement Monitor', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                feedback_data = analyze_session(process_frame.engagement_data_points, session_stats)
                show_feedback_flag = True
            elif key == ord('m'):
                use_multi_mode = not use_multi_mode
                mode_text = "Multi-face mode ENABLED" if use_multi_mode else "Single-face mode ENABLED"
                print(f"[Toggle] {mode_text}")
            else:
                new_model = keyboard_controls(key)
                if new_model:
                    ml_model = new_model

    print("Generating final analytics...")

    def generate_engagement_chart(history_data):
        times = []
        scores = []
        events = []
        event_times = []
        event_scores = []
        event_labels = []

        attention_levels = {'High': [], 'Medium': [], 'Low': []}
        attention_times = {'High': [], 'Medium': [], 'Low': []}

        for entry in history_data:
            times.append(entry["elapsed_time"])
            scores.append(entry["score"])


            if "attention_level" in entry:
                level = entry["attention_level"]
                if level in attention_levels:
                    attention_levels[level].append(entry["score"])
                    attention_times[level].append(entry["elapsed_time"])

            if entry.get("event"):
                events.append(entry["event"])
                event_times.append(entry["elapsed_time"])
                event_scores.append(entry["score"])
                event_labels.append(entry["event"])


        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})


        ax1.plot(times, scores, 'b-', linewidth=2, label='Engagement Score')


        if len(scores) > 10:
            window_size = min(10, len(scores) // 5)
            moving_avg = np.convolve(scores, np.ones(window_size) / window_size, mode='valid')
            ma_times = times[window_size - 1:]
            ax1.plot(ma_times, moving_avg, 'r--', linewidth=1.5, label='Trend (Moving Avg)')


        ax1.axhline(y=75, color='g', linestyle='--', alpha=0.7, label='High Engagement')
        ax1.axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Medium Engagement')


        ax1.axhspan(75, 100, facecolor='lightgreen', alpha=0.2)
        ax1.axhspan(40, 75, facecolor='khaki', alpha=0.2)
        ax1.axhspan(0, 40, facecolor='lightcoral', alpha=0.2)


        if event_times:
            ax1.scatter(event_times, event_scores, color='red', s=50, zorder=5, marker='*')

            for i, label in enumerate(event_labels):

                ax1.axvline(x=event_times[i], color='gray', linestyle=':', alpha=0.5)

                ax1.annotate(label, (event_times[i], event_scores[i]),
                             textcoords="offset points", xytext=(0, 10), ha='center',
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))


        for level, color in zip(['High', 'Medium', 'Low'], ['green', 'orange', 'red']):
            if attention_times[level]:
                ax1.scatter(attention_times[level], attention_levels[level],
                            marker='o', color=color, alpha=0.5, s=30, label=f'{level} Attention')


        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Engagement Score (%)')
        ax1.set_title(f'Engagement Analysis - {user_settings["user_name"]}')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 105)
        ax1.legend(loc='upper right')


        behavior_data = {'looking_away': [], 'eyes_closed': [], 'yawning': [],
                         'talking': [], 'confused': [], 'stressed': []}
        behavior_times = {'looking_away': [], 'eyes_closed': [], 'yawning': [],
                          'talking': [], 'confused': [], 'stressed': []}


        for entry in history_data:
            if "behaviors" in entry:
                for behavior in behavior_data.keys():
                    if behavior in entry["behaviors"] and entry["behaviors"][behavior]:
                        behavior_data[behavior].append(1)
                        behavior_times[behavior].append(entry["elapsed_time"])


        behaviors_to_plot = behavior_data.keys()
        colors = ['red', 'purple', 'brown', 'green', 'orange', 'blue']
        markers = ['x', '+', 'o', '*', 's', 'd']


        y_positions = np.linspace(0.9, 0.1, len(behaviors_to_plot))

        for i, (behavior, color, marker) in enumerate(zip(behaviors_to_plot, colors, markers)):
            if behavior_times[behavior]:
                ax2.scatter(behavior_times[behavior],
                            [y_positions[i]] * len(behavior_times[behavior]),
                            marker=marker, color=color, label=behavior.replace('_', ' ').title(),
                            s=40, alpha=0.7)


        ax2.set_yticks(y_positions)
        ax2.set_yticklabels([b.replace('_', ' ').title() for b in behaviors_to_plot])
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Behavior Patterns')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(ax1.get_xlim())


        if times:
            max_time = max(times)
            interval = max(int(max_time / 10), 60)  # Make intervals at least 60 seconds
            time_markers = range(0, int(max_time) + interval, interval)
            for t in time_markers:
                ax2.axvline(x=t, color='gray', linestyle=':', alpha=0.5)

        plt.tight_layout()


        if len(scores) > 0:
            avg_score = np.mean(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)

            info_text = (f"Average: {avg_score:.1f}%  Max: {max_score:.1f}%  "
                         f"Min: {min_score:.1f}%  Duration: {max(times) / 60:.1f} min")
            fig.text(0.5, 0.01, info_text, ha='center', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))


        plt.savefig(chart_file, dpi=100, bbox_inches='tight')
        plt.close()


        if len(times) > 10:
            try:
                plt.figure(figsize=(12, 3))


                max_time = max(times)
                bins = min(50, len(times) // 2)

                hist_data, x_edges = np.histogram(times, bins=bins, weights=scores, density=False)
                hist_count, _ = np.histogram(times, bins=bins)


                hist_data = hist_data / np.maximum(hist_count, 1)


                heatmap_data = hist_data.reshape(1, -1)


                plt.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
                plt.colorbar(label='Engagement Level')


                x_ticks = np.linspace(0, bins - 1, 10).astype(int)
                x_labels = [f"{x_edges[i] / 60:.1f}" for i in x_ticks]
                plt.xticks(x_ticks, x_labels)

                plt.xlabel('Time (minutes)')
                plt.title('Engagement Heatmap')
                plt.yticks([])

                heatmap_file = chart_file.replace('.png', '_heatmap.png')
                plt.savefig(heatmap_file, dpi=100, bbox_inches='tight')
                plt.close()

                logger.info(f"Heatmap saved to {heatmap_file}")
            except Exception as e:
                logger.error(f"Error creating heatmap: {e}")

        logger.info(f"Engagement chart saved to {chart_file}")



if __name__ == "__main__":
    main()


