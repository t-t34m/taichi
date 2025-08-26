import os
import cv2
import math
import tempfile
import mediapipe as mp
import pandas as pd
from typing import Annotated
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import joblib
import xgboost


app = FastAPI()

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


def get_coords(landmarks, index):
    lm = landmarks[index]
    return (lm.x, lm.y)


def calculate_angle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    angle = math.degrees(
        math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx)
    )
    return abs(angle if angle >= 0 else angle + 360)


def calculate_distance(a, b):
    ax, ay = a
    bx, by = b
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


@app.post("/analyze_video/")
async def analyze_video(file: Annotated[UploadFile, File()]):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        return {"error": "FPS is 0, invalid video"}

    video_name = file.filename
    frame_results = {"video": video_name}

    # Target seconds (1–60)
    seconds_to_extract = range(1, 61)
    target_frames = {int(fps * s): s for s in seconds_to_extract}

    current_frame = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in target_frames:
            second = target_frames[current_frame]

            # Default fill = -1
            frame_results[f"right_arm_angle_s{second}"] = -1
            frame_results[f"left_arm_angle_s{second}"] = -1
            frame_results[f"right_leg_angle_s{second}"] = -1
            frame_results[f"left_leg_angle_s{second}"] = -1
            frame_results[f"core_angle_s{second}"] = -1
            frame_results[f"hand_distance_s{second}"] = -1
            frame_results[f"feet_distance_s{second}"] = -1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark

                    r_shoulder = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                    r_elbow    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ELBOW.value)
                    r_wrist    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_WRIST.value)

                    l_shoulder = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                    l_elbow    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value)
                    l_wrist    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value)

                    r_hip      = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_HIP.value)
                    r_knee     = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_KNEE.value)
                    r_ankle    = get_coords(landmarks, mp_pose.PoseLandmark.RIGHT_ANKLE.value)

                    l_hip      = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value)
                    l_knee     = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_KNEE.value)
                    l_ankle    = get_coords(landmarks, mp_pose.PoseLandmark.LEFT_ANKLE.value)

                    frame_results[f"right_arm_angle_s{second}"] = calculate_angle(r_shoulder, r_elbow, r_wrist)
                    frame_results[f"left_arm_angle_s{second}"]  = calculate_angle(l_shoulder, l_elbow, l_wrist)
                    frame_results[f"right_leg_angle_s{second}"] = calculate_angle(r_hip, r_knee, r_ankle)
                    frame_results[f"left_leg_angle_s{second}"]  = calculate_angle(l_hip, l_knee, l_ankle)
                    frame_results[f"core_angle_s{second}"]      = calculate_angle(r_hip, r_shoulder, l_shoulder)
                    frame_results[f"hand_distance_s{second}"]   = calculate_distance(r_wrist, l_wrist)
                    frame_results[f"feet_distance_s{second}"]   = calculate_distance(r_ankle, l_ankle)
                except Exception:
                    pass

        current_frame += 1

    cap.release()
    os.remove(tmp_path)

    # Convert results to DataFrame (one row)
    df = pd.DataFrame([frame_results]).fillna(-1)
    model = joblib.load("xgb_sklearn_model.pkl")
    probs = model.predict_proba(df.drop(columns=["video"]))[0].tolist()
    max_probs = float(probs[0])
    

    return JSONResponse(content={"probabilities": max_probs,})
