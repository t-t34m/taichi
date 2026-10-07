Taichi — Backend

An API that scores a Tai Chi performance from a one-minute video. It extracts body pose with MediaPipe Pose, computes joint angles and distances, and passes them to an XGBoost model that returns a probability score.

The web app that calls this API lives at t-t34m/taichi_FRONT.

Tech stack
FastAPI + Uvicorn — web framework and server
MediaPipe Pose — body landmark detection
OpenCV — video frame reading
XGBoost + pandas + joblib — model loading and prediction
How it works
Receives a video file and saves it to a temporary file
Samples one frame per second, from second 1 to second 60
Runs MediaPipe Pose on each sampled frame to find body landmarks
Computes 7 values per frame, for 420 features per video
Feeds them to an XGBClassifier (binary classification) and returns the result as JSON
Getting started

Requires Python 3.9 or later (pick a version MediaPipe supports).

bash
git clone https://github.com/t-t34m/taichi.git
cd taichi

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirement.txt   # note: no "s" in the file name
pip install scikit-learn         # needed by XGBClassifier but missing from requirement.txt

uvicorn app:app --reload --port 8000

Open http://localhost:8000/docs to try the API in Swagger UI.

Run uvicorn from the folder that contains xgb_sklearn_model.pkl. The code loads the model using a relative path.

API
POST /analyze_video/

Request — multipart/form-data

Field	Type	Description
file	video file	The video to analyze. Should be 60 seconds long.

Example

bash
curl -X POST http://localhost:8000/analyze_video/ \
  -F "file=@my_taichi.mp4"

Response

json
{ "probabilities": 0.8731 }

probabilities is the model's probability (0–1) for class 0.

If the video cannot be read, the API returns {"error": "FPS is 0, invalid video"} (the HTTP status is still 200).

Model features

For each second, the API computes 7 values from 2D landmark coordinates (normalized x, y):

Feature	Computed from
right_arm_angle_s{n}	Angle at the right elbow (shoulder–elbow–wrist)
left_arm_angle_s{n}	Angle at the left elbow (shoulder–elbow–wrist)
right_leg_angle_s{n}	Angle at the right knee (hip–knee–ankle)
left_leg_angle_s{n}	Angle at the left knee (hip–knee–ankle)
core_angle_s{n}	Angle at the right shoulder (right hip–right shoulder–left shoulder)
hand_distance_s{n}	Distance between the two wrists
feet_distance_s{n}	Distance between the two ankles

{n} is the second, from 1 to 60. Angles are in degrees (0–360). If no person is detected in a frame, or the video is shorter than 60 seconds, the missing values are filled with -1.

Project structure
taichi/
├── app.py                  # FastAPI app, feature extraction and prediction
├── requirement.txt         # Python dependencies
└── xgb_sklearn_model.pkl   # Trained XGBClassifier model
Deployment

The API is currently deployed on Render at https://taichi-1.onrender.com. To deploy it again, use these settings:

Build command: pip install -r requirement.txt
Start command: uvicorn app:app --host 0.0.0.0 --port $PORT

Render's free tier sleeps when idle, so the first request afterwards can be slow.

Known limitations
CORS allows every origin (allow_origins=["*"]). Restrict it to the frontend's domain before production use.
Dependency versions are not pinned. The model was saved with an older XGBoost release, so the latest version may show warnings or fail to load it. Pin versions in requirement.txt.
The model is reloaded on every request. Load it once at server startup instead.
The temporary file is always saved as .mp4, even when the frontend sends WebM. Whether it can be read depends on the codecs available to OpenCV on the server.
The training data and training code are not in this repository.
License

No license has been specified.
