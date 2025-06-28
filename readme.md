# Smart Classroom Software

Smart Classroom Software is an AI-powered classroom management system that enables automated student attendance, mobile phone usage detection, and real-time engagement analysis using computer vision and machine learning.

## Features

- Face recognition-based student attendance using DeepFace
- Real-time mobile phone detection using YOLOv8
- Engagement scoring using eye, mouth, gaze, and posture analysis with MediaPipe
- Streamlit-based frontend dashboard for class scheduling, attendance, and performance management
- Optional machine learning-based engagement classifier (Random Forest)
- Session analytics with data export and visual engagement charts

## Technology Stack

| Feature               | Technology Used              |
|----------------------|------------------------------|
| Face Recognition     | DeepFace, OpenCV             |
| Mobile Detection     | YOLOv8 (Ultralytics)         |
| Engagement Analysis  | MediaPipe, NumPy, Pandas     |
| Backend API          | FastAPI                      |
| Frontend Dashboard   | Streamlit                    |
| ML Model             | scikit-learn (RandomForest)  |
| Data Export/Analysis | Matplotlib, Pandas, Excel    |
| Storage              | CSV, JSON                    |

## Directory Structure

smart-classroom-software/
├── backend/
│ ├── main.py
│ ├── register_student.py
│ ├── recognize_student.py
│ ├── smart_classroom_core.py
│ ├── workingsmartcclassroom.py
│ └── engagementmultiple.py
├── frontend/
│ └── finalfrontend2.py
├── data/
│ └── logs, session stats, exported Excel files
├── registered_students/
│ └── saved face images for recognition
├── models/
│ └── yolov8n.pt (model weights if needed)
├── requirements.txt
├── .gitignore
└── README.md


## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/smart-classroom-software.git
cd smart-classroom-software

pip install -r requirements.txt

2. Install the required dependencies:

pip install -r requirements.txt

3. Ensure you have:

i) A working webcam

ii) YOLOv8 weights (yolov8n.pt) in the models/ directory

iii) Haar Cascade file (usually pre-included in OpenCV installation)

Running the System

Start the Backend API (FastAPI)

uvicorn main:app --reload

Launch the Streamlit Dashboard

streamlit run finalfrontend2.py

Register a New Student (Face Images) 

python register_student.py

Run the Smart Classroom Session 

python workingsmartcclassroom.py

Data Logging

i) Attendance logs are saved as CSV in the root or data/ folder with a timestamp.

ii) Engagement session statistics and behavior logs are saved as .csv, .json, and .xlsx.

iii) Visual engagement charts are saved as PNG images.

Screenshots / Demo Videos

i) Mobile Detection Sample

ii) Dashboard Screenshot

Future Enhancements

i) Live database integration (Firebase or PostgreSQL)

ii) Student-wise performance tracking over time

iii) Enhanced voice and tone-based emotion recognition

iv) Custom face recognition model (age-invariant embeddings)

v) Classroom-wide monitoring with dashboards for teachers and administrators

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

[Ultralytics YOLOv8] (https://github.com/ultralytics/ultralytics)

[DeepFace](https://github.com/serengil/deepface)

[MediaPipe] (https://github.com/google-ai-edge/mediapipe)

[Streamlit] (https://streamlit.io/)

[OpenCV] (https://opencv.org/)

[Pandas] (https://pandas.pydata.org/)

[Matplotlib] (https://matplotlib.org/)

[Scikit learn] (https://scikit-learn.org/stable/)

