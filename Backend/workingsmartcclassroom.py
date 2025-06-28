import cv2
import os
import pandas as pd
from datetime import datetime
from deepface import DeepFace
from ultralytics import YOLO
import numpy as np
import time
from collections import defaultdict
from engagementmultiple import process_frame_multi



SAVE_PATH = "registered_students"
os.makedirs(SAVE_PATH, exist_ok=True)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')



model = YOLO("yolov8n.pt")

def get_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def get_distance(c1, c2):
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def register_student(name, max_photos=5):
    cam = cv2.VideoCapture(0)
    count = 0
    print(f"[INFO] Registering {name}. Press 's' to save photo, 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_crop = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                img_path = os.path.join(SAVE_PATH, f"{name}_{count}.jpg")
                cv2.imwrite(img_path, face_crop)
                print(f"[INFO] Saved image {img_path}")
                count += 1
                if count >= max_photos:
                    cam.release()
                    cv2.destroyAllWindows()
                    return

        cv2.imshow(f"Registering: {name}", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def recognize_and_track():
    cap = cv2.VideoCapture(0)
    print("[INFO] Running Smart Classroom System. Press 'q' to quit.")

    present_students = set()
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M")
    session_log = []
    student_engagement_data = defaultdict(list)
    last_engagement_log_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_results = face_cascade.detectMultiScale(gray, 1.3, 5)

        results = model(frame, verbose=False)[0]
        phone_results = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls_id] if hasattr(model, 'names') else str(cls_id)

            print(f"[DEBUG] Detected: {label} (ID: {cls_id}), Confidence: {conf:.2f}")

            if cls_id == 67 and conf > 0.3:  # 67 = cell phone
                phone_results.append((x1, y1, x2, y2))

        student_data = []

        for (x, y, w, h) in face_results:
            face_crop = frame[y:y + h, x:x + w]
            label = "Unknown"

            try:
                results = DeepFace.find(img_path=face_crop, db_path=SAVE_PATH, enforce_detection=False)
                df = results[0]
                if not df.empty:
                    # identity_full = os.path.basename(df.iloc[0]["identity"]).split(".")[0]  # e.g. "asim_0"
                    # identity = identity_full.split('_')[0]  # e.g. "asim"
                    try:
                        results = DeepFace.find(img_path=face_crop, db_path=SAVE_PATH, enforce_detection=False)
                        df = results[0]
                        if not df.empty:
                            identity_full = os.path.basename(df.iloc[0]["identity"]).split(".")[0]  # asim_0
                            identity = identity_full.split('_')[0]  # asim
                            label = identity  # ✅ assign label properly
                            if identity not in present_students:
                                present_students.add(identity)
                                now = datetime.now().strftime("%H:%M:%S")
                                print(f"[ATTENDANCE] Marked present: {identity} at {now}")
                    except:
                        label = "Unknown"

                    if identity not in present_students:
                        present_students.add(identity)
                        now = datetime.now().strftime("%H:%M:%S")
                        print(f"[ATTENDANCE] Marked present: {identity} at {now}")
                    # identity = os.path.basename(df.iloc[0]["identity"]).split(".")[0]
                    # label = identity
                    # if identity not in present_students:
                    #     present_students.add(identity)
                    #     now = datetime.now().strftime("%H:%M:%S")
                    #     print(f"[ATTENDANCE] Marked present: {identity} at {now}")
            except:
                label = "Unknown"

            student_data.append({
                "name": label,
                "box": (x, y, x + w, y + h),
                "phone": False,
                "engagement": 0
            })

        for (x1, y1, x2, y2) in phone_results:
            phone_center = get_center((x1, y1, x2, y2))

            min_dist = float('inf')
            matched_student = None

            for student in student_data:
                student_center = get_center(student["box"])
                dist = get_distance(phone_center, student_center)
                if dist < 150 and dist < min_dist:
                    min_dist = dist
                    matched_student = student

            if matched_student:
                matched_student["phone"] = True

        # Engagement detection
        engagement_frame = process_frame_multi(frame.copy())
        gray = cv2.cvtColor(engagement_frame, cv2.COLOR_BGR2GRAY)
        engagement_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for idx, (x, y, w, h) in enumerate(engagement_faces):
            face_center = get_center((x, y, x + w, y + h))
            for student in student_data:
                student_center = get_center(student["box"])
                if get_distance(face_center, student_center) < 50:
                    try:
                        roi = engagement_frame[y-30:y, x:x+w]
                        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        score = np.mean(roi_gray)  # Placeholder for engagement
                        student["engagement"] = int(score)
                    except:
                        pass

        # Display results
        for student in student_data:
            x1, y1, x2, y2 = student["box"]
            label = student["name"]
            phone = student["phone"]
            engagement = student["engagement"]

            color = (0, 255, 0) if phone else (255, 255, 255)
            status = "Phone: YES" if phone else "Phone: NO"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, status, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"Engagement: {engagement}%", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            session_log.append({
                "Name": label,
                "Time": datetime.now().strftime("%H:%M:%S"),
                "Phone_Used": "Yes" if phone else "No",
                "Engagement": engagement
            })

            if label != "Unknown":
                student_engagement_data[label].append(engagement)

        cv2.rectangle(frame, (10, 10), (400, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Total Present: {len(present_students)}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Smart Classroom System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Engagement logging every 60s
        if time.time() - last_engagement_log_time > 60:
            for name, scores in student_engagement_data.items():
                if scores:
                    avg_score = sum(scores) / len(scores)
                    print(f"[ENGAGEMENT] {name}: Avg score = {avg_score:.1f}%")
                    student_engagement_data[name] = []
            last_engagement_log_time = time.time()

    cap.release()
    cv2.destroyAllWindows()

    if session_log:
        df = pd.DataFrame(session_log)
        file_name = f"classroom_log_{timestamp_str}.csv"
        df.to_csv(file_name, index=False)
        print(f"[INFO] Session log saved to {file_name}")
    else:
        print("[INFO] No session data to record.")

def main():
    while True:
        print("\n📘 Smart Classroom System")
        print("1. Register New Student")
        print("2. Start Smart Classroom (Attendance + Mobile + Identity)")
        print("3. Exit")
        choice = input("Enter your choice (1/2/3): ").strip()

        if choice == '1':
            name = input("Enter student name: ").strip().replace(" ", "_")
            register_student(name)
        elif choice == '2':
            recognize_and_track()
        elif choice == '3':
            print("Exiting system.")
            break
        else:
            print("Invalid input. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
