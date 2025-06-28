# recognize_student.py
from deepface import DeepFace
import cv2
import os

def recognize_from_source(source="webcam", database_path="registered_students", media_path=None):
    if source == "webcam":
        cap = cv2.VideoCapture(0)
    elif source == "video":
        cap = cv2.VideoCapture(media_path)
    elif source == "image":
        print("[INFO] Running recognition on image...")
        results = DeepFace.find(img_path=media_path, db_path=database_path, enforce_detection=False)
        print("[INFO] Recognition Results:\n", results[0][["identity", "distance"]])
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            results = DeepFace.find(img_path=frame, db_path=database_path, enforce_detection=False)
            df = results[0]
            for _, row in df.iterrows():
                identity = os.path.basename(row["identity"]).split(".")[0]
                print(f"[MATCHED] {identity} (distance: {round(row['distance'], 3)})")
        except Exception as e:
            print("[WARN] Detection failed:", e)

        cv2.imshow("Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_from_source(source="webcam")
