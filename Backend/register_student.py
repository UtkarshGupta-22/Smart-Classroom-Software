# register_student.py
import cv2
import os

def register_student(name, save_path="registered_students"):
    os.makedirs(save_path, exist_ok=True)
    cam = cv2.VideoCapture(0)
    print("Press 's' to save photo, 'q' to quit.")

    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow(f"Register Face - {name}", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            img_path = os.path.join(save_path, f"{name}_{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Saved image {img_path}")
            count += 1
            if count >= 5:
                break
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    student_name = input("Enter student name: ").strip().replace(" ", "_")
    register_student(student_name)
