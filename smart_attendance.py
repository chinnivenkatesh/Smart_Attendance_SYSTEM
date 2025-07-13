import cv2
import face_recognition
import os
import numpy as np

def load_known_faces(folder_path='known_faces'):
    known_encodings = []
    known_names = []

    for person_name in os.listdir(folder_path):
        person_folder = os.path.join(folder_path, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)

            img_bgr = cv2.imread(img_path)

            if img_bgr is None:
                print(f"[ERROR] Failed to load: {img_path}")
                continue

            # Convert to grayscale
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Convert back to RGB from grayscale
            img_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB)

            # Safety checks
            if img_rgb.dtype != np.uint8 or img_rgb.shape[2] != 3:
                print(f"[ERROR] Skipping invalid image: {img_path}")
                continue

            try:
                encodings = face_recognition.face_encodings(img_rgb)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(person_name)
                    print(f"[OK] Encoded: {img_path}")
                else:
                    print(f"[WARN] No face detected in: {img_path}")
            except Exception as e:
                print(f"[ERROR] Error encoding {img_path}: {e}")

    return known_encodings, known_names

def recognize_faces_with_keypress(known_face_encodings, known_face_names):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Press 'q' in the video window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)

            # Draw labels and boxes
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 1)

            cv2.imshow("Face Recognition (press 'q' to quit)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Webcam released.")

# Load faces and run
known_face_encodings, known_face_names = load_known_faces()

if known_face_encodings:
    recognize_faces_with_keypress(known_face_encodings, known_face_names)
else:
    print("[ERROR] No valid face encodings found. Please check your images.")
