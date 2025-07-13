from flask import Flask, request, jsonify, send_file
import cv2
import face_recognition
import numpy as np
import os
import uuid
import base64
import json
from datetime import datetime
from flask_cors import CORS
import csv

app = Flask(__name__)
CORS(app)

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_LOG = "attendance_logs.json"
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Function to encode face
def encode_face(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    return encodings[0] if encodings else None

# Check if face already exists
def is_face_already_registered(new_encoding):
    for person in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person)
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            existing_encoding = encode_face(img)
            if existing_encoding is not None:
                result = face_recognition.compare_faces([existing_encoding], new_encoding, tolerance=0.45)
                if result[0]:
                    return True
    return False

# Register route
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    name = data.get("name")
    image_data = data.get("image")

    if not name or not image_data:
        return jsonify({"status": "error", "message": "Missing name or image"})

    try:
        image_base64 = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"status": "error", "message": "Failed to decode image"})

    except Exception as e:
        return jsonify({"status": "error", "message": f"Image decode error: {str(e)}"})

    encoding = encode_face(frame)
    if encoding is None:
        return jsonify({"status": "error", "message": "No face detected"})

    if is_face_already_registered(encoding):
        return jsonify({"status": "exists", "message": "Face already registered"})

    person_path = os.path.join(KNOWN_FACES_DIR, name)

    # ✅ FIX: Make the folder if it doesn't exist
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    filename = os.path.join(person_path, f"{uuid.uuid4().hex}.jpg")
    success = cv2.imwrite(filename, frame)

    if not success:
        return jsonify({"status": "error", "message": "Failed to save image"})

    return jsonify({"status": "success", "message": "Face registered successfully"})


# Download attendance as CSV
@app.route("/download-csv", methods=["GET"])
def download_csv():
    csv_file = "attendance_export.csv"

    if os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "r") as f:
            logs = json.load(f)

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["NAME", "DATE", "TIME"])
            writer.writeheader()
            writer.writerows([
                {"NAME": l["name"], "DATE": l["date"], "TIME": l["time"]}
                for l in logs
            ])

        return send_file(csv_file, as_attachment=True)

    return jsonify({"status": "error", "message": "No attendance logs found"})

# Filter logs by name or date
@app.route("/filter-attendance", methods=["GET"])
def filter_attendance():
    name_filter = request.args.get("name", "").strip().lower()
    date_filter = request.args.get("date", "").strip()

    if os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "r") as f:
            logs = json.load(f)

        filtered = []
        for log in logs:
            name_match = name_filter in log["name"].lower() if name_filter else True
            date_match = log["date"] == date_filter if date_filter else True
            if name_match and date_match:
                filtered.append(log)
        filtered.reverse()
        return jsonify({"status": "success", "data": filtered})
    else:
        return jsonify({"status": "error", "message": "No attendance logs found"})

# Mark attendance route
@app.route("/mark-attendance", methods=["POST"])
def mark_attendance():
    data = request.get_json()
    image_data = data.get("image")

    if not image_data:
        return jsonify({"status": "error", "message": "Image is missing"})

    try:
        image_base64 = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Image decode error: {str(e)}"})

    encoding = encode_face(frame)
    if encoding is None:
        return jsonify({"status": "error", "message": "No face detected"})

    matched_name = None
    for person in os.listdir(KNOWN_FACES_DIR):
        person_folder = os.path.join(KNOWN_FACES_DIR, person)
        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            known_img = cv2.imread(img_path)
            if known_img is None:
                continue
            known_encoding = encode_face(known_img)
            if known_encoding is not None:
                result = face_recognition.compare_faces([known_encoding], encoding, tolerance=0.45)
                distance = face_recognition.face_distance([known_encoding], encoding)[0]
                print(f"Comparing with {person}, distance = {distance:.4f}")  # 🧪 Debugging

                if result[0]:
                    matched_name = person
                    break
        if matched_name:
            break

    if not matched_name:
        return jsonify({"status": "error", "message": "Face not recognized"})

    now = datetime.now()
    date = now.strftime("%d/%m/%Y")
    time = now.strftime("%I:%M:%S %p")

    if os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "r") as f:
            logs = json.load(f)
    else:
        logs = []

    logs.append({
        "name": matched_name,
        "date": date,
        "time": time
    })

    with open(ATTENDANCE_LOG, "w") as f:
        json.dump(logs, f, indent=2)

    return jsonify({
        "status": "success",
        "message": f"Attendance marked for {matched_name}",
        "name": matched_name,
        "date": date,
        "time": time
    })

# Health check
@app.route("/", methods=["GET"])
def index():
    return "Server is running"

# Get latest 3 attendance entries
@app.route("/get-latest-attendance", methods=["GET"])
def get_latest_attendance():
    if os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "r") as f:
            logs = json.load(f)
            latest_logs = logs[-3:]
            latest_logs.reverse()
            return jsonify({"status": "success", "data": latest_logs})
    else:
        return jsonify({"status": "error", "message": "No attendance records found"})

# Run server
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True)
