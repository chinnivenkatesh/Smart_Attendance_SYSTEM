# Smart_Attendance_SYSTEM
A real-time attendance system built with Python, Flask, OpenCV, and face_recognition. This project captures faces through a webcam, compares them with pre-registered images, and automatically marks attendance. It features duplicate detection, dynamic user registration, and stores attendance logs with date and time in a database.

## 📁 Project Folder Structure

Here’s a breakdown of how the Smart Attendance project is organized:

📦 smart-attendance/
├── 📄 app.py                      → Flask backend handling routes
├── 📄 smart_attendance.py         → Core face recognition logic
├── 📄 face_recognition.ipynb      → (Optional) Jupyter Notebook version
├── 📁 frontend/                   → Web interface (HTML/CSS/JS)
│   └── index.html                 → Example landing page
├── 📁 known_faces/                → Folder to store registered face images
│   └── john_doe.jpg               → Example image (use your name)
├── 📁 attendance_logs/            → Folder to store attendance records
│   └── sample_log.json            → Example log with name, date, time
├── 📄 requirements.txt            → Python dependencies
├── 📄 .gitignore                  → Ignore unnecessary/large files
├── 📄 README.md                   → Project documentation
├── 📄 cmake-4.0.2-windows-x86_64.exe   → (Optional) Installer if needed
├── 📄 dlib-19.22.99-cp310-cp310-win_amd64.whl → (Optional) dlib installer


⚠️ Only include small, sample files in `known_faces/` and `attendance_logs/` to keep your repository lightweight. Use `.gitignore` to exclude the rest.


> ⚠️ Only include sample files in `known_faces/` and `attendance_logs/`. Use `.gitignore` to exclude large/unnecessary files.

---

## ⚙️ How It Works

1. **Face Registration**
   - The user clicks "Register" on the frontend.
   - A webcam image is captured and saved in the `known_faces/` folder.
   - The image is named based on the user’s input (e.g., `john_doe.jpg`).

2. **Attendance Capture**
   - Clicking "Give Attendance" triggers face scanning via webcam.
   - The live face is encoded and compared with the saved face images.
   - If a match is found:
     - Attendance is marked.
     - Timestamp is recorded in `attendance_logs/`.

3. **Logs**
   - Each recognition result is saved in `.json` with the format:
     ```json
     {
       "name": "John Doe",
       "time": "10:15:03",
       "date": "2025-07-13"
     }
     ```

---

## 🧰 Installation

> ⚠️ Requires Python 3.10+ and dlib-compatible system

### 1. Clone the Repo
```bash
git clone https://github.com/your-username/smart-attendance.git
cd smart-attendance
