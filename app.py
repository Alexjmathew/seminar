from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import json
import os
import threading
import logging
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
import speech_recognition as sr
import pyttsx3
from cryptography.fernet import Fernet
import requests
import chromadb
from chromadb.utils import embedding_functions
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from typing import List, Dict, Any

csrf = CSRFProtect(app)

app = Flask(__name__)
app.secret_key = os.getenv('SESSION_SECRET_KEY', 'your_secret_key')  # Use environment variable
app.permanent_session_lifetime = timedelta(minutes=30)  # Session timeout
bcrypt = Bcrypt(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Firebase safely
try:
    if not firebase_admin.get_app():
        cred = credentials.Certificate('firebase.json')
        firebase_admin.initialize_app(cred)
except ValueError as e:
    logger.error(f"Firebase initialization error: {e}")
db = firestore.client()

# Load encryption key
ENCRYPTION_KEY_FILE = 'encryption_key.key'
if os.path.exists(ENCRYPTION_KEY_FILE):
    with open(ENCRYPTION_KEY_FILE, 'rb') as f:
        key = f.read()
else:
    key = Fernet.generate_key()
    with open(ENCRYPTION_KEY_FILE, 'wb') as f:
        f.write(key)
cipher = Fernet(key)

# MediaPipe setup
mp_pose = mp.solutions.pose
pose = mp.Pose()
mp_drawing = mp.solutions.drawing_utils

# Voice command setup
recognizer = sr.Recognizer()
microphone = sr.Microphone()
engine = pyttsx3.init()

# Load models
try:
    with open('models/fatigue_model_v1.pkl', 'rb') as f:
        fatigue_model = pickle.load(f)
    with open('models/quality_model_v1.pkl', 'rb') as f:
        quality_model = pickle.load(f)
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    raise

# RAG Chatbot setup
chroma_client = chromadb.Client()
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
try:
    collection = chroma_client.get_or_create_collection(name="exercise_science", embedding_function=embedding_fn)
    if collection.count() == 0:
        collection.add(
            documents=[
                "Proper posture for squats: Keep knees behind toes, maintain a straight back.",
                "Knee raises strengthen core muscles and improve balance."
            ],
            ids=["1", "2"]
        )
except Exception as e:
    logger.error(f"RAG setup error: {e}")

# Thread-safe session state
class SessionState:
    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0
        self.target_count = 0
        self.position = None
        self.exercise_started = False
        self.feedback = "Begin Posture Training!"
        self.start_time = None
        self.last_rep_time = None
        self.exercise = None
        self.fatigue_level = 0.0
        self.streak = 0
        self.progress_level = "Beginner"
        self.rep_quality = 0.0
        self.rep_speeds = []
        self.postures = []

state = SessionState()

# Exercise definitions
EXERCISES = {
    "posture_training": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 45,
        "threshold": 10,
        "optimal_speed_range": (1.0, 2.5),
        "injury_angle": 120
    },
    "squats": {
        "joints": ["LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"],
        "target_angle": 90,
        "threshold": 15,
        "optimal_speed_range": (2.0, 4.0),
        "injury_angle": 150
    }
}

def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle between three points."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_tremor(landmarks: List, joint: str) -> float:
    """Calculate tremor based on landmark variance."""
    try:
        joint_idx = getattr(mp_pose.PoseLandmark, joint).value
        positions = [landmarks[joint_idx].x, landmarks[joint_idx].y]
        return np.var(positions) if len(positions) > 1 else 0.0
    except Exception as e:
        logger.error(f"Tremor calculation error: {e}")
        return 0.0

def speak(text: str) -> None:
    """Provide audio feedback."""
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        logger.error(f"Speech error: {e}")

def process_voice_command() -> None:
    """Process voice commands for exercise control."""
    with state.lock:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio).lower()
            if "start" in command:
                state.exercise_started = True
                state.feedback = "Exercise started!"
                speak(state.feedback)
            elif "stop" in command:
                state.exercise_started = False
                state.feedback = "Exercise stopped."
                speak(state.feedback)
            elif "set target" in command:
                try:
                    target = int(command.split("set target")[-1])
                    state.target_count = target
                    state.count = 0
                    state.feedback = f"Target set to {target} reps."
                    speak(state.feedback)
                except ValueError:
                    state.feedback = "Invalid target. Please specify a number."
                    speak(state.feedback)
        except sr.UnknownValueError:
            state.feedback = "Could not understand audio."
            speak(state.feedback)
        except sr.RequestError:
            state.feedback = "Voice recognition service unavailable."
            speak(state.feedback)
        except Exception as e:
            logger.error(f"Voice command error: {e}")

def voice_listener() -> None:
    """Run voice command listener in a separate thread."""
    while True:
        if state.exercise_started:
            process_voice_command()
        time.sleep(1)

threading.Thread(target=voice_listener, daemon=True).start()

def predict_fatigue(rep_speed: float, angle_deviation: float, tremor: float, count: int) -> float:
    """Predict fatigue level using hybrid LSTM + RF model."""
    try:
        X = np.array([[rep_speed, angle_deviation, tremor, count]])
        X_lstm = X.reshape(X.shape[0], X.shape[1], 1)
        lstm_pred = fatigue_model['lstm'].predict(X_lstm, verbose=0)[0][0]
        rf_pred = fatigue_model['rf'].predict_proba(X)[:, 1][0]
        return (lstm_pred + rf_pred) / 2
    except Exception as e:
        logger.error(f"Fatigue prediction error: {e}")
        return 0.0

def predict_quality(angle: float, speed_consistency: float, range_of_motion: float) -> float:
    """Predict repetition quality using RF model."""
    try:
        X = np.array([[angle, speed_consistency, range_of_motion]])
        return quality_model.predict_proba(X)[:, 1][0]
    except Exception as e:
        logger.error(f"Quality prediction error: {e}")
        return 0.0

def update_models(fatigue_data: np.ndarray, quality_data: np.ndarray) -> None:
    """Update fatigue and quality models with new data."""
    try:
        # Load existing data (simulated for now)
        X_fatigue = np.random.rand(100, 4) * [3, 30, 0.1, 20]
        y_fatigue = (X_fatigue[:, 0] + X_fatigue[:, 1] / 30 + X_fatigue[:, 2] * 10) / 4
        X_quality = np.random.rand(100, 3) * [180, 2, 50]
        y_quality = (X_quality[:, 0] / 180 + X_quality[:, 1] / 2 + X_quality[:, 2] / 50) / 3

        # Append new data
        X_fatigue = np.vstack([X_fatigue, fatigue_data[:, :-1]])
        y_fatigue = np.append(y_fatigue, fatigue_data[:, -1])
        X_quality = np.vstack([X_quality, quality_data[:, :-1]])
        y_quality = np.append(y_quality, quality_data[:, -1])

        # Retrain fatigue model
        lstm = Sequential([
            LSTM(50, input_shape=(X_fatigue.shape[1], 1), return_sequences=False),
            Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer='adam', loss='mse')
        X_lstm = X_fatigue.reshape(X_fatigue.shape[0], X_fatigue.shape[1], 1)
        lstm.fit(X_lstm, y_fatigue, epochs=10, verbose=0)
        rf = RandomForestRegressor()
        rf.fit(X_fatigue, y_fatigue > 0.5)
        fatigue_model_new = {'lstm': lstm, 'rf': rf}

        # Retrain quality model
        rf_quality = RandomForestRegressor()
        rf_quality.fit(X_quality, y_quality > 0.5)

        # Save new models with version
        version = len([f for f in os.listdir('models') if 'fatigue_model' in f]) + 1
        with open(f'models/fatigue_model_v{version}.pkl', 'wb') as f:
            pickle.dump(fatigue_model_new, f)
        with open(f'models/quality_model_v{version}.pkl', 'wb') as f:
            pickle.dump(rf_quality, f)

        logger.info(f"Models updated to version {version}")
    except Exception as e:
        logger.error(f"Model update error: {e}")

def sync_google_fit(email: str, session_data: Dict[str, Any]) -> bool:
    """Sync session data with Google Fit (placeholder)."""
    try:
        # TODO: Implement OAuth for Google Fit
        logger.warning("Google Fit sync not implemented. Replace with valid OAuth token.")
        return False
    except Exception as e:
        logger.error(f"Google Fit sync failed: {e}")
        return False

def log_chat_message(sender: str, receiver: str, message: str) -> None:
    """Log encrypted chat messages for HIPAA compliance."""
    try:
        log_entry = {
            "sender": sender,
            "receiver": receiver,
            "message": cipher.encrypt(message.encode()).decode(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        os.makedirs('audit_logs', exist_ok=True)
        log_file = f'audit_logs/{sender}_{receiver}_{int(time.time())}.json'
        with open(log_file, 'w') as f:
            json.dump(log_entry, f)
        logger.info(f"Chat message logged: {log_file}")
    except Exception as e:
        logger.error(f"Chat logging error: {e}")

def generate_frames():
    """Generate video frames with posture analysis."""
    cap = cv2.VideoCapture(0)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera frame capture failed")
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            with state.lock:
                if results.pose_landmarks and state.exercise_started and state.exercise:
                    landmarks = results.pose_landmarks.landmark
                    joints = state.exercise["joints"]
                    coords = [
                        [landmarks[getattr(mp_pose.PoseLandmark, joint).value].x,
                         landmarks[getattr(mp_pose.PoseLandmark, joint).value].y]
                        for joint in joints
                    ]
                    angle = calculate_angle(*[np.array(c) for c in coords])
                    state.postures.append(angle)

                    # Injury prevention
                    if angle > state.exercise["injury_angle"]:
                        state.feedback = "Warning: Excessive joint angle! Risk of injury."
                        speak(state.feedback)

                    # Posture training logic
                    if angle > state.exercise["target_angle"] + state.exercise["threshold"]:
                        state.position = "up"
                    if state.position == "up" and angle < state.exercise["target_angle"] - state.exercise["threshold"]:
                        state.position = "down"
                        state.count += 1
                        current_time = time.time()
                        if state.last_rep_time:
                            rep_time = current_time - state.last_rep_time
                            state.rep_speeds.append(rep_time)
                            angle_deviation = abs(angle - state.exercise["target_angle"])
                            tremor = calculate_tremor(landmarks, joints[1])
                            range_of_motion = max(state.postures) - min(state.postures) if state.postures else 0
                            speed_consistency = np.std(state.rep_speeds) if state.rep_speeds else 0
                            state.fatigue_level = predict_fatigue(rep_time, angle_deviation, tremor, state.count)
                            state.rep_quality = predict_quality(angle, speed_consistency, range_of_motion)

                            if state.exercise["optimal_speed_range"][0] <= rep_time <= state.exercise["optimal_speed_range"][1]:
                                state.feedback = f"Good speed! Rep quality: {state.rep_quality:.2f}"
                            elif rep_time < state.exercise["optimal_speed_range"][0]:
                                state.feedback = f"Too fast! Rep quality: {state.rep_quality:.2f}"
                            else:
                                state.feedback = f"Too slow! Rep quality: {state.rep_quality:.2f}"
                            if state.fatigue_level > 0.7:
                                state.feedback += " Fatigue detected! Consider resting."
                                speak(state.feedback)
                        state.last_rep_time = current_time
                        if state.count == 1:
                            state.start_time = current_time

                    # Feedback
                    if angle < state.exercise["target_angle"] - state.exercise["threshold"]:
                        state.feedback = f"Lower your {'knee' if state.exercise['joints'][1] == 'LEFT_KNEE' else 'body'} slightly."
                    elif angle > state.exercise["target_angle"] + state.exercise["threshold"]:
                        state.feedback = f"Raise your {'knee' if state.exercise['joints'][1] == 'LEFT_KNEE' else 'body'} higher!"

                    # Draw feedback
                    cv2.putText(image, f'Angle: {int(angle)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(image, f'Count: {state.count}/{state.target_count}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(image, state.feedback, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(image, f'Fatigue: {state.fatigue_level:.2f}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.putText(image, f'Rep Quality: {state.rep_quality:.2f}', (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # Stop exercise
                    if state.count >= state.target_count:
                        state.exercise_started = False
                        total_time = time.time() - state.start_time if state.start_time else 0
                        state.feedback = f"Posture Training Complete! Total time: {total_time:.2f}s"
                        speak(state.feedback)
                        state.streak += 1
                        if 'email' in session:
                            user_ref = db.collection('users').document(session['email'])
                            user_data = user_ref.get().to_dict()
                            total_reps = state.count + sum([s['count'] for s in user_data.get('sessions', [])])
                            state.progress_level = (
                                "Expert" if total_reps > 1000 else
                                "Advanced" if total_reps > 500 else
                                "Intermediate" if total_reps > 100 else
                                "Beginner"
                            )
                            session_data = {
                                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "count": state.count,
                                "total_time": total_time,
                                "average_speed": total_time / state.count if state.count > 0 else 0,
                                "streak": state.streak,
                                "progress_level": state.progress_level,
                                "encrypted_data": cipher.encrypt(str({
                                    "count": state.count,
                                    "total_time": total_time
                                }).encode()).decode(),
                                "fatigue_level": float(state.fatigue_level),
                                "rep_quality": float(state.rep_quality)
                            }
                            user_ref.update({"sessions": firestore.ArrayUnion([session_data])})
                            new_data_fatigue = np.array([[
                                state.rep_speeds[-1] if state.rep_speeds else 0,
                                abs(state.postures[-1] - state.exercise["target_angle"]) if state.postures else 0,
                                tremor,
                                state.count,
                                state.fatigue_level
                            ]])
                            new_data_quality = np.array([[
                                state.postures[-1] if state.postures else 0,
                                np.std(state.rep_speeds) if state.rep_speeds else 0,
                                max(state.postures) - min(state.postures) if state.postures else 0,
                                state.rep_quality
                            ]])
                            threading.Thread(target=update_models, args=(new_data_fatigue, new_data_quality)).start()
                            sync_google_fit(session['email'], session_data)
                        state.start_time = None
                        state.last_rep_time = None
                        state.count = 0
                        state.rep_speeds = []
                        state.postures = []

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logger.error(f"Frame generation error: {e}")
    finally:
        cap.release()

@app.route('/')
def index():
    """Redirect to appropriate dashboard based on role."""
    if 'email' in session:
        try:
            user_ref = db.collection('users').document(session['email'])
            user = user_ref.get()
            if user.exists:
                role = user.to_dict().get('role', 'Patient')
                return redirect(url_for('admin' if role == 'Admin' else 'doctor' if role == 'Doctor' else 'profile'))
        except Exception as e:
            logger.error(f"Index route error: {e}")
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login with password hashing."""
    if request.method == 'POST':
        try:
            email = request.form['email']
            password = request.form['password']
            users_ref = db.collection('users').where('email', '==', email).limit(1)
            users = list(users_ref.stream())
            if users and bcrypt.check_password_hash(users[0].to_dict()['password'], password):
                session['email'] = email
                session['username'] = users[0].to_dict()['username']
                session['role'] = users[0].to_dict().get('role', 'Patient')
                logger.info(f"User logged in: {email}")
                return redirect(url_for('index'))
            else:
                return render_template('login.html', error="Invalid email or password.")
        except Exception as e:
            logger.error(f"Login error: {e}")
            return render_template('login.html', error="Server error. Please try again.")
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Handle user logout."""
    try:
        session.clear()
        logger.info("User logged out")
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration with password hashing."""
    if request.method == 'POST':
        try:
            username = request.form['username']
            email = request.form['email']
            password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
            age = request.form['age']
            height = request.form['height']
            weight = request.form['weight']
            blood_group = request.form['blood_group']
            role = request.form.get('role', 'Patient') if session.get('role') == 'Admin' else 'Patient'
            email_query = db.collection('users').where('email', '==', email).limit(1)
            if list(email_query.stream()):
                return render_template('register.html', error="Email already registered.")
            user_ref = db.collection('users').document(email)
            user_data = {
                "username": username,
                "email": email,
                "password": password,
                "age": int(age),
                "height": float(height),
                "weight": float(weight),
                "blood_group": blood_group,
                "sessions": [],
                "streak": 0,
                "progress_level": "Beginner",
                "role": role,
                "assigned_doctor": None
            }
            user_ref.set(user_data)
            session['email'] = email
            session['username'] = username
            session['role'] = role
            logger.info(f"User registered: {email}")
            return redirect(url_for('index'))
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return render_template('register.html', error="Registration failed. Please try again.")
    return render_template('register.html')

@app.route('/profile')
def profile():
    """Display patient profile with session statistics."""
    if 'email' not in session or session['role'] != 'Patient':
        return redirect(url_for('login'))
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        user_data = user.to_dict()
        sessions = user_data.get('sessions', [])
        session_dates = [s['date'] for s in sessions]
        session_counts = [s['count'] for s in sessions]
        session_total_times = [s['total_time'] for s in sessions]
        session_average_speeds = [s['average_speed'] for s in sessions]
        session_fatigue_levels = [s['fatigue_level'] for s in sessions]
        session_rep_qualities = [s['rep_quality'] for s in sessions]
        return render_template('profile.html',
                              user=user_data,
                              session_dates=session_dates,
                              session_counts=session_counts,
                              session_total_times=session_total_times,
                              session_average_speeds=session_average_speeds,
                              session_fatigue_levels=session_fatigue_levels,
                              session_rep_qualities=session_rep_qualities)
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return redirect(url_for('login'))

@app.route('/admin')
def admin():
    """Display admin dashboard."""
    if 'email' not in session or session['role'] != 'Admin':
        return redirect(url_for('login'))
    try:
        users = [u.to_dict() for u in db.collection('users').stream()]
        return render_template('admin.html', users=users)
    except Exception as e:
        logger.error(f"Admin dashboard error: {e}")
        return redirect(url_for('login'))

@app.route('/doctor')
def doctor():
    """Display doctor dashboard with patient data."""
    if 'email' not in session or session['role'] != 'Doctor':
        return redirect(url_for('login'))
    try:
        patients = [p.to_dict() for p in db.collection('users').where('assigned_doctor', '==', session['email']).stream()]
        return render_template('doctor.html', patients=patients)
    except Exception as e:
        logger.error(f"Doctor dashboard error: {e}")
        return redirect(url_for('login'))

@app.route('/assign_doctor', methods=['POST'])
def assign_doctor():
    """Assign a doctor to a patient."""
    if 'email' not in session or session['role'] != 'Admin':
        return jsonify({'success': False, 'error': 'Unauthorized'})
    try:
        patient_email = request.form['patient_email']
        doctor_email = request.form['doctor_email']
        patient_ref = db.collection('users').document(patient_email)
        if patient_ref.get().exists:
            patient_ref.update({'assigned_doctor': doctor_email})
            logger.info(f"Assigned doctor {doctor_email} to patient {patient_email}")
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Patient not found'})
    except Exception as e:
        logger.error(f"Assign doctor error: {e}")
        return jsonify({'success': False, 'error': 'Server error'})

@app.route('/prescribe_exercise', methods=['POST'])
def prescribe_exercise():
    """Prescribe an exercise to a patient."""
    if 'email' not in session or session['role'] != 'Doctor':
        return jsonify({'success': False, 'error': 'Unauthorized'})
    try:
        patient_email = request.form['patient_email']
        exercise_name = request.form['exercise']
        if exercise_name not in EXERCISES:
            return jsonify({'success': False, 'error': 'Invalid exercise'})
        patient_ref = db.collection('users').document(patient_email)
        if patient_ref.get().exists:
            patient_ref.update({'prescribed_exercise': exercise_name})
            logger.info(f"Prescribed {exercise_name} to patient {patient_email}")
            return jsonify({'success': True})
        return jsonify({'success': False, 'error': 'Patient not found'})
    except Exception as e:
        logger.error(f"Prescribe exercise error: {e}")
        return jsonify({'success': False, 'error': 'Server error'})


@app.route('/recommendation')
def recommendation():
    if 'email' not in session or session['role'] != 'Patient':
        return redirect(url_for('login'))
    try:
        user_ref = db.collection('users').document(session['email'])
        user = user_ref.get()
        if not user.exists:
            return redirect(url_for('login'))
        prescribed_exercise = user.to_dict().get('prescribed_exercise')
        return render_template('recommendation.html', prescribed_exercise=prescribed_exercise)
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        return redirect(url_for('profile'))

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """Handle HIPAA-compliant chat with RAG chatbot."""
    if 'email' not in session:
        return redirect(url_for('login'))
    try:
        if request.method == 'POST':
            receiver = request.form['receiver']
            message = request.form['message']
            log_chat_message(session['email'], receiver, message)
            query = collection.query(query_texts=[message], n_results=1)
            response = query['documents'][0][0] if query['documents'] else "Please consult your doctor."
            log_chat_message('RAG_Chatbot', session['email'], response)
            return jsonify({'response': response})
        users = [u.to_dict() for u in db.collection('users').stream() if u.to_dict()['email'] != session['email']]
        return render_template('chat.html', users=users)
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return redirect(url_for('profile'))

@app.route('/select_exercise', methods=['GET', 'POST'])
def select_exercise():
    """Allow patients to select an exercise."""
    if 'email' not in session or session['role'] != 'Patient':
        return redirect(url_for('login'))
    try:
        with state.lock:
            if request.method == 'POST':
                exercise_name = request.form['exercise']
                if exercise_name in EXERCISES:
                    state.exercise = EXERCISES[exercise_name]
                    return redirect(url_for('training'))
                return render_template('select_exercise.html', error="Invalid exercise")
            user_ref = db.collection('users').document(session['email'])
            prescribed_exercise = user_ref.get().to_dict().get('prescribed_exercise')
            return render_template('select_exercise.html', prescribed_exercise=prescribed_exercise)
    except Exception as e:
        logger.error(f"Select exercise error: {e}")
        return redirect(url_for('profile'))

@app.route('/training')
def training():
    """Render training page for patients."""
    if 'email' not in session or session['role'] != 'Patient':
        return redirect(url_for('login'))
    try:
        return render_template('training.html')
    except Exception as e:
        logger.error(f"Training page error: {e}")
        return redirect(url_for('profile'))

@app.route('/video_feed')
def video_feed():
    """Stream video feed with posture analysis."""
    try:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Video feed error: {e}")
        return jsonify({'error': 'Video feed unavailable'}), 500

@app.route('/get_count')
def get_count():
    """Return current exercise state."""
    try:
        with state.lock:
            return jsonify({
                'count': state.count,
                'target': state.target_count,
                'feedback': state.feedback,
                'fatigue': state.fatigue_level,
                'quality': state.rep_quality
            })
    except Exception as e:
        logger.error(f"Get count error: {e}")
        return jsonify({'error': 'Server error'}), 500

@app.route('/set_target', methods=['POST'])
def set_target():
    """Set target repetitions for exercise."""
    try:
        with state.lock:
            data = request.json
            state.target_count = int(data.get('target', 0))
            state.count = 0
            state.exercise_started = True
            state.feedback = "Begin Posture Training!"
            state.start_time = None
            state.last_rep_time = None
            state.rep_speeds = []
            state.postures = []
            return jsonify({'success': True, 'target': state.target_count})
    except Exception as e:
        logger.error(f"Set target error: {e}")
        return jsonify({'success': False, 'error': 'Invalid target'}), 400

@app.route('/leaderboard')
def leaderboard():
    """Display leaderboard of patient progress."""
    if 'email' not in session:
        return redirect(url_for('login'))
    try:
        users = [u.to_dict() for u in db.collection('users').where('role', '==', 'Patient').stream()]
        leaderboard = [
            {'username': u['username'], 'total_reps': sum(s['count'] for s in u.get('sessions', []))}
            for u in users
        ]
        leaderboard.sort(key=lambda x: x['total_reps'], reverse=True)
        return render_template('leaderboard.html', leaderboard=leaderboard)
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        return redirect(url_for('profile'))

@app.route('/delete_account')
def delete_profile():
    """Delete user account."""
    if 'email' not in session:
        return redirect(url_for('login'))
    try:
        user_ref = db.collection('users').document(session['email'])
        user_ref.delete()
        session.clear()
        logger.info(f"Account deleted: {session.get('email')}")
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Delete account error: {e}")
        return redirect(url_for('profile'))

if __name__ == "__main__":
    app.run(debug=True)
