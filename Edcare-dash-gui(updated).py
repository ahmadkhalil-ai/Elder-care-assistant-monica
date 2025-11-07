import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import base64
import cv2
import numpy as np
import threading
import time
import queue
import io
from PIL import Image
import logging


import speech_recognition as sr
import datetime
import webbrowser
import os
import platform
import subprocess
import asyncio
import tempfile
import pyjokes
import json
import random
import requests
from collections import deque
import mediapipe as mp

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Thread lock for thread-safe access to daily_log
_daily_log_lock = threading.Lock()

# Thread lock for speech to prevent overlap
speak_lock = threading.Lock()

# Global MediaPipe instances
_hands = None
_face_detection = None

# Configuration and global state
CACHE_DIR = tempfile.gettempdir()
VOICE = "en-US-AriaNeural"
RATE = "-5%"
WAKE_WORD = "monica"
MAX_ERRORS = 3
INACTIVITY_TIMEOUT = 60

# Email configuration
EMAIL_ENABLED = True
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "Your email"
EMAIL_PASSWORD = "Key password"
EMERGENCY_CONTACTS = ["email to recieve emergency", ""]

# Camera monitoring configuration
CAMERA_ENABLED = True
POSE_DETECTION_ENABLED = True
FALL_DETECTION_ENABLED = True
GESTURE_RECOGNITION_ENABLED = True

# Gesture recognition settings
GESTURE_CONFIDENCE_THRESHOLD = 0.5
GESTURE_HISTORY_LENGTH = 20
MIN_WAVING_FRAMES = 10
MIN_HAND_RAISE_FRAMES = 8

# Customizable settings for different room setups
ROOM_CONFIG = {
    "ceiling_height": 8.0,
    "person_height": 5.11,
    "average_standing_height": 5.9,
    "average_sitting_height": 3.0,
    "safe_zones": [
        {"x_min": 0.2, "x_max": 0.8, "y_min": 0.2, "y_max": 0.8}
    ]
}

# Elder care specific settings
MEDICINE_SCHEDULE = {}
TEMPERATURE_THRESHOLDS = {"cold": 65, "hot": 78}
HYDRATION_INTERVAL = 2 * 60 * 60
MOVEMENT_INTERVAL = 3 * 60 * 60
IDLE_CHECK_INTERVAL = 6 * 60 * 60

# Global state variables
_error_cnt = 0
_spoken_for_this_turn = False
_last_interaction_time = time.time()
_last_hydration_reminder = time.time()
_last_movement_reminder = time.time()
_emergency_mode = False
_current_posture = "unknown"
_posture_history = deque(maxlen=30)
_camera = None
_camera_window_open = False
_last_gesture_time = 0
_gesture_cooldown = 10
_last_fall_detection_time = 0
_fall_cooldown = 30
_is_speaking = False
_awaiting_response = False
reminders_enabled = True
_last_morning_date = None
_last_evening_date = None
_last_medicine_reminder = {}

# Gesture recognition variables
_gesture_history = deque(maxlen=GESTURE_HISTORY_LENGTH)
_hand_landmarks_history = deque(maxlen=GESTURE_HISTORY_LENGTH)
_wrist_y_history = deque(maxlen=MIN_WAVING_FRAMES)  # For waving detection

# Initialize daily_log
_daily_log = {
    "medicine_taken": {},
    "wellness_checks": [],
    "hydration_reminders": 0,
    "movement_encouragements": 0,
    "posture_alerts": 0,
    "fall_detections": 0,
    "gesture_recognitions": 0
}

# Global variables for the web dashboard
camera_frame_queue = queue.Queue(maxsize=10)
assistant_status = "Ready"
conversation_log = []
emergency_triggered = False

# ===== DASH APP SETUP =====
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Monica Elder Care Assistant"

# ===== CORE FUNCTION IMPLEMENTATIONS =====
def medicine_reminder(medicine_name=None):
    """Remind to take medicine"""
    if not reminders_enabled:
        return
    
    if medicine_name:
        speak(f"It's time to take your {medicine_name} medicine.")
    else:
        speak("It's time to take your medicine.")
    
    # Ask for confirmation
    speak("Did you take your medicine? Please say yes or no.")
    response = listen_once(timeout=10, phrase_time_limit=5)
    
    if response and "yes" in response:
        current_time = datetime.datetime.now().strftime("%H:%M")
        with _daily_log_lock:
            _daily_log["medicine_taken"][current_time] = medicine_name if medicine_name else "medicine"
            save_daily_log()
        speak("Great! I've noted that you took your medicine.")
    else:
        speak("Please remember to take your medicine soon. I'll remind you again in 30 minutes.")
        # Schedule another reminder
        threading.Timer(30 * 60, lambda: medicine_reminder(medicine_name)).start()

def hydration_reminder():
    """Remind to drink water"""
    if not reminders_enabled:
        return
    speak("It's been a while since you had some water. Would you like to drink some water now?")
    try:
        with _daily_log_lock:
            _daily_log["hydration_reminders"] += 1
            save_daily_log()
    except Exception as e:
        print(f"Error updating hydration reminders: {e}")

def movement_encouragement():
    """Encourage movement"""
    if not reminders_enabled:
        return
    activities = [
        "How about taking a short walk around the room?",
        "Would you like to do some light stretching?",
        "Let's do some simple exercises to keep you active."
    ]
    speak(random.choice(activities))
    try:
        with _daily_log_lock:
            _daily_log["movement_encouragements"] += 1
            save_daily_log()
    except Exception as e:
        print(f"Error updating movement encouragements: {e}")

def idle_check_in():
    """Check in when idle for too long"""
    if not reminders_enabled:
        return
    greetings = [
        "Just checking in to see how you're doing.",
        "Hello there, is everything alright?",
        "I'm here if you need anything."
    ]
    speak(random.choice(greetings))
    try:
        with _daily_log_lock:
            _daily_log["wellness_checks"].append(datetime.datetime.now().isoformat())
            save_daily_log()
    except Exception as e:
        print(f"Error updating wellness checks: {e}")

def morning_routine():
    if not reminders_enabled:
        return
    speak("Good morning! Did you sleep well?")
    resp = listen_once(timeout=15, phrase_time_limit=10)
    speak("Please take your morning medicine if any is due, and have some water. I'll check in later.")

def evening_routine():
    if not reminders_enabled:
        return
    speak("Good evening! How was your day?")
    resp = listen_once(timeout=15, phrase_time_limit=10)
    speak("Please remember your evening medicine. Sleep well, I'll be here tomorrow.")

def conversation():
    topics = [
        "Tell me about your favorite memory.",
        "What's something that made you smile recently?",
        "If you could travel anywhere, where would you go?",
        "What's your favorite season and why?"
    ]
    
    speak(random.choice(topics))
    # Just listen - no need to process the response for this simple implementation
    listen_once(timeout=30, phrase_time_limit=20)

def memory_game():
    questions = [
        ("What day is it today?", datetime.datetime.now().strftime("%A")),
        ("What month is it?", datetime.datetime.now().strftime("%B")),
        ("What year is it?", datetime.datetime.now().strftime("%Y"))
    ]
    
    question, answer = random.choice(questions)
    speak(question)
    
    response = listen_once(timeout=20, phrase_time_limit=10)
    if response and answer.lower() in response:
        speak("That's correct! Well done!")
    else:
        speak(f"The answer is {answer}. It's good to keep our memory sharp.")

def get_temperature():
    """Get temperature from sensor or API (placeholder implementation)"""
    # In a real implementation, this would read from a sensor or weather API
    # For now, we'll return a random temperature between 60-85°F
    return random.randint(60, 85)

def temperature_check():
    temp = get_temperature()
    if temp < TEMPERATURE_THRESHOLDS["cold"]:
        speak(f"I notice it's quite cold ({temp}°F). Would you like me to suggest turning on the heater or getting a blanket?")
    elif temp > TEMPERATURE_THRESHOLDS["hot"]:
        speak(f"I notice it's quite warm ({temp}°F). Would you like me to suggest turning on the fan or getting some water?")
    else:
        return
    
    response = listen_once(timeout=20, phrase_time_limit=10)
    if response and "yes" in response:
        if temp < TEMPERATURE_THRESHOLDS["cold"]:
            speak("You might want to get a warm blanket or sweater, or I can remind you to adjust the thermostat.")
        else:
            speak("You might want to turn on a fan, have a cool drink, or wear lighter clothing.")

def small_talk(cmd):
    global reminders_enabled
    if any(w in cmd for w in ("hello", "hi", "good morning", "good afternoon", "good evening")):
        hour = datetime.datetime.now().hour
        if hour < 12:
            speak("Good morning! How are you today?")
        elif hour < 18:
            speak("Good afternoon! How are you feeling?")
        else:
            speak("Good evening! How was your day?")
        # Listen after greeting
        response = listen_once(timeout=20, phrase_time_limit=10)
        if response:
            small_talk(response)  # Recursive for follow-up
    elif any(w in cmd for w in ("how are you", "how do you feel")):
        speak("I'm doing wonderfully, thank you for asking. How are you?")
        response = listen_once(timeout=15, phrase_time_limit=10)
        if response:
            if "good" in response or "fine" in response:
                speak("That's great to hear!")
            else:
                speak("I'm here if you need to talk.")
    elif any(w in cmd for w in ("thank you", "thanks")):
        speak("You're very welcome. I'm always here if you need anything.")
    elif any(w in cmd for w in ("i love you", "you're nice")):
        speak("That is so sweet of you to say. You brighten my day!")
    elif any(w in cmd for w in ("i'm lonely", "i feel alone")):
        speak("I'm here with you. Would you like to play a memory game or just chat?")
        response = listen_once(timeout=10, phrase_time_limit=5)
        if response and "game" in response:
            memory_game()
        elif response and "chat" in response:
            conversation()
    elif any(w in cmd for w in ("play game", "memory game", "let's play")):
        memory_game()
    elif any(w in cmd for w in ("talk", "chat", "conversation")):
        conversation()
    elif any(w in cmd for w in ("temperature", "weather", "hot", "cold")):
        temperature_check()
    elif any(w in cmd for w in ("emergency", "help", "i need help")):
        handle_emergency()
    elif "turn off reminders" in cmd or "stop reminders" in cmd:
        reminders_enabled = False
        speak("Reminders have been turned off. Say 'turn on reminders' to enable them again.")
        return True
    elif "turn on reminders" in cmd:
        reminders_enabled = True
        speak("Reminders have been turned on.")
        return True
    else:
        return False
    return True

def tell_joke():
    """Tell a joke"""
    try:
        joke = pyjokes.get_joke()
        speak(joke)
    except:
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a fake noodle? An impasta!"
        ]
        speak(random.choice(jokes))

def give_time():
    """Tell the current time"""
    current_time = datetime.datetime.now().strftime("%I:%M %p")
    speak(f"The current time is {current_time}")

def give_date():
    """Tell the current date"""
    current_date = datetime.datetime.now().strftime("%A, %B %d, %Y")
    speak(f"Today is {current_date}")

def search_google():
    """Search on Google"""
    speak("What would you like me to search for?")
    query = listen_once(timeout=10, phrase_time_limit=5)
    if query:
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(search_url)
        speak(f"Here are the search results for {query}")

def open_app(cmd):
    """Open an application"""
    apps = {
        "chrome": "chrome",
        "browser": "chrome",
        "notepad": "notepad",
        "calculator": "calc",
        "paint": "mspaint",
        "media player": "wmplayer"
    }
    
    for app_name, app_command in apps.items():
        if app_name in cmd:
            try:
                if platform.system() == "Windows":
                    os.system(f"start {app_command}")
                else:
                    os.system(app_command)
                speak(f"Opening {app_name}")
                return
            except:
                speak(f"Sorry, I couldn't open {app_name}")
                return
    
    speak("I'm not sure which application you want me to open.")

def manage_medicine(cmd):
    """Manage medicine schedule"""
    if "add medicine" in cmd or "new medicine" in cmd:
        speak("Which medicine would you like to schedule?")
        medicine = listen_once(timeout=10, phrase_time_limit=5)
        if medicine:
            speak("What time should I remind you? Please say the time in hours and minutes.")
            time_str = listen_once(timeout=10, phrase_time_limit=5)
            if time_str:
                MEDICINE_SCHEDULE[medicine] = time_str
                save_medicine_schedule()
                speak(f"I've set a reminder for {medicine} at {time_str}")
    else:
        speak("You can ask me to set medicine reminders or check your medicine schedule.")

# ===== ENHANCED GESTURE RECOGNITION WITH MEDIAPIPE =====
def initialize_camera():
    """Initialize camera and MediaPipe for monitoring with optimized settings"""
    global _camera, _hands, _face_detection
    try:
        if _camera is None or not _camera.isOpened():
            _camera = cv2.VideoCapture(0)
            _camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            _camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            _camera.set(cv2.CAP_PROP_FPS, 30)
            _camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if _hands is None:
            _hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        
        if _face_detection is None:
            _face_detection = mp_face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=0.5)
            
        if _camera.isOpened():
            return True
        else:
            return False
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return False

def release_camera():
    """Properly release camera and MediaPipe resources"""
    global _camera, _camera_window_open, _hands, _face_detection
    try:
        if _camera is not None:
            _camera.release()
        _camera = None
        if _hands is not None:
            _hands.close()
            _hands = None
        if _face_detection is not None:
            _face_detection.close()
            _face_detection = None
        if _camera_window_open:
            cv2.destroyAllWindows()
            _camera_window_open = False
    except Exception as e:
        print(f"Error releasing camera: {e}")

def detect_hands_and_gestures(frame):
    """Detect hands and recognize gestures using MediaPipe"""
    gesture = "none"
    hand_landmarks_list = []
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            gesture_type, confidence = analyze_gesture(hand_landmarks, frame.shape)
            if confidence > GESTURE_CONFIDENCE_THRESHOLD:
                gesture = gesture_type
                hand_landmarks_list.append(hand_landmarks)
    
    return gesture, frame, hand_landmarks_list

def analyze_gesture(hand_landmarks, frame_shape):
    """Analyze hand landmarks to recognize specific gestures"""
    frame_height, frame_width = frame_shape[:2]
    
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.append((lm.x * frame_width, lm.y * frame_height))
    
    wrist = landmarks[0]
    hand_vertical_position = wrist[1] / frame_height
    
    # Hand raise detection: hand high in frame + multiple fingers extended
    if hand_vertical_position < 0.4:
        fingers_extended = count_extended_fingers(landmarks)
        if fingers_extended >= 3:
            return "hand_raised", 0.85
    
    # Waving detection
    if len(_wrist_y_history) > MIN_WAVING_FRAMES:
        if is_waving_gesture(landmarks):
            return "waving", 0.9
    
    return "none", 0.0

def count_extended_fingers(landmarks):
    """Count how many fingers are extended"""
    # Thumb
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    thumb_cmc = landmarks[2]
    thumb_extended = False
    # For thumb, check if it's extended away from palm (approximate based on y for vertical, but adjust for thumb)
    if thumb_tip[1] < thumb_ip[1]:  # Thumb up if tip higher than IP
        thumb_extended = True

    # Other fingers: tip y < pip y
    index_tip, index_pip = landmarks[8], landmarks[6]
    middle_tip, middle_pip = landmarks[12], landmarks[10]
    ring_tip, ring_pip = landmarks[16], landmarks[14]
    pinky_tip, pinky_pip = landmarks[20], landmarks[18]

    extended_count = sum([
        thumb_extended,
        index_tip[1] < index_pip[1],
        middle_tip[1] < middle_pip[1],
        ring_tip[1] < ring_pip[1],
        pinky_tip[1] < pinky_pip[1]
    ])
    
    return extended_count

def is_waving_gesture(current_landmarks):
    """Check if hand is waving using temporal analysis of wrist x position"""
    landmarks = []
    for lm in current_landmarks.landmark:
        landmarks.append((lm.x, lm.y))  # Normalized coordinates
    
    current_wrist_x = landmarks[0][0]  # Wrist x (normalized)
    
    # Append current wrist x to history
    _wrist_y_history.append(current_wrist_x)  # Note: using x for horizontal movement
    
    if len(_wrist_y_history) < MIN_WAVING_FRAMES:
        return False
    
    wrist_movements = list(_wrist_y_history)
    
    # Count direction changes for waving (horizontal movement)
    direction_changes = 0
    threshold = 0.02  # Minimum movement threshold to count as change
    for i in range(1, len(wrist_movements) - 1):
        prev_diff = wrist_movements[i] - wrist_movements[i-1]
        curr_diff = wrist_movements[i+1] - wrist_movements[i]
        
        if abs(prev_diff) > threshold and abs(curr_diff) > threshold and prev_diff * curr_diff < 0:
            direction_changes += 1
    
    # Waving if at least 2 direction changes in recent history
    return direction_changes >= 2

def detect_posture(frame):
    """Posture detection using MediaPipe Face Detection with height adjustments"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _face_detection.process(rgb_frame)
    
    posture = "unknown"
    face_regions = []
    
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)
            
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            face_regions.append((x, y, width, height))
            
            vertical_position = y + height/2
            standing_threshold = h * (1 - (ROOM_CONFIG["average_standing_height"] / ROOM_CONFIG["ceiling_height"]))
            sitting_threshold = h * (1 - (ROOM_CONFIG["average_sitting_height"] / ROOM_CONFIG["ceiling_height"]))
            
            if vertical_position < standing_threshold:
                posture = "standing"
                cv2.putText(frame, "STANDING", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            elif vertical_position < sitting_threshold:
                posture = "sitting"
                cv2.putText(frame, "SITTING", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                posture = "lying"
                cv2.putText(frame, "LYING DOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return posture, frame, face_regions

def check_for_fall():
    """Check if a fall has occurred based on posture history"""
    if len(_posture_history) < 5:
        return False
    
    recent_postures = list(_posture_history)[-5:]
    
    if all(p in ["standing", "sitting"] for p in recent_postures[:3]) and \
       all(p == "lying" for p in recent_postures[3:]):
        return True
    
    return False

def camera_monitor_loop():
    """Dedicated thread for high-FPS camera monitoring"""
    global _current_posture, _camera_window_open, CAMERA_ENABLED
    global _last_gesture_time, _last_fall_detection_time, _is_speaking
    global _gesture_history, _hand_landmarks_history, _awaiting_response
    
    while True:
        if not CAMERA_ENABLED:
            release_camera()
            time.sleep(0.1)
            continue
        
        if not initialize_camera():
            time.sleep(1)
            continue
        
        ret, frame = _camera.read()
        if not ret:
            continue
        
        if _is_speaking or _awaiting_response:
            continue
        
        display_frame = frame.copy()
        
        face_regions = []
        if POSE_DETECTION_ENABLED:
            posture, display_frame, face_regions = detect_posture(display_frame)
            _current_posture = posture
            _posture_history.append(posture)
            
            current_hour = datetime.datetime.now().hour
            if (posture == "lying" and 
                current_hour >= 8 and current_hour <= 20 and 
                list(_posture_history).count("lying") > 5):
                pass
        
        gesture_detected = "none"
        hand_landmarks_list = []
        if GESTURE_RECOGNITION_ENABLED:
            gesture_detected, display_frame, hand_landmarks_list = detect_hands_and_gestures(display_frame)
            
            if hand_landmarks_list:
                # Store serializable landmarks for history
                serializable_landmarks = []
                for lm in hand_landmarks_list[0].landmark:
                    serializable_landmarks.append((lm.x, lm.y))
                _hand_landmarks_history.append(serializable_landmarks)
            else:
                _hand_landmarks_history.append([])
            
            _gesture_history.append(gesture_detected)
        
        current_time = time.time()
        if (FALL_DETECTION_ENABLED and check_for_fall() and 
            current_time - _last_fall_detection_time > _fall_cooldown):
            _last_fall_detection_time = current_time
            handle_fall_detection(immediate=True)
        
        # Convert frame to JPEG for web display
        _, buffer = cv2.imencode('.jpg', display_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        
        # Add frame to queue for web display (non-blocking)
        if not camera_frame_queue.full():
            try:
                camera_frame_queue.put_nowait(jpg_as_text)
            except:
                pass  # Queue is full, skip this frame
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            CAMERA_ENABLED = False
            release_camera()

def confirm_gesture_with_temporal_analysis(gesture):
    """Confirm gesture using temporal analysis to reduce false positives"""
    if len(_gesture_history) < 3:
        return False
    
    recent_gestures = list(_gesture_history)[-3:]
    gesture_count = sum(1 for g in recent_gestures if g == gesture)
    return gesture_count >= 2

def handle_gesture(gesture):
    """Handle detected gestures without asking for confirmation"""
    global _last_interaction_time
    
    if time.time() - _last_interaction_time > 2:
        if gesture == "hand_raised":
            speak("I see you raised your hand. Alerting emergency contacts.")
            handle_emergency("Hand raised for help")
            try:
                with _daily_log_lock:
                    _daily_log["gesture_recognitions"] += 1
                    save_daily_log()
            except Exception as e:
                print(f"Error updating gesture recognitions: {e}")
                
        elif gesture == "waving":
            speak("Hello! I see you're waving. How can I help you?")
            try:
                with _daily_log_lock:
                    _daily_log["gesture_recognitions"] += 1
                    save_daily_log()
            except Exception as e:
                print(f"Error updating gesture recognitions: {e}")

def handle_fall_detection(immediate=False):
    """Handle detected falls with direct emergency response"""
    global _last_interaction_time, _awaiting_response, _emergency_mode
    
    if immediate or time.time() - _last_interaction_time > 2:
        speak("Emergency! I detected a fall! Alerting your emergency contacts.")
        handle_emergency("Fall detected")
        try:
            with _daily_log_lock:
                _daily_log["fall_detections"] += 1
                save_daily_log()
        except Exception as e:
            print(f"Error updating fall detections: {e}")
        _awaiting_response = False
        _emergency_mode = False  # Ensure reset to normal mode
        return

def save_daily_log():
    """Save daily log to file with thread-safe access"""
    try:
        with _daily_log_lock:
            # Validate _daily_log structure
            if not isinstance(_daily_log, dict):
                print("Error: _daily_log corrupted, reinitializing")
                _daily_log.update({
                    "medicine_taken": {},
                    "wellness_checks": [],
                    "hydration_reminders": 0,
                    "movement_encouragements": 0,
                    "posture_alerts": 0,
                    "fall_detections": 0,
                    "gesture_recognitions": 0
                })
            with open("daily_log.json", "w") as f:
                json.dump(_daily_log, f, indent=2)
    except (PermissionError, IOError) as e:
        print(f"Error saving daily log (file access issue): {e}")
    except Exception as e:
        print(f"Error saving daily log: {e}")

# ===== SPEECH FUNCTIONS =====
async def _edge_speak(text: str):
    global _is_speaking, assistant_status
    _is_speaking = True
    assistant_status = "Speaking..."
    conversation_log.append(f"Monica: {text}")
    try:
        from edge_tts import Communicate
        cache = os.path.join(CACHE_DIR, f"monica_{abs(hash(text))}.mp3")
        if not os.path.exists(cache):
            com = Communicate(text=text, voice=VOICE, rate=RATE)
            await com.save(cache)
        import playsound
        playsound.playsound(cache, block=True)
    except Exception as e:
        print(f"Error in _edge_speak: {e}")
    finally:
        _is_speaking = False
        assistant_status = "Ready"

def speak(text: str):
    global _spoken_for_this_turn, _last_interaction_time, _is_speaking
    with speak_lock:
        print(f"Monica: {text}")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_edge_speak(text))
            finally:
                loop.close()
        except Exception as e:
            print(f"Error in speak: {e}")
        finally:
            _spoken_for_this_turn = False
            _last_interaction_time = time.time()
            _is_speaking = False  # Ensure _is_speaking is reset

def listen_once(timeout=15, phrase_time_limit=15):
    global _error_cnt, _spoken_for_this_turn, _last_interaction_time, _is_speaking, assistant_status
    
    # Wait until speaking is complete
    while _is_speaking:
        time.sleep(0.1)
    
    assistant_status = "Listening..."
    r = sr.Recognizer()
    try:
        with sr.Microphone() as src:
            print("Listening…")
            r.pause_threshold = 1.0
            r.energy_threshold = 300
            try:
                audio = r.listen(src, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except sr.WaitTimeoutError:
                assistant_status = "Ready"
                return None
        try:
            cmd = r.recognize_google(audio, language="en-US")
            print(f"You: {cmd}")
            conversation_log.append(f"You: {cmd}")
            _error_cnt = 0
            _last_interaction_time = time.time()
            assistant_status = "Ready"
            return cmd.lower()
        except sr.UnknownValueError:
            assistant_status = "Ready"
            return None
        except sr.RequestError:
            speak("I seem to be offline. Let me know when you need me.")
            assistant_status = "Ready"
            return None
    except Exception as e:
        print(f"Error in listen_once: {e}")
        assistant_status = "Ready"
        return None

# ===== DATA MANAGEMENT =====
def load_data():
    global MEDICINE_SCHEDULE, _daily_log
    try:
        with open("medicine_schedule.json", "r") as f:
            MEDICINE_SCHEDULE = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        MEDICINE_SCHEDULE = {
            "morning": "08:00",
            "afternoon": "14:00",
            "evening": "00:00"
        }
        save_medicine_schedule()

    try:
        with _daily_log_lock:
            with open("daily_log.json", "r") as f:
                _daily_log.update(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    except Exception as e:
        print(f"Error loading daily log: {e}")

def save_medicine_schedule():
    try:
        with open("medicine_schedule.json", "w") as f:
            json.dump(MEDICINE_SCHEDULE, f, indent=2)
    except Exception as e:
        print(f"Error saving medicine schedule: {e}")

# ===== OTHER FUNCTIONS =====
def wait_for_wake():
    r = sr.Recognizer()
    while True:
        try:
            with sr.Microphone() as src:
                r.pause_threshold = 0.8
                r.energy_threshold = 300
                try:
                    audio = r.listen(src, timeout=5, phrase_time_limit=3)
                except sr.WaitTimeoutError:
                    continue
            try:
                phrase = r.recognize_google(audio).lower()
                if WAKE_WORD in phrase:
                    return phrase  # Return the full phrase for greeting handling
            except:
                continue
        except Exception as e:
            print(f"Error in wait_for_wake: {e}")
            time.sleep(2)

def send_emergency_email(emergency_type="General Emergency"):
    """Send emergency notification email to contacts"""
    if not EMAIL_ENABLED:
        return False
        
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = ", ".join(EMERGENCY_CONTACTS)
        msg['Subject'] = f"🚨 EMERGENCY ALERT: {emergency_type}"
        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
EMERGENCY ALERT FROM MONICA CARE ASSISTANT

Type: {emergency_type}
Time: {current_time}
Current Posture: {_current_posture}

This is an automated emergency alert. The elder care assistant has 
detected an emergency situation and requires immediate attention.

Please contact the person immediately to check on their well-being.

Sent by Monica Care Assistant
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            
        return True
        
    except Exception as e:
        print(f"Error sending emergency email: {e}")
        return False

def handle_emergency(emergency_type="Voice Emergency Trigger"):
    """Handle emergency with timeout to prevent getting stuck"""
    global _emergency_mode, _awaiting_response, emergency_triggered
    _emergency_mode = True
    _awaiting_response = True
    emergency_triggered = True
    
    email_sent = send_emergency_email(emergency_type)
    
    try:
        import winsound
        for _ in range(5):  # Reduced beeps to avoid delay
            winsound.Beep(1000, 1000)
            time.sleep(1)
    except:
        for _ in range(3):  # Reduced to avoid delay
            speak("Emergency! Help needed!")
            time.sleep(1)
    
    if email_sent:
        speak("I've alerted your emergency contacts via email. Help is on the way.")
    else:
        speak("I tried to alert your contacts but encountered an error. Please call for help directly.")
    
    # Ask for confirmation with timeout
    speak("Are you okay now? ")
    response = listen_once(timeout=10, phrase_time_limit=10)
    
    if response and ("yes" in response or "ok" in response or "fine" in response):
        speak("I'm glad to hear you're okay. I'll return to normal mode.")
    else:
        speak("No response received. Returning to normal mode. Say 'Monica' if you need me.")
    
    # Force reset to normal mode
    _emergency_mode = False
    _awaiting_response = False
    _last_interaction_time = time.time()
    emergency_triggered = False

def check_scheduled_events():
    """Background thread to check for scheduled events"""
    global _last_hydration_reminder, _last_movement_reminder, _last_interaction_time
    global _last_morning_date, _last_evening_date, _last_medicine_reminder
    
    while True:
        if not reminders_enabled:
            time.sleep(5)
            continue
        
        try:
            now = datetime.datetime.now()
            current_time = now.strftime("%H:%M")
            current_hour = now.hour
            current_date = now.date()
            
            # Medicine reminders with cooldown
            for medicine, schedule_time in MEDICINE_SCHEDULE.items():
                if current_time == schedule_time:
                    last_time = _last_medicine_reminder.get(medicine, 0)
                    if time.time() - last_time > 60:  # 1 min cooldown
                        medicine_reminder(medicine)
                        _last_medicine_reminder[medicine] = time.time()
            
            # Hydration reminder
            if time.time() - _last_hydration_reminder > HYDRATION_INTERVAL:
                hydration_reminder()
                _last_hydration_reminder = time.time()
            
            # Movement reminder
            if time.time() - _last_movement_reminder > MOVEMENT_INTERVAL:
                movement_encouragement()
                _last_movement_reminder = time.time()
            
            # Idle check
            if time.time() - _last_interaction_time > IDLE_CHECK_INTERVAL:
                idle_check_in()
                _last_interaction_time = time.time()
            
            # Morning and evening routines once per day
            if current_hour == 8 and (_last_morning_date is None or _last_morning_date < current_date):
                morning_routine()
                _last_morning_date = current_date
            elif current_hour == 19 and (_last_evening_date is None or _last_evening_date < current_date):
                evening_routine()
                _last_evening_date = current_date
            
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in scheduled events: {e}")
            time.sleep(1)

def main_backend():
    """Main backend loop to run in a separate thread"""
    global _emergency_mode, _last_interaction_time
    
    load_data()
    
    schedule_thread = threading.Thread(target=check_scheduled_events, daemon=True)
    schedule_thread.start()
    
    camera_thread = threading.Thread(target=camera_monitor_loop, daemon=True)
    camera_thread.start()
    
    speak("Hello, I'm Monica, your care assistant. I'm here to help you.")
    
    while True:
        if _emergency_mode:
            speak("I'm still in emergency mode. Are you okay now?")
            response = listen_once(timeout=30, phrase_time_limit=15)
            if response and ("yes" in response or "ok" in response or "fine" in response):
                _emergency_mode = False
                speak("I'm glad to hear you're okay. I'll return to normal mode.")
            else:
                _emergency_mode = False  # Force reset to avoid getting stuck
                speak("No response received. Returning to normal mode. Say 'Monica' if you need me.")
            continue
        
        wake_phrase = wait_for_wake()
        # Handle greeting if wake phrase includes hello
        if wake_phrase and "hello" in wake_phrase:
            hour = datetime.datetime.now().hour
            if hour < 12:
                speak("Good morning! How are you?")
            elif hour < 18:
                speak("Good afternoon! Nice to hear from you.")
            else:
                speak("Good evening! How can I assist?")
        else:
            speak("Yes? How can I help you?")
        
        while True:
            if time.time() - _last_interaction_time > INACTIVITY_TIMEOUT:
                speak("I haven't heard from you in a while. Say 'Monica' when you need me.")
                break
                
            cmd = listen_once(timeout=30, phrase_time_limit=20)
            
            if not cmd:
                continue
                
            if any(w in cmd for w in ["emergency", "help", "i need help", "i fell"]):
                handle_emergency()
                break
                
            if any(w in cmd for w in ("bye", "exit", "quit", "goodbye", "stop", "off")):
                speak("Take care. I'll be right here whenever you need me.")
                release_camera()
                save_daily_log()
                return
                
            if small_talk(cmd):
                continue
            if "joke" in cmd:
                tell_joke()
            elif "time" in cmd:
                give_time()
            elif "date" in cmd or "day" in cmd:
                give_date()
            elif "search" in cmd or "google" in cmd:
                search_google()
            elif "open" in cmd:
                open_app(cmd)
            elif "medicine" in cmd:
                manage_medicine(cmd)
            else:
                speak("I'm not sure what you meant. You can ask me about medicine, time, or we can just chat.")

# ===== DASH LAYOUT =====
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("Monica Elder Care Assistant", 
                   className="text-center mb-4",
                   style={"color": "#2c3e50", "fontWeight": "bold"})
        ])
    ]),
    
    # Status and Controls Row
   # Camera Feed Row - Full width above controls
# Camera Feed Row - Proper vertical aspect ratio
dbc.Row([
    dbc.Col([
        html.H3("Live Camera Monitoring", className="text-center mb-3", style={"color": "#34495e"}),
        html.Div(
            html.Img(
                id="camera-image",
                style={
                    "width": "100%",
                    "height": "auto",
                    "maxWidth": "400px",  # Limit maximum width
                    "borderRadius": "10px",
                    "border": "3px solid #3498db"
                }
            ),
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "center"
            }
        ),
        dcc.Interval(
            id='camera-interval',
            interval=100,
            n_intervals=0
        )
    ], width=12),
], className="mb-4"),

# Controls Row - Below camera in horizontal layout
dbc.Row([
    dbc.Col([
        dbc.Row([
            # Status Card
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Assistant Status", className="text-white", style={"backgroundColor": "#3498db"}),
                    dbc.CardBody([
                        html.H4(id="status-text", children="Ready", className="card-text"),
                        html.P(id="posture-text", children="Posture: Unknown", className="card-text"),
                        html.P(id="gesture-text", children="Last Gesture: None", className="card-text")
                    ])
                ])
            ], width=4),
            
            # Emergency and Speak Buttons
            dbc.Col([
                # Emergency Button
                dbc.Button(
                    "🚨 EMERGENCY",
                    id="emergency-btn",
                    color="danger",
                    size="lg",
                    className="w-100 mb-3",
                    style={"fontSize": "1.2rem", "fontWeight": "bold", "height": "80px"}
                ),
                
                # Speak Button
                dbc.Button(
                    "🎤 Speak to Monica",
                    id="speak-btn",
                    color="primary",
                    size="lg",
                    className="w-100",
                    style={"fontSize": "1.2rem", "fontWeight": "bold", "height": "60px"}
                ),
            ], width=4),
            
            # Conversation Log
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Conversation Log", className="text-white", style={"backgroundColor": "#2c3e50"}),
                    dbc.CardBody([
                        html.Div(
                            id="conversation-log",
                            style={
                                "height": "150px",
                                "overflowY": "auto",
                                "backgroundColor": "#f8f9fa",
                                "padding": "10px",
                                "borderRadius": "5px",
                                "fontSize": "0.9rem"
                            }
                        )
                    ])
                ])
            ], width=4)
        ])
    ], width=12)  # Full width
]),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.P("Monica Elder Care Assistant - Always here to help", 
                  className="text-center mt-4 text-muted")
        ])
    ])
], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh", "padding": "20px"})

# ===== DASH CALLBACKS =====
@app.callback(
    [Output("camera-image", "src"),
     Output("status-text", "children"),
     Output("posture-text", "children"),
     Output("gesture-text", "children"),
     Output("conversation-log", "children")],
    [Input("camera-interval", "n_intervals"),
     Input("emergency-btn", "n_clicks"),
     Input("speak-btn", "n_clicks")],
    prevent_initial_call=False
)
def update_dashboard(n, emergency_clicks, speak_clicks):
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Handle emergency button
    if triggered_id == "emergency-btn" and emergency_clicks:
        threading.Thread(target=handle_emergency, args=("Web Dashboard Emergency Trigger",), daemon=True).start()
    
    # Handle speak button
    if triggered_id == "speak-btn" and speak_clicks:
        threading.Thread(target=listen_and_respond, daemon=True).start()
    
    # Get latest camera frame
    camera_frame = None
    if not camera_frame_queue.empty():
        try:
            camera_frame = camera_frame_queue.get_nowait()
        except:
            pass
    
    # Create camera feed component
    if camera_frame:
        camera_src = f"data:image/jpeg;base64,{camera_frame}"
    else:
        camera_src = ""  # Empty when no frame
    
    # Update status texts
    status_text = f"Status: {assistant_status}"
    posture_text = f"Posture: {_current_posture.title()}"
    
    # Get last gesture
    last_gesture = "None"
    if _gesture_history:
        for gesture in reversed(_gesture_history):
            if gesture != "none":
                last_gesture = gesture.replace("_", " ").title()
                break
    
    gesture_text = f"Last Gesture: {last_gesture}"
    
    # Update conversation log
    conversation_display = []
    for i, msg in enumerate(conversation_log[-10:]):  # Show last 10 messages
        conversation_display.append(
            html.P(msg, style={
                "margin": "2px 0",
                "padding": "2px 5px",
                "backgroundColor": "#e9ecef" if "Monica:" in msg else "white",
                "borderRadius": "3px"
            })
        )
    
    if not conversation_display:
        conversation_display = [html.P("No conversation yet...", className="text-muted")]
    
    return camera_src, status_text, posture_text, gesture_text, conversation_display

def listen_and_respond():
    """Function to handle listening when speak button is pressed"""
    cmd = listen_once(timeout=20, phrase_time_limit=15)
    if cmd:
        # Process the command
        if any(w in cmd for w in ["emergency", "help", "i need help", "i fell"]):
            handle_emergency()
        elif any(w in cmd for w in ("bye", "exit", "quit", "goodbye", "stop", "off")):
            speak("Take care. I'll be right here whenever you need me.")
        elif small_talk(cmd):
            pass
        elif "joke" in cmd:
            tell_joke()
        elif "time" in cmd:
            give_time()
        elif "date" in cmd or "day" in cmd:
            give_date()
        elif "search" in cmd or "google" in cmd:
            search_google()
        elif "open" in cmd:
            open_app(cmd)
        elif "medicine" in cmd:
            manage_medicine(cmd)
        else:
            speak("I'm not sure what you meant. You can ask me about medicine, time, or we can just chat.")

# ===== START APPLICATION =====
if __name__ == "__main__":
    # Start the backend in a separate thread
    backend_thread = threading.Thread(target=main_backend, daemon=True)
    backend_thread.start()
    
    # Start the Dash app
    print("Starting Monica Elder Care Assistant Web Dashboard...")
    print("Access the dashboard at: http://127.0.0.1:8050")
    app.run_server(debug=False, host='0.0.0.0', port=8050)