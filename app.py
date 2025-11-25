"""Indian Sign Language (ISL) Translator - Flask Web Application

This application provides real-time Indian Sign Language detection and translation
using computer vision and machine learning. It uses MediaPipe for hand tracking
and a TensorFlow/Keras model for gesture classification.

Features:
    - Real-time hand gesture recognition using MediaPipe
    - Letter-by-letter word building with hold-to-confirm mechanism
    - ASL 'H' gesture for space insertion
    - Multi-strategy autocorrect (pyspellchecker, Yandex API, frequency-based)
    - Sentence persistence and export functionality
    - Cross-platform optimization (Windows/macOS/Linux)

Architecture:
    - Multi-threaded design with separate capture and processing threads
    - MJPEG streaming for real-time video feed
    - RESTful API endpoints for detection metadata

 
"""

from flask import Flask, render_template, Response, jsonify, request
import cv2
cv2.setNumThreads(1)
import mediapipe as mp
import copy
import itertools
import numpy as np
import pandas as pd
import string
import time
from datetime import datetime
import os
import json
import base64
import threading
import platform
from queue import Queue
from collections import deque
import math
import traceback
import difflib
import requests
import tensorflow as tf

def _configure_tf():
    """Configure TensorFlow threading for optimal performance.
    
    Limits inter and intra operation parallelism to prevent CPU overutilization
    and maintain responsive model inference during real-time processing.
    
    Raises:
        RuntimeError: If TensorFlow threading configuration fails (non-critical).
    """
    try:
        tf.config.threading.set_inter_op_parallelism_threads(2)
        tf.config.threading.set_intra_op_parallelism_threads(2)
    except RuntimeError as e:
        print(f"TF threading config skipped: {e}")

_configure_tf()

from tensorflow import keras

app = Flask(__name__)

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
    _spell = SpellChecker(distance=1)
    _spell2 = None
    print("pyspellchecker available")
except Exception:
    SPELLCHECKER_AVAILABLE = False
    _spell = None
    _spell2 = None
    print("pyspellchecker NOT available, falling back to lighter autocorrect")

try:
    from wordfreq import zipf_frequency
    WORDFREQ_AVAILABLE = True
    print("wordfreq available")
except Exception:
    WORDFREQ_AVAILABLE = False
    print("wordfreq NOT available, using fallback frequencies")
model = None
try:
    if os.path.exists("model.h5"):
        model = keras.models.load_model("model.h5", compile=False)
        print("Model loaded successfully")
    else:
        print("Warning: model.h5 not found. Running without model. Place model.h5 next to this script.")
except Exception as e:
    print(f"Error loading model.h5: {e}")
    traceback.print_exc()
    model = None

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# ===== CONFIGURATION CONSTANTS =====
alphabet = ['1','2','3','4','5','6','7','8','9'] + list(string.ascii_uppercase)

# ===== GLOBAL STATE VARIABLES =====
camera = None
hands_detector = None

frame_queue = Queue(maxsize=2)
latest_frame = None
processed_result = None
result_lock = threading.Lock()
frame_lock = threading.Lock()
latest_jpeg_bytes = None
jpeg_lock = threading.Lock()
capture_thread = None
processing_thread = None
is_capturing = False
is_processing = False

# ===== PLATFORM-SPECIFIC SETTINGS =====
IS_WINDOWS = platform.system() == 'Windows'

FRAME_WIDTH = 640 if not IS_WINDOWS else 480
FRAME_HEIGHT = 480 if not IS_WINDOWS else 360
PROCESS_EVERY_N_FRAMES = 1 if not IS_WINDOWS else 2
JPEG_QUALITY = 85 if not IS_WINDOWS else 70
PROCESSING_WIDTH = 320 if not IS_WINDOWS else 240
PROCESSING_HEIGHT = 240 if not IS_WINDOWS else 180
ENCODE_PROCESSED_FRAME = False
DRAW_LANDMARKS = True
CAPTURE_GRAB_SKIP = 1 if not IS_WINDOWS else 2

last_process_time = 0
MIN_PROCESS_INTERVAL = 0.1 if not IS_WINDOWS else 0.1

# ===== GESTURE RECOGNITION STATE =====
current_sentence = ""
current_word = ""
last_prediction = ""
prediction_start_time = 0
HOLD_TIME = 1.5
last_prediction_time = 0

# ===== SPACE GESTURE STATE =====
ASL_SPACE_ENABLED = True
ASL_SPACE_HOLD_TIME = 1.5
space_gesture_active = False
space_gesture_start_time = 0.0
space_gesture_cooldown_until = 0.0

# ===== AUTOCORRECT STATE =====
last_autocorrected_word = ""
last_autocorrect_message = ""

# ===== AUTOCORRECT CONFIGURATION =====
COMMON_WORD_FREQ = {
    'good': 6.5, 'day': 6.4, 'morning': 5.8, 'evening': 5.6, 'night': 6.0,
    'hello': 5.9, 'hi': 5.5, 'bye': 5.1, 'please': 6.1, 'sorry': 5.6,
    'thanks': 6.2, 'thank': 6.2, 'yes': 6.7, 'no': 7.1, 'okay': 5.8, 'ok': 6.5,
    'name': 6.1, 'help': 6.0, 'like': 6.5, 'want': 6.3, 'go': 6.9, 'come': 6.4,
    'where': 6.3, 'what': 7.0, 'who': 6.7, 'when': 6.8, 'why': 6.3, 'how': 6.9,
    'today': 6.0, 'tomorrow': 5.7, 'yesterday': 5.7,
    'food': 5.5, 'hood': 4.8, 'look': 5.7, 'book': 5.9, 'back': 6.2,
    'time': 7.0, 'have': 7.1, 'that': 7.2, 'with': 6.6, 'from': 6.5,
}
_FREQ_MARGIN = 0.35

# ===== AUTOCORRECT HELPER FUNCTIONS =====

def _frequency_score(word: str) -> float:
    """Calculate frequency score for a word.
    
    Returns a Zipf-style frequency score used for ranking autocorrect candidates.
    Uses wordfreq library if available, otherwise falls back to COMMON_WORD_FREQ.
    
    Args:
        word: The word to score.
    
    Returns:
        float: Frequency score (higher means more common).
    """
    if not word:
        return 0.0
    if WORDFREQ_AVAILABLE:
        try:
            return zipf_frequency(word, "en")
        except Exception:
            pass
    return COMMON_WORD_FREQ.get(word.lower(), 0.0)


def _frequency_candidates(word: str):
    """Generate edit-distance-1 spelling variants.
    
    Creates candidate words by applying single-character edits:
    deletions, transpositions, replacements, and insertions.
    
    Args:
        word: The original word.
    
    Returns:
        set: Set of candidate words (capped at 2048 for performance).
    """
    letters = string.ascii_lowercase
    w = word.lower()
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + (R[1:] if R else '') for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    cands = set(deletes + transposes + replaces + inserts)
    cands.discard(w)
    if len(cands) > 2048:
        cands = set(list(cands)[:2048])
    return cands


def _frequency_based_suggestion(word: str) -> str:
    """Suggest corrections using frequency-based ranking.
    
    Searches edit-distance-1 and edit-distance-2 neighbors, preferring
    higher-frequency alternatives over the original word.
    
    Args:
        word: The word to correct.
    
    Returns:
        str: Suggested correction, or empty string if no better alternative found.
    """
    original_lower = word.lower()
    if len(original_lower) <= 2:
        return ""
    original_score = _frequency_score(original_lower)

    def pick_best(candidates, base_score, margin):
        best = ""
        best_score = base_score
        for cand in candidates:
            if not any(ch.isalpha() for ch in cand):
                continue
            score = _frequency_score(cand)
            if score > best_score + margin:
                best, best_score = cand, score
        return best

    best = pick_best(_frequency_candidates(original_lower), original_score, _FREQ_MARGIN)
    if best:
        return best

    ed2 = set()
    for e1 in list(_frequency_candidates(original_lower))[:512]:
        ed2.update(_frequency_candidates(e1))
        if len(ed2) > 4096:
            break
    best = pick_best(ed2, original_score, _FREQ_MARGIN + 0.1)
    return best


def _preserve_case(suggestion: str, original: str) -> str:
    """Preserve the original word's casing pattern.
    
    Applies the casing pattern of the original word to the suggested correction:
    - All uppercase -> convert suggestion to uppercase
    - First letter capitalized -> capitalize suggestion
    - Otherwise -> return suggestion as-is
    
    Args:
        suggestion: The suggested correction.
        original: The original word.
    
    Returns:
        str: Suggestion with preserved casing.
    """
    if not suggestion:
        return suggestion
    if original.isupper():
        return suggestion.upper()
    if original and original[0].isupper():
        return suggestion.capitalize()
    return suggestion

def get_autocorrect_suggestion(word):
    """Get spelling correction suggestion using multiple strategies.
    
    Attempts correction in the following order:
    1. pyspellchecker (offline, edit distances 1-2)
    2. Yandex SpellService API (online, English)
    3. Frequency-based correction (using word frequency heuristics)
    4. Local difflib with system wordlist fallback
    
    Args:
        word: The word to correct.
    
    Returns:
        str: Corrected word with original casing preserved, or original word if no correction found.
    """
    try:
        if not word or not word.strip():
            return word
        original = word
        word = word.strip()
        word_lower = word.lower()

        try:
            if SPELLCHECKER_AVAILABLE and _spell is not None:
                cand = _spell.correction(word_lower)
                if cand and cand.lower() != word_lower:
                    return _preserve_case(cand, original)
                global _spell2
                if _spell2 is None:
                    try:
                        _spell2 = SpellChecker(distance=2)
                    except Exception:
                        _spell2 = None
                if _spell2 is not None:
                    cand2 = _spell2.correction(word_lower)
                    if cand2 and cand2.lower() != word_lower:
                        return _preserve_case(cand2, original)
        except Exception:
            pass

        try:
            resp = requests.get(
                "https://speller.yandex.net/services/spellservice.json/checkText",
                params={"text": word, "lang": "en"},
                timeout=1.2,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        suggestions = item.get("s") or []
                        for cand in suggestions:
                            if not isinstance(cand, str):
                                continue
                            cand = cand.strip()
                            if not cand or cand.lower() == word_lower:
                                continue
                            if any(ch.isalpha() for ch in cand):
                                return _preserve_case(cand, original)
        except Exception:
            pass

        try:
            freq_candidate = _frequency_based_suggestion(word_lower)
            if freq_candidate and freq_candidate != word_lower:
                return _preserve_case(freq_candidate, original)
        except Exception:
            pass

        try:
            small_wordlist = [
                'good','day','morning','evening','night','hello','hi','bye','please','sorry',
                'thanks','thank','yes','no','okay','ok','name','help','like','want','go','come',
                'where','what','who','when','why','how','today','tomorrow','yesterday',
                'the','and','is','in','it','you','that','he','was','for','on','are','as','with','his','they',
                'i','this','have','be','at','one','not','but','what','all','were','we','when','your','can','said',
                'there','use','an','each','which','she','do','how','their','if','will','up','other','about'
            ]
            sys_list = []
            try:
                if os.path.exists('/usr/dict/words'):
                    with open('/usr/dict/words', 'r', encoding='utf-8', errors='ignore') as f:
                        sys_list = [w.strip() for w in f if w.strip()]
                elif os.path.exists('/usr/share/dict/words'):
                    with open('/usr/share/dict/words', 'r', encoding='utf-8', errors='ignore') as f:
                        sys_list = [w.strip() for w in f if w.strip()]
            except Exception:
                sys_list = []

            candidates = sys_list if sys_list else small_wordlist
            matches = difflib.get_close_matches(word_lower, candidates, n=3, cutoff=0.6)
            if matches:
                best = max(matches, key=_frequency_score)
                return _preserve_case(best, original)
        except Exception:
            pass

        return original
    except Exception:
        return word

# ===== GESTURE DETECTION HELPER FUNCTIONS =====

def _dist(p1, p2):
    """Calculate Euclidean distance between two 2D points.
    
    Args:
        p1: First point as (x, y) tuple or list.
        p2: Second point as (x, y) tuple or list.
    
    Returns:
        float: Euclidean distance between the points.
    """
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def detect_asl_space_gesture(landmarks):
    """Detect ASL 'H' gesture used as space trigger.
    
    Detects the ASL 'H' hand shape using heuristics:
    - Index and middle fingers extended (tips above PIP joints)
    - Ring and pinky fingers folded (tips below PIP joints)
    - Index and middle fingertips close together (parallel)
    
    Args:
        landmarks: List of 21 [x, y] points representing hand landmarks in pixel coordinates.
    
    Returns:
        bool: True if 'H' gesture detected, False otherwise.
    """
    try:
        if not landmarks or len(landmarks) != 21:
            return False
        xs = [p[0] for p in landmarks]; ys = [p[1] for p in landmarks]
        w = (max(xs) - min(xs)) + 1e-5
        h = (max(ys) - min(ys)) + 1e-5

        y_margin = 0.12 * h
        close_margin = 0.22 * w

        index_tip, index_pip = landmarks[8], landmarks[6]
        middle_tip, middle_pip = landmarks[12], landmarks[10]
        ring_tip, ring_pip = landmarks[16], landmarks[14]
        pinky_tip, pinky_pip = landmarks[20], landmarks[18]

        index_extended = index_tip[1] < (index_pip[1] - 0.5*y_margin)
        middle_extended = middle_tip[1] < (middle_pip[1] - 0.5*y_margin)
        ring_folded = ring_tip[1] > (ring_pip[1] + 0.35*y_margin)
        pinky_folded = pinky_tip[1] > (pinky_pip[1] + 0.35*y_margin)
        index_middle_close = _dist(index_tip, middle_tip) < close_margin

        return index_extended and middle_extended and ring_folded and pinky_folded and index_middle_close
    except Exception:
        return False

# ===== VIDEO PROCESSING THREADS =====

def capture_frames():
    """Capture video frames continuously from camera.
    
    Runs in a dedicated thread to pull frames from the camera and store them
    in a shared buffer for processing. Implements OS-specific optimizations
    for Windows (grab/retrieve pattern) and other platforms (direct read).
    
    Global Variables Modified:
        latest_frame: Updated with newly captured frames.
    """
    global camera, latest_frame, is_capturing
    frame_count = 0
    while is_capturing:
        if camera is not None and camera.isOpened():
            try:
                if IS_WINDOWS:
                    grabbed = camera.grab()
                    if not grabbed:
                        time.sleep(0.005)
                        continue
                    for _ in range(CAPTURE_GRAB_SKIP):
                        camera.grab()
                    success, frame = camera.retrieve()
                else:
                    success, frame = camera.read()

                if success:
                    frame_count += 1
                    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                        with frame_lock:
                            latest_frame = frame.copy()
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.01)
        else:
            time.sleep(0.1)


def process_frames():
    """Process video frames and perform gesture recognition.
    
    Runs in a dedicated thread to:
    1. Retrieve frames from the capture buffer
    2. Detect hand landmarks using MediaPipe
    3. Classify gestures using the trained model
    4. Track letter hold times for word building
    5. Detect space gestures and apply autocorrect
    6. Update global state with detection results
    
    Global Variables Modified:
        processed_result: Latest detection results dictionary.
        latest_jpeg_bytes: JPEG-encoded frame for streaming.
        current_word: In-progress word being built.
        current_sentence: Complete sentence with finished words.
        last_prediction: Previously detected letter.
        space_gesture_active: Space gesture tracking flag.
        last_autocorrected_word: Most recently corrected word.
        last_autocorrect_message: Autocorrect notification message.
    """
    global latest_frame, processed_result, is_processing, hands_detector
    global current_sentence, current_word, last_prediction, prediction_start_time
    global last_prediction_time, latest_jpeg_bytes
    global space_gesture_active, space_gesture_start_time, space_gesture_cooldown_until
    global last_autocorrected_word, last_autocorrect_message

    while is_processing:
        try:
            _last_autocorrect_info = {
                'autocorrected': False,
                'original_word': '',
                'corrected_word': '',
                'message': last_autocorrect_message
            }

            loop_start = time.time()
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.03)
                    continue
                frame = latest_frame.copy()
            frame = cv2.flip(frame, 1)
            processing_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
            rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)

            local_hands = hands_detector
            if local_hands is None:
                time.sleep(0.05)
                continue

            results = local_hands.process(rgb_frame)

            current_time = time.time()
            detected_letter = ""
            hand_detected = False

            letter_progress = 0.0
            space_progress = 0.0
            asl_space_this_frame = False

            if results and getattr(results, 'multi_hand_landmarks', None):
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    if DRAW_LANDMARKS:
                        mp_drawing.draw_landmarks(
                            processing_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style()
                        )
                    landmark_list = calc_landmark_list(processing_frame, hand_landmarks)
                    pre_processed_landmarks = pre_process_landmark(landmark_list)

                    if model is not None:
                        try:
                            input_arr = np.asarray([pre_processed_landmarks], dtype=np.float32)
                            predictions = model.predict(input_arr, verbose=0)
                            predicted_classes = np.argmax(predictions, axis=1)
                            detected_letter = alphabet[int(predicted_classes[0])] if len(alphabet) > int(predicted_classes[0]) else ''
                        except Exception as pe:
                            print(f"Model prediction failed: {pe}")
                            detected_letter = ''

                    if ASL_SPACE_ENABLED and current_time >= space_gesture_cooldown_until:
                        try:
                            if detect_asl_space_gesture(landmark_list):
                                asl_space_this_frame = True
                                if not space_gesture_active:
                                    space_gesture_active = True
                                    space_gesture_start_time = current_time
                                space_progress = min((current_time - space_gesture_start_time) / ASL_SPACE_HOLD_TIME, 1.0)
                                if (current_time - space_gesture_start_time) >= ASL_SPACE_HOLD_TIME:
                                    original_word = current_word
                                    autocorrected_flag = False
                                    corrected_word = ""
                                    message = ""
                                    if current_word:
                                        # Use fast local autocorrect only (skip network calls)
                                        try:
                                            if SPELLCHECKER_AVAILABLE and _spell is not None:
                                                cand = _spell.correction(current_word.lower())
                                                if cand and cand.lower() != current_word.lower():
                                                    corrected_word = _preserve_case(cand, current_word)
                                                else:
                                                    corrected_word = current_word
                                            else:
                                                corrected_word = current_word
                                        except Exception:
                                            corrected_word = current_word
                                        
                                        last_autocorrected_word = corrected_word
                                        current_sentence += corrected_word + " "
                                        current_word = ""
                                        autocorrected_flag = (corrected_word != original_word)
                                        if autocorrected_flag:
                                            message = f"Autocorrected '{original_word}' -> '{corrected_word}'"
                                    else:
                                        if not current_sentence.endswith(" "):
                                            current_sentence += " "
                                    last_autocorrect_message = message
                                    _last_autocorrect_info = {
                                        'autocorrected': autocorrected_flag,
                                        'original_word': original_word,
                                        'corrected_word': corrected_word,
                                        'message': message
                                    }

                                    last_prediction = ""
                                    prediction_start_time = current_time
                                    last_prediction_time = current_time
                                    space_gesture_active = False
                                    space_gesture_start_time = 0.0
                                    space_gesture_cooldown_until = current_time + 1.5
                                    detected_letter = "SPACE(H)"
                            else:
                                space_gesture_active = False
                                space_gesture_start_time = 0.0
                        except Exception:
                            space_gesture_active = False
                            space_gesture_start_time = 0.0

                    if not asl_space_this_frame and detected_letter:
                        if detected_letter == last_prediction:
                            if current_time - prediction_start_time >= HOLD_TIME:
                                if current_time - last_prediction_time >= HOLD_TIME:
                                    current_word += detected_letter
                                    last_prediction_time = current_time
                                    prediction_start_time = current_time
                        else:
                            last_prediction = detected_letter
                            prediction_start_time = current_time

            if not hand_detected:
                last_prediction = ""
                prediction_start_time = current_time

            if hand_detected and detected_letter == last_prediction and not asl_space_this_frame:
                letter_progress = min((current_time - prediction_start_time) / HOLD_TIME, 1.0)

            with jpeg_lock:
                try:
                    ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if ok:
                        latest_jpeg_bytes = buf.tobytes()
                except Exception:
                    pass

            with result_lock:
                processed_result = {
                    'status': 'success',
                    'detected_letter': detected_letter,
                    'current_word': current_word,
                    'current_sentence': current_sentence,
                    'letter_progress': letter_progress,
                    'space_progress': space_progress,
                    'hand_detected': hand_detected,
                    'last_autocorrected_word': last_autocorrected_word,
                    'timestamp': current_time,
                    'autocorrected': _last_autocorrect_info.get('autocorrected', False),
                    'original_word': _last_autocorrect_info.get('original_word', ''),
                    'corrected_word': _last_autocorrect_info.get('corrected_word', ''),
                    'autocorrect_message': _last_autocorrect_info.get('message', '')
                }

            sleep_for = MIN_PROCESS_INTERVAL - (time.time() - loop_start)
            if sleep_for > 0:
                time.sleep(sleep_for)
        except Exception as e:
            print(f"Error in process_frames: {e}")
            traceback.print_exc()
            time.sleep(0.1)

# ===== CAMERA AND MEDIAPIPE INITIALIZATION =====

def init_camera():
    """Initialize camera, MediaPipe detector, and processing threads.
    
    Sets up the video capture device, configures MediaPipe Hands detector,
    and starts the capture and processing worker threads.
    
    Returns:
        bool: True if initialization successful, False otherwise.
    
    Global Variables Modified:
        camera: Video capture object.
        hands_detector: MediaPipe Hands detector instance.
        capture_thread: Frame capture thread.
        processing_thread: Frame processing thread.
        is_capturing: Capture thread flag.
        is_processing: Processing thread flag.
    """
    global camera, hands_detector, capture_thread, processing_thread, is_capturing, is_processing
    try:
        if camera is None:
            print("Initializing camera...")
            camera = cv2.VideoCapture(0, cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if IS_WINDOWS:
                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            if not camera.isOpened():
                print("Failed to open camera")
                camera = None
                return False
            print("Camera initialized successfully")

        if hands_detector is None:
            print("Initializing MediaPipe hands detector...")
            hands_detector = mp_hands.Hands(
                static_image_mode=False,
                model_complexity=0,
                max_num_hands=1,
                min_detection_confidence=0.7 if IS_WINDOWS else 0.5,
                min_tracking_confidence=0.7 if IS_WINDOWS else 0.5
            )
            print("MediaPipe hands detector initialized successfully")

        if capture_thread is None or not capture_thread.is_alive():
            is_capturing = True
            capture_thread = threading.Thread(target=capture_frames, daemon=True)
            capture_thread.start()
            print("Capture thread started")

        if processing_thread is None or not processing_thread.is_alive():
            is_processing = True
            processing_thread = threading.Thread(target=process_frames, daemon=True)
            processing_thread.start()
            print("Processing thread started")

        return camera is not None and camera.isOpened()
    except Exception as e:
        print(f"Error initializing camera: {e}")
        traceback.print_exc()
        return False

# ===== LANDMARK PROCESSING FUNCTIONS =====

def calc_landmark_list(image, landmarks):
    """Convert MediaPipe landmarks to pixel coordinates.
    
    Transforms normalized landmark coordinates (0-1 range) to actual
    pixel coordinates bounded by image dimensions.
    
    Args:
        image: Input image (numpy array).
        landmarks: MediaPipe hand landmarks object.
    
    Returns:
        list: List of 21 [x, y] pixel coordinate pairs.
    """
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, lm in enumerate(landmarks.landmark):
        landmark_x = min(int(lm.x * image_width), image_width - 1)
        landmark_y = min(int(lm.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    """Normalize hand landmarks for model input.
    
    Performs coordinate normalization:
    1. Translates coordinates relative to wrist (index 0)
    2. Flattens 2D coordinates to 1D array
    3. Normalizes by maximum absolute value
    
    Args:
        landmark_list: List of 21 [x, y] landmark coordinates.
    
    Returns:
        list: Flattened and normalized coordinate array (42 values).
    """
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 1
    if max_value > 0:
        temp_landmark_list = [n / max_value for n in temp_landmark_list]
    return temp_landmark_list

# ===== FILE I/O FUNCTIONS =====

def save_sentence_to_file(sentence, filename=None):
    """Save recognized sentence to a text file.
    
    Saves the sentence with timestamp to the saved_sentences directory.
    Creates the directory if it doesn't exist.
    
    Args:
        sentence: The sentence text to save.
        filename: Optional custom filename (without extension).
    
    Returns:
        bool: True if save successful, False otherwise.
    """
    if not sentence.strip():
        return False
    if not os.path.exists("saved_sentences"):
        os.makedirs("saved_sentences")
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saved_sentences/sentence_{timestamp}.txt"
    else:
        filename = f"saved_sentences/{filename}.txt"
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"ISL Sentence - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write("="*50 + "\n")
            file.write(sentence.strip() + "\n")
        return True
    except Exception as e:
        print(f"Error saving file: {e}")
        return False

# ===== FLASK ROUTE HANDLERS =====

@app.route('/autocorrect_word', methods=['POST'])
def autocorrect_word_route():
    """Test endpoint for autocorrect functionality.
    
    Accepts a word via POST request and returns the autocorrected version.
    
    Request JSON:
        word (str): Word to autocorrect.
    
    Returns:
        JSON: {'status': 'success', 'original': str, 'corrected': str}
    """
    data = request.get_json() or {}
    word = data.get('word', '')
    corrected_word = get_autocorrect_suggestion(word)
    return jsonify({'status': 'success', 'original': word, 'corrected': corrected_word})

@app.route('/')
def index():
    """Render landing page.
    
    Returns:
        HTML: Rendered index.html template.
    """
    return render_template('index.html')

@app.route('/detector')
def detector():
    """Render gesture detector page.
    
    Returns:
        HTML: Rendered detector.html template with real-time detection interface.
    """
    return render_template('detector.html')

@app.route('/about')
def about():
    """Render about page.
    
    Returns:
        HTML: Rendered about.html template with project information.
    """
    return render_template('about.html')

@app.route('/voice_to_sign')
def voice_to_sign():
    """Render voice-to-sign conversion page.
    
    Returns:
        HTML: Rendered voice_to_sign.html template with speech recognition interface.
    """
    return render_template('voice_to_sign.html')

@app.route('/api/text_to_signs', methods=['POST'])
def text_to_signs():
    """Convert text to sign language image sequence.
    
    Accepts text input and returns a sequence of sign language images
    corresponding to each letter in the text.
    
    Request JSON:
        text (str): Text to convert to signs.
    
    Returns:
        JSON: {
            'status': 'success'|'error',
            'signs': [{'char': str, 'image': str, 'type': 'letter'|'space'}],
            'message': str (optional)
        }
    """
    try:
        data = request.get_json() or {}
        text = data.get('text', '').upper()
        
        if not text.strip():
            return jsonify({'status': 'error', 'message': 'No text provided'})
        
        sign_sequence = []
        for char in text:
            if char.isalpha() and char in alphabet:
                # Map to your sign images (A-Z)
                sign_sequence.append({
                    'char': char,
                    'image': f'/static/Images/{char}.jpg',
                    'type': 'letter'
                })
            elif char == ' ':
                # Add space indicator
                sign_sequence.append({
                    'char': 'SPACE',
                    'image': '',
                    'type': 'space'
                })
            elif char.isdigit() and char in alphabet:
                # Numbers if available
                sign_sequence.append({
                    'char': char,
                    'image': f'/static/Images/{char}.jpg',
                    'type': 'number'
                })
        
        return jsonify({
            'status': 'success',
            'signs': sign_sequence,
            'message': f'Converted {len([s for s in sign_sequence if s["type"] == "letter"])} letters'
        })
    except Exception as e:
        print(f"Error in text_to_signs: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/test')
def test():
    """Health check endpoint.
    
    Returns:
        JSON: {'status': 'success', 'message': str}
    """
    return jsonify({'status': 'success', 'message': 'Server is responding'})

@app.route('/start_detection')
def start_detection():
    """Initialize and start gesture detection.
    
    Initializes camera, MediaPipe detector, and processing threads.
    
    Returns:
        JSON: {'status': 'success'|'error', 'message': str}
    """
    try:
        print("Start detection route called")
        if init_camera():
            print("Camera initialization successful")
            return jsonify({'status': 'success', 'message': 'Camera initialized'})
        else:
            print("Camera initialization failed")
            return jsonify({'status': 'error', 'message': 'Failed to initialize camera. Please check if camera is available and not being used by another application.'})
    except Exception as e:
        print(f"Error in start_detection route: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': f'Camera initialization error: {str(e)}'})

@app.route('/get_detection_meta')
def get_detection_meta():
    """Get detection metadata without video frame.
    
    Returns the latest detection results including recognized letter,
    current word/sentence, progress indicators, and autocorrect information.
    
    Returns:
        JSON: Detection metadata dictionary or error message.
    """
    global processed_result
    try:
        if camera is None or hands_detector is None:
            return jsonify({'status': 'error', 'message': 'Camera not initialized'})
        with result_lock:
            if processed_result is None:
                return jsonify({'status': 'error', 'message': 'No frame processed yet'})
            return jsonify(processed_result.copy())
    except Exception as e:
        print(f"Error in get_detection_meta: {e}")
        return jsonify({'status': 'error', 'message': f'Detection error: {str(e)}'})


def _mjpeg_generator():
    """Generate MJPEG stream frames.
    
    Generator function that yields JPEG frames with multipart boundaries
    for MJPEG video streaming over HTTP.
    
    Yields:
        bytes: MJPEG frame data with HTTP multipart headers.
    """
    global latest_jpeg_bytes
    while is_capturing or is_processing:
        frame_bytes = None
        with jpeg_lock:
            if latest_jpeg_bytes:
                frame_bytes = latest_jpeg_bytes
        if frame_bytes:
            header = (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(frame_bytes)).encode('ascii') + b'\r\n\r\n'
            )
            yield header + frame_bytes + b'\r\n'
            time.sleep(0.03)
        else:
            time.sleep(0.05)

@app.route('/video_stream')
def video_stream():
    """Provide live video stream endpoint.
    
    Streams processed video frames with hand landmarks as MJPEG.
    
    Returns:
        Response: MJPEG stream or 503 if camera not initialized.
    """
    if camera is None or hands_detector is None:
        return Response(status=503)
    return Response(_mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_detection')
def get_detection():
    """Get detection results with optional frame data.
    
    Returns detection metadata along with base64-encoded current frame
    for polling-based clients.
    
    Returns:
        JSON: Detection metadata with optional 'frame' field containing base64 image.
    """
    global processed_result, latest_jpeg_bytes
    try:
        if camera is None or hands_detector is None:
            return jsonify({'status': 'error', 'message': 'Camera not initialized'})
        with result_lock:
            if processed_result is None:
                return jsonify({'status': 'error', 'message': 'No frame processed yet'})
            payload = processed_result.copy()
        with jpeg_lock:
            if latest_jpeg_bytes:
                payload['frame'] = base64.b64encode(latest_jpeg_bytes).decode('utf-8')
        return jsonify(payload)
    except Exception as e:
        print(f"Error in get_detection: {e}")
        return jsonify({'status': 'error', 'message': f'Detection error: {str(e)}'})

@app.route('/save_sentence', methods=['POST'])
def save_sentence():
    """Save current sentence to file.
    
    Saves the accumulated sentence (including current incomplete word)
    to a timestamped text file.
    
    Returns:
        JSON: {'status': 'success'|'error', 'message': str}
    """
    global current_sentence, current_word
    final_sentence = current_sentence
    if current_word:
        final_sentence += current_word
    if final_sentence.strip():
        if save_sentence_to_file(final_sentence):
            return jsonify({'status': 'success', 'message': 'Sentence saved successfully!'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to save sentence'})
    else:
        return jsonify({'status': 'error', 'message': 'No sentence to save'})

@app.route('/backspace_word', methods=['POST'])
def backspace_word():
    """Remove last character from current word.
    
    Deletes the last character in the current word buffer.
    
    Returns:
        JSON: {'status': 'success', 'message': 'Character deleted', 'word': current_word}
    """
    global current_word
    if current_word:
        current_word = current_word[:-1]
        return jsonify({'status': 'success', 'message': 'Character deleted', 'word': current_word})
    else:
        return jsonify({'status': 'error', 'message': 'No characters to delete'})

@app.route('/reset_word', methods=['POST'])
def reset_word():
    """Clear current word buffer.
    
    Resets the in-progress word being built, keeping the sentence intact.
    
    Returns:
        JSON: {'status': 'success', 'message': 'Word reset'}
    """
    global current_word
    current_word = ""
    return jsonify({'status': 'success', 'message': 'Word reset'})

@app.route('/reset_sentence', methods=['POST'])
def reset_sentence():
    """Clear entire sentence and word buffers.
    
    Resets both sentence and word, along with autocorrect metadata.
    
    Returns:
        JSON: {'status': 'success', 'message': 'Sentence reset'}
    """
    global current_sentence, current_word, last_autocorrected_word, last_autocorrect_message
    current_sentence = ""
    current_word = ""
    last_autocorrected_word = ""
    last_autocorrect_message = ""
    return jsonify({'status': 'success', 'message': 'Sentence reset'})

@app.route('/stop_detection')
def stop_detection():
    """Stop detection and release resources.
    
    Stops processing threads, releases camera, closes MediaPipe detector,
    and clears all transient state.
    
    Returns:
        JSON: {'status': 'success', 'message': 'Detection stopped'}
    """
    global camera, hands_detector, is_capturing, is_processing
    global capture_thread, processing_thread, latest_frame, processed_result, latest_jpeg_bytes
    global last_autocorrected_word, last_autocorrect_message

    is_capturing = False
    is_processing = False

    if capture_thread is not None:
        capture_thread.join(timeout=2)
    if processing_thread is not None:
        processing_thread.join(timeout=2)

    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
        camera = None

    if hands_detector is not None:
        try:
            hands_detector.close()
        except Exception:
            pass
        hands_detector = None

    with frame_lock:
        latest_frame = None

    with result_lock:
        processed_result = None

    with jpeg_lock:
        latest_jpeg_bytes = None

    last_autocorrected_word = ""
    last_autocorrect_message = ""

    return jsonify({'status': 'success', 'message': 'Detection stopped'})

if __name__ == '__main__':
    print("Open http://localhost:5001 in your browser to use the detector.")
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
