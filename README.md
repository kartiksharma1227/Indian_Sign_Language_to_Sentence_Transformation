# Indian Sign Language (ISL) Translator

## Project Overview

This is a comprehensive Indian Sign Language (ISL) translation web application built with Flask, OpenCV, MediaPipe, and TensorFlow. The application provides **bidirectional communication support**:

1. **Sign to Text**: Real-time sign language gesture recognition using webcam, automatically translating ISL gestures into text
2. **Voice/Text to Sign**: Convert spoken words or typed text into visual sign language sequences for learning and communication

This dual-functionality makes the application useful for both hearing-impaired individuals to communicate with others and for people to learn and understand sign language.

## Screenshots

### Home Page

![Home Page](Demo/home_page.png)
_Landing page with navigation to different features_

### Sign to Text Detection

![Sign Detection](Demo/detector_page.png)
_Real-time sign language gesture recognition interface_

### Voice to Sign Conversion

![Voice to Sign](Demo/voice_to_sign_page.png)
_Speech recognition and sign language display interface_

## Features

### Sign to Text Translation

- **Real-time Detection**: Instant sign language gesture recognition using webcam
- **Word Building**: Automatically builds words from detected letters
- **Sentence Formation**: Intelligently creates sentences with automatic spacing
- **High Accuracy**: 95%+ accuracy rate with 43 supported signs (numbers 1-9 and letters A-Z)
- **Save Functionality**: Export translated sentences to text files

### Voice to Sign Translation

- **Speech Recognition**: Convert spoken words into sign language sequences
- **Manual Text Input**: Type text to convert into sign language
- **Visual Display**: Shows corresponding sign images for each letter
- **Playback Controls**: Play, pause, stop, and replay sign sequences
- **Adjustable Speed**: Control playback speed for better learning
- **Export Options**: Save sign sequences and translations

### General Features

- **Cross-platform**: Works on Windows, macOS, and Linux with optimized performance
- **Bidirectional Communication**: Supports both sign-to-text and voice/text-to-sign conversion

## Technology Stack

### Backend

- **Flask**: Python web framework for API and routing
- **TensorFlow/Keras**: Deep learning model for gesture classification
- **MediaPipe**: Hand tracking and landmark detection
- **OpenCV**: Computer vision and video processing
- **NumPy**: Numerical computations

### Frontend

- **HTML/CSS**: Responsive UI design
- **JavaScript**: Interactive client-side functionality
- **Jinja2**: Template engine for dynamic content

## Project Structure

```
IVP/
├── app.py                  # Main Flask application
├── model.h5                # Pre-trained Keras model for gesture classification
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── templates/             # HTML templates
│   ├── base.html          # Base template with navigation
│   ├── index.html         # Home page
│   ├── detector.html      # Main detector interface
│   ├── voice_to_sign.html # Voice/text to sign converter
│   └── about.html         # About page
├── static/                # Static assets
│   ├── css/              # Stylesheets
│   │   ├── base.css
│   │   ├── index.css
│   │   ├── detector.css
│   │   ├── voice_to_sign.css
│   │   └── about.css
│   ├── Images/           # Sign language images (A-Z)
│   └── js/               # JavaScript files
│       ├── main.js       # Common functionality
│       ├── detector.js   # Detector page logic
│       └── voice_to_sign.js  # Voice to sign conversion logic
├── saved_sentences/       # Directory for saved sentence files
└── Demo/                  # Demo images/videos
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- pip package manager

### Setup Instructions

1. **Clone or download the project**

   ```bash
   cd /path/to/project
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv isl
   ```

3. **Activate the virtual environment**

   - On macOS/Linux:

     ```bash
     source isl/bin/activate
     ```

   - On Windows:
     ```bash
     isl\Scripts\activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**

   ```bash
   python app.py
   ```

6. **Access the application**
   - Open your browser and navigate to: `http://localhost:5001`

## Usage Guide

### Sign to Text Detection

#### Starting Detection

1. Navigate to the **Detector** page
2. Click the **"Start Detection"** button or press **Enter**
3. Allow camera permissions when prompted
4. Position your hand clearly in front of the camera

#### Making Gestures

1. **Form a sign** with your hand (A-Z or 1-9)
2. **Hold the gesture steady** for 1.5 seconds
3. The letter will be added to your current word
4. **Use the ASL H symbol** to add a space between words

#### Keyboard Shortcuts

- **Enter**: Start/Stop detection
- **Space**: Save sentence to file
- **R**: Reset current word
- **S**: Reset entire sentence

#### Controls

- **Reset Word**: Clear the current word being built
- **Reset Sentence**: Clear the entire sentence
- **Save Sentence**: Export the sentence to a timestamped text file

### Voice/Text to Sign Conversion

#### Using Voice Input

1. Navigate to the **Voice to Sign** page
2. Click the **microphone button** to start recording
3. Speak clearly into your device's microphone
4. The recognized text will appear automatically
5. Click **"Convert to Signs"** to generate the sign sequence

#### Using Manual Text Input

1. Navigate to the **Voice to Sign** page
2. Type your text in the input field
3. Click **"Convert to Signs"** to generate the sign sequence

#### Playback Controls

- **Play**: Start displaying the sign sequence
- **Pause**: Temporarily pause the playback
- **Stop**: Stop and reset the playback
- **Replay**: Start the sequence from the beginning
- **Speed Control**: Adjust playback speed (0.5x to 2x)

#### Additional Features

- **Progress Bar**: Visual indicator of playback progress
- **Statistics**: View letter count, word count, and estimated duration
- **Export**: Save the sign sequence for later reference
- **Clear**: Reset all inputs and outputs

## How It Works

### Detection Pipeline

1. **Hand Detection**: MediaPipe identifies hand landmarks (21 key points)
2. **Feature Extraction**: Landmarks are normalized and converted to feature vectors
3. **Classification**: Neural network classifies the gesture into one of 43 signs
4. **Word Building**: Detected letters are combined using timing logic

### Architecture Components

#### Backend Processing

- **Capture Thread**: Continuously grabs frames from camera
- **Processing Thread**: Processes frames for hand detection and classification
- **Threading Synchronization**: Uses locks to safely share data between threads

#### Detection Logic

- **Hold Time**: 1.5 seconds to confirm letter addition
- **Space Gesture**: ASL H symbol to add space between words
- **Frame Rate Control**: Optimized processing intervals for performance

## Model Information

The application uses a pre-trained neural network (`model.h5`) that:

- Accepts normalized hand landmark coordinates (42 values)
- Outputs probabilities for 43 classes (1-9, A-Z)
- Trained on thousands of sign language gesture samples
- Achieves 95%+ accuracy on test data

## Performance Optimization

### OS-Specific Settings

- **Windows**: More aggressive optimization for better performance

  - Lower resolution processing
  - Frame skipping
  - Disabled landmark drawing
  - DirectShow camera backend

- **macOS/Linux**: Balanced performance and quality
  - Higher resolution processing
  - Full frame processing
  - Landmark visualization enabled

### Configuration Variables

All performance settings can be adjusted in `app.py`:

- `FRAME_WIDTH`, `FRAME_HEIGHT`: Camera capture resolution
- `PROCESSING_WIDTH`, `PROCESSING_HEIGHT`: Processing frame size
- `JPEG_QUALITY`: Video stream compression quality
- `PROCESS_EVERY_N_FRAMES`: Frame processing frequency

## API Endpoints

### GET Routes

- `/` - Home page
- `/detector` - Detector interface page (Sign to Text)
- `/voice_to_sign` - Voice/Text to Sign converter page
- `/about` - About page
- `/start_detection` - Initialize camera and start detection
- `/stop_detection` - Stop detection and release resources
- `/get_detection_meta` - Get detection results (without video frame)
- `/get_detection` - Get detection results with video frame
- `/video_stream` - Motion JPEG video stream

### POST Routes

- `/save_sentence` - Save current sentence to file
- `/reset_word` - Reset current word
- `/reset_sentence` - Reset entire sentence
- `/api/text_to_signs` - Convert text input to sign language image sequence

## File Saving

Sentences are automatically saved to the `saved_sentences/` directory with:

- Timestamp in filename: `sentence_YYYYMMDD_HHMMSS.txt`
- Formatted content with date and separator
- UTF-8 encoding

## Troubleshooting

### Camera Issues

- **Camera not detected**: Ensure no other application is using the camera
- **Permission denied**: Grant camera permissions in browser/OS settings
- **Poor performance**: Reduce `FRAME_WIDTH` and `FRAME_HEIGHT` in `app.py`

### Detection Issues

- **Low accuracy**: Ensure good lighting and clear hand visibility
- **Slow response**: Increase `PROCESS_EVERY_N_FRAMES` for faster processing
- **Letters not adding**: Hold gesture steady for full 1.5 seconds

### Performance Issues

- **High CPU usage**: Enable Windows optimizations (set `IS_WINDOWS = True`)
- **Laggy video**: Reduce `JPEG_QUALITY` or `FRAME_WIDTH`/`FRAME_HEIGHT`
- **Memory issues**: Restart the application periodically

## Browser Compatibility

### Sign Detection (Camera)

- **Chrome/Edge**: Recommended (best performance)
- **Firefox**: Fully supported
- **Safari**: Supported with minor limitations
 

### Voice Recognition

- **Chrome/Edge**: Full support for Web Speech API
- **Safari**: Supported on macOS and iOS
- **Firefox**: Limited support (manual text input recommended)

---

**Note**: This application requires a webcam and runs best on desktop/laptop computers with good lighting conditions.
