/**
 * Voice to Sign Language Conversion
 * Speech Recognition and Sign Display Logic
 */

// ===== STATE MANAGEMENT =====
let recognition = null;
let isRecording = false;
let currentSignSequence = [];
let currentSignIndex = 0;
let isPlaying = false;
let isPaused = false;
let playbackSpeed = 1;
let playbackInterval = null;

// ===== DOM ELEMENTS =====
const micButton = document.getElementById("micButton");
const micStatus = document.getElementById("micStatus");
const recognitionStatus = document.getElementById("recognitionStatus");
const recognizedText = document.getElementById("recognizedText");
const manualText = document.getElementById("manualText");
const convertBtn = document.getElementById("convertBtn");
const clearBtn = document.getElementById("clearBtn");
const exportBtn = document.getElementById("exportBtn");
const signImageContainer = document.getElementById("signImageContainer");
const currentLetter = document.getElementById("currentLetter");
const playbackControls = document.getElementById("playbackControls");
const playBtn = document.getElementById("playBtn");
const pauseBtn = document.getElementById("pauseBtn");
const stopBtn = document.getElementById("stopBtn");
const replayBtn = document.getElementById("replayBtn");
const progressContainer = document.getElementById("progressContainer");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const signSequence = document.getElementById("signSequence");
const letterCount = document.getElementById("letterCount");
const wordCount = document.getElementById("wordCount");
const durationEstimate = document.getElementById("durationEstimate");
const signCount = document.getElementById("signCount");
const notification = document.getElementById("notification");
const notificationText = document.getElementById("notificationText");

// ===== SPEECH RECOGNITION SETUP =====
function initSpeechRecognition() {
  // Check for browser support
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    showNotification(
      "Speech recognition not supported in this browser. Please use Chrome, Edge, or Safari.",
      "error"
    );
    micButton.disabled = true;
    return false;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onstart = () => {
    console.log("Speech recognition started");
    updateStatus("Listening... Speak now", "success");
  };

  recognition.onresult = (event) => {
    let interimTranscript = "";
    let finalTranscript = "";

    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript + " ";
      } else {
        interimTranscript += transcript;
      }
    }

    // Update display
    if (finalTranscript) {
      const currentText =
        recognizedText.textContent === "Your speech will appear here..."
          ? ""
          : recognizedText.textContent;
      recognizedText.textContent = (currentText + finalTranscript).trim();

      // Auto-convert to signs
      convertTextToSigns(recognizedText.textContent);
    }
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    let errorMessage = "Recognition error: ";
    switch (event.error) {
      case "no-speech":
        errorMessage += "No speech detected. Please try again.";
        // Don't stop for no-speech, just continue
        return;
      case "audio-capture":
        errorMessage += "No microphone found. Please check your device.";
        break;
      case "not-allowed":
        errorMessage += "Microphone permission denied. Please allow access.";
        break;
      case "aborted":
        // User stopped it, don't show error
        return;
      default:
        errorMessage += event.error;
    }
    updateStatus(errorMessage, "error");
    stopRecording();
  };

  recognition.onend = () => {
    console.log("Speech recognition ended, isRecording:", isRecording);
    if (isRecording) {
      // Restart if still in recording mode
      try {
        setTimeout(() => {
          if (isRecording && recognition) {
            recognition.start();
          }
        }, 100);
      } catch (error) {
        console.error("Error restarting recognition:", error);
        stopRecording();
      }
    } else {
      updateStatus("Ready to listen", "");
    }
  };

  return true;
}

// ===== RECORDING CONTROLS =====
let lastToggleTime = 0;
function toggleRecording() {
  // Prevent rapid toggling
  const now = Date.now();
  if (now - lastToggleTime < 500) {
    console.log("Toggle too fast, ignoring");
    return;
  }
  lastToggleTime = now;

  console.log("Toggle recording, current state:", isRecording);
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

function startRecording() {
  if (!recognition) {
    if (!initSpeechRecognition()) {
      return;
    }
  }

  try {
    isRecording = true;
    micButton.classList.add("recording");
    micStatus.textContent = "Recording...";
    recognizedText.textContent = "";
    recognition.start();
    console.log("Started recording");
  } catch (error) {
    console.error("Error starting recognition:", error);
    // If already started, that's ok
    if (error.message && error.message.includes("already started")) {
      console.log("Recognition already started, continuing...");
      isRecording = true;
      micButton.classList.add("recording");
      micStatus.textContent = "Recording...";
    } else {
      showNotification("Failed to start recording. Please try again.", "error");
      isRecording = false;
      micButton.classList.remove("recording");
      micStatus.textContent = "Click to start";
    }
  }
}

function stopRecording() {
  console.log("Stopping recording");
  isRecording = false;
  micButton.classList.remove("recording");
  micStatus.textContent = "Click to start";

  if (recognition) {
    try {
      recognition.stop();
      updateStatus("Ready to listen", "");
    } catch (error) {
      console.error("Error stopping recognition:", error);
    }
  }
}

// ===== TEXT TO SIGNS CONVERSION =====
async function convertTextToSigns(text) {
  if (!text || !text.trim()) {
    showNotification("No text to convert", "error");
    return;
  }

  try {
    const response = await fetch("/api/text_to_signs", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text: text.trim() }),
    });

    const data = await response.json();

    if (data.status === "success") {
      currentSignSequence = data.signs;
      displaySignSequence();
      updateStatistics();
      exportBtn.disabled = false;
      showNotification(data.message || "Conversion successful!", "success");
    } else {
      showNotification(data.message || "Conversion failed", "error");
    }
  } catch (error) {
    console.error("Conversion error:", error);
    // showNotification("Failed to convert text to signs", "error");
  }
}

// ===== SIGN DISPLAY =====
function displaySignSequence() {
  // Clear sequence display
  signSequence.innerHTML = "";

  // Add thumbnails
  currentSignSequence.forEach((sign, index) => {
    const thumb = document.createElement("div");
    thumb.className = "vts-sign-thumb";

    if (sign.type === "space") {
      thumb.classList.add("space");
      thumb.textContent = "SPACE";
    } else {
      const img = document.createElement("img");
      img.src = sign.image;
      img.alt = sign.char;
      thumb.appendChild(img);
    }

    thumb.addEventListener("click", () => {
      displaySignAtIndex(index);
    });

    signSequence.appendChild(thumb);
  });

  // Show controls
  playbackControls.style.display = "flex";
  progressContainer.style.display = "block";

  // Auto-start playback
  if (currentSignSequence.length > 0) {
    startPlayback();
  }
}

function displaySignAtIndex(index) {
  if (index < 0 || index >= currentSignSequence.length) {
    return;
  }

  currentSignIndex = index;
  const sign = currentSignSequence[index];

  // Update image container
  if (sign.type === "space") {
    signImageContainer.innerHTML = `
      <div class="vts-placeholder">
        <i class="fas fa-space-bar"></i>
        <p>SPACE</p>
      </div>
    `;
    currentLetter.innerHTML = "<span>SPACE</span>";
  } else {
    signImageContainer.innerHTML = `<img src="${sign.image}" alt="${sign.char}">`;
    currentLetter.innerHTML = `<span>${sign.char}</span>`;
  }

  // Update progress
  updateProgress();

  // Highlight thumbnail
  const thumbs = signSequence.querySelectorAll(".vts-sign-thumb");
  thumbs.forEach((thumb, i) => {
    thumb.classList.toggle("active", i === index);
  });
}

// ===== PLAYBACK CONTROLS =====
function startPlayback() {
  if (currentSignSequence.length === 0) {
    showNotification("No signs to play", "error");
    return;
  }

  isPlaying = true;
  isPaused = false;
  playBtn.style.display = "none";
  pauseBtn.style.display = "inline-flex";

  if (currentSignIndex >= currentSignSequence.length) {
    currentSignIndex = 0;
  }

  playNext();
}

function playNext() {
  if (!isPlaying || isPaused) {
    return;
  }

  if (currentSignIndex < currentSignSequence.length) {
    displaySignAtIndex(currentSignIndex);

    // Calculate delay based on sign type and speed
    const sign = currentSignSequence[currentSignIndex];
    const baseDelay = sign.type === "space" ? 500 : 1500;
    const delay = baseDelay / playbackSpeed;

    currentSignIndex++;

    playbackInterval = setTimeout(() => {
      playNext();
    }, delay);
  } else {
    // Playback complete
    stopPlayback();
    showNotification("Playback complete!", "success");
  }
}

function pausePlayback() {
  isPaused = true;
  isPlaying = false;
  playBtn.style.display = "inline-flex";
  pauseBtn.style.display = "none";

  if (playbackInterval) {
    clearTimeout(playbackInterval);
  }
}

function stopPlayback() {
  isPlaying = false;
  isPaused = false;
  playBtn.style.display = "inline-flex";
  pauseBtn.style.display = "none";
  currentSignIndex = 0;

  if (playbackInterval) {
    clearTimeout(playbackInterval);
  }

  // Reset to first sign
  if (currentSignSequence.length > 0) {
    displaySignAtIndex(0);
  }
}

function replayPlayback() {
  stopPlayback();
  startPlayback();
}

// ===== PROGRESS UPDATE =====
function updateProgress() {
  const progress =
    currentSignSequence.length > 0
      ? (currentSignIndex / currentSignSequence.length) * 100
      : 0;

  progressBar.style.width = `${progress}%`;
  progressText.textContent = `${currentSignIndex} / ${currentSignSequence.length}`;
}

// ===== STATISTICS =====
function updateStatistics() {
  const text = recognizedText.textContent || manualText.value || "";

  // Count letters (excluding spaces)
  const letters = text.replace(/[^a-zA-Z]/g, "");
  letterCount.textContent = letters.length;

  // Count words
  const words = text
    .trim()
    .split(/\s+/)
    .filter((w) => w.length > 0);
  wordCount.textContent = words.length;

  // Sign count
  signCount.textContent = currentSignSequence.length;

  // Estimate duration (1.5s per letter, 0.5s per space)
  const letterTime = letters.length * 1.5;
  const spaceTime = (text.match(/\s/g) || []).length * 0.5;
  const totalSeconds = (letterTime + spaceTime) / playbackSpeed;
  durationEstimate.textContent = `${Math.round(totalSeconds)}s`;
}

// ===== SPEED CONTROL =====
function setupSpeedControl() {
  const speedButtons = document.querySelectorAll(".vts-speed-btn");

  speedButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      speedButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      playbackSpeed = parseFloat(btn.dataset.speed);
      updateStatistics();
      showNotification(`Speed set to ${playbackSpeed}x`, "success");
    });
  });
}

// ===== UI HELPERS =====
function updateStatus(message, type = "") {
  recognitionStatus.innerHTML = `<i class="fas fa-info-circle"></i><span>${message}</span>`;
  recognitionStatus.className = "vts-status-box";
  if (type) {
    recognitionStatus.classList.add(type);
  }
}

function showNotification(message, type = "success") {
  notificationText.textContent = message;
  notification.className = `vts-notification ${type}`;
  notification.classList.add("show");

  // Update icon
  const icon = notification.querySelector("i");
  icon.className =
    type === "error" ? "fas fa-exclamation-circle" : "fas fa-check-circle";

  setTimeout(() => {
    notification.classList.remove("show");
  }, 3000);
}

function clearAll() {
  recognizedText.textContent = "Your speech will appear here...";
  manualText.value = "";
  currentSignSequence = [];
  currentSignIndex = 0;
  signSequence.innerHTML = "";
  signImageContainer.innerHTML = `
    <div class="vts-placeholder">
      <i class="fas fa-hands"></i>
      <p>Signs will appear here</p>
    </div>
  `;
  currentLetter.innerHTML = "<span>-</span>";
  playbackControls.style.display = "none";
  progressContainer.style.display = "none";
  exportBtn.disabled = true;
  updateStatistics();
  showNotification("Cleared all content", "success");
}

function exportSigns() {
  const text =
    recognizedText.textContent !== "Your speech will appear here..."
      ? recognizedText.textContent
      : manualText.value;

  if (!text.trim()) {
    showNotification("No text to export", "error");
    return;
  }

  // Create export data
  const exportData = {
    text: text,
    signs: currentSignSequence.length,
    timestamp: new Date().toISOString(),
    sequence: currentSignSequence.map((s) => s.char).join(""),
  };

  // Download as JSON
  const blob = new Blob([JSON.stringify(exportData, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `voice-to-sign-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);

  showNotification("Export successful!", "success");
}

// ===== EVENT LISTENERS =====
function setupEventListeners() {
  // Microphone toggle
  micButton.addEventListener("click", toggleRecording);

  // Manual conversion
  convertBtn.addEventListener("click", () => {
    const text = manualText.value.trim();
    if (text) {
      recognizedText.textContent = text;
      convertTextToSigns(text);
    } else {
      showNotification("Please enter some text", "error");
    }
  });

  // Clear button
  clearBtn.addEventListener("click", clearAll);

  // Export button
  exportBtn.addEventListener("click", exportSigns);

  // Playback controls
  playBtn.addEventListener("click", startPlayback);
  pauseBtn.addEventListener("click", pausePlayback);
  stopBtn.addEventListener("click", stopPlayback);
  replayBtn.addEventListener("click", replayPlayback);

  // Manual text input changes
  manualText.addEventListener("input", () => {
    updateStatistics();
  });

  // Keyboard shortcuts
  document.addEventListener("keydown", (e) => {
    // Space to play/pause
    if (e.code === "Space" && e.target.tagName !== "TEXTAREA") {
      e.preventDefault();
      if (isPlaying) {
        pausePlayback();
      } else if (currentSignSequence.length > 0) {
        startPlayback();
      }
    }

    // R to replay
    if (e.code === "KeyR" && e.ctrlKey) {
      e.preventDefault();
      replayPlayback();
    }

    // Escape to stop recording
    if (e.code === "Escape" && isRecording) {
      stopRecording();
    }
  });
}

// ===== INITIALIZATION =====
document.addEventListener("DOMContentLoaded", () => {
  console.log("Voice to Sign - Initializing...");

  // Setup
  setupEventListeners();
  setupSpeedControl();
  updateStatistics();

  // Check for speech recognition support
  const SpeechRecognition =
    window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    updateStatus(
      "Speech recognition not supported. Use manual text input.",
      "error"
    );
    micButton.disabled = true;
    micButton.style.opacity = "0.5";
    micButton.style.cursor = "not-allowed";
    showNotification(
      "Speech recognition not available. Please use Chrome, Edge, or Safari.",
      "error"
    );
  } else {
    updateStatus("Click microphone to start", "");
    console.log("Speech recognition available");
  }

  console.log("Voice to Sign - Ready!");
});

// ===== CLEANUP =====
window.addEventListener("beforeunload", () => {
  if (recognition && isRecording) {
    recognition.stop();
  }
  if (playbackInterval) {
    clearTimeout(playbackInterval);
  }
});
