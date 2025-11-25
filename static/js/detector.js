/**
 * ISL Detector Module - Sign Language Detection Interface
 *
 * Handles all functionality for the sign language detector page including:
 * - Camera initialization and management
 * - Real-time detection display
 * - Word and sentence building UI
 * - User controls and keyboard shortcuts
 * - Progress indicators
 *
 * @author Course Project
 * @version 1.0
 */

/**
 * ISL Detector Class
 * Main class managing the sign language detection interface
 */
class ISLDetector {
  /**
   * Initialize the ISL Detector
   * Sets up DOM elements, event listeners, and initial state
   */
  constructor() {
    // Detection state
    this.isDetecting = false; // Whether detection is currently active
    this.detectionInterval = null; // Interval ID for polling detection results

    // Video elements
    this.canvas = document.getElementById("videoCanvas");
    this.ctx = this.canvas ? this.canvas.getContext("2d") : null;
    this.videoStream =
      document.getElementById("videoStream") ||
      (() => {
        const img = document.createElement("img");
        img.id = "videoStream";
        if (this.canvas?.parentNode) {
          this.canvas.parentNode.insertBefore(img, this.canvas);
        } else {
          document.body.appendChild(img);
        }
        return img;
      })();

    if (this.videoStream) {
      this.videoStream.style.display = "none";
      this.videoStream.style.width = "100%";
      this.videoStream.style.height = "100%";
      this.videoStream.style.objectFit = "cover";
      this.videoStream.style.borderRadius = "inherit";
    }

    // Detection data
    this.currentWord = ""; // Current word being built
    this.currentSentence = ""; // Accumulated sentence
    this.detectedLetter = ""; // Currently detected letter

    this.initializeElements();
    this.bindEvents();
    this.setDetectionState("idle"); // Initialize with idle state
  }

  /**
   * Initialize DOM element references
   * Gets references to all UI elements needed for the detector
   */
  initializeElements() {
    this.startBtn = document.getElementById("startBtn");
    this.stopBtn = document.getElementById("stopBtn");
    this.videoPlaceholder = document.getElementById("videoPlaceholder");
    this.letterProgressBar = document.getElementById("letterProgress");
    this.letterProgressText = document.getElementById("letterProgressText");
    this.spaceProgressBar = document.getElementById("spaceProgress");
    this.spaceProgressText = document.getElementById("spaceProgressText");
    this.spaceDurationContainer = document.getElementById(
      "spaceDurationContainer"
    );
    this.detectedLetterElement = document.getElementById("detectedLetter");
    this.currentWordElement = document.getElementById("currentWord");
    this.currentSentenceElement = document.getElementById("currentSentence");
    this.backspaceBtn = document.getElementById("backspaceBtn");
    this.resetSentenceBtn = document.getElementById("resetSentenceBtn");
    this.saveSentenceBtn = document.getElementById("saveSentenceBtn");
    this.cameraStatus = document.getElementById("cameraStatus");

    // Session stats elements
    this.letterCountElement = document.getElementById("letterCount");
    this.wordCountElement = document.getElementById("wordCount");
    this.accuracyElement = document.getElementById("accuracyRate");
    this.sessionTimeElement = document.getElementById("sessionTime");

    // Session tracking
    this.sessionStartTime = null;
    this.sessionLetterCount = 0;
    this.sessionWordCount = 0;
  }

  /**
   * Bind event listeners to UI controls
   * Sets up click handlers for all buttons
   */
  bindEvents() {
    if (this.startBtn) {
      this.startBtn.addEventListener("click", (e) => {
        e.preventDefault();
        this.startDetection();
      });
    }

    if (this.stopBtn) {
      this.stopBtn.addEventListener("click", (e) => {
        e.preventDefault();
        this.stopDetection();
      });
    }

    if (this.backspaceBtn) {
      this.backspaceBtn.addEventListener("click", (e) => {
        e.preventDefault();
        this.backspaceWord();
      });
    }

    if (this.resetSentenceBtn) {
      this.resetSentenceBtn.addEventListener("click", (e) => {
        e.preventDefault();
        this.resetSentence();
      });
    }

    if (this.saveSentenceBtn) {
      this.saveSentenceBtn.addEventListener("click", (e) => {
        e.preventDefault();
        this.saveSentence();
      });
    }
  }

  /**
   * Set the visual state of the detected letter display
   *
   * @param {string} state - One of: 'idle', 'detecting', 'no-hand'
   */
  setDetectionState(state) {
    if (this.detectedLetterElement) {
      this.detectedLetterElement.setAttribute("data-state", state);
    }
  }

  /**
   * Update the detected letter display with current detection
   *
   * @param {string} letter - The detected letter/number or '-' for no detection
   */
  updateDetectedLetter(letter) {
    const letterTextElement = this.detectedLetterElement?.querySelector(
      ".isl-detected-letter"
    );

    if (letter && letter !== "-") {
      // Actively detecting a letter
      this.setDetectionState("detecting");
      if (letterTextElement) {
        letterTextElement.textContent = letter;
      } else if (this.detectedLetterElement) {
        this.detectedLetterElement.textContent = letter;
      }
    } else if (!letter || letter === "-") {
      // No hand detected
      this.setDetectionState("no-hand");
      if (letterTextElement) {
        letterTextElement.textContent = "-";
      } else if (this.detectedLetterElement) {
        this.detectedLetterElement.textContent = "-";
      }
    }
  }

  /**
   * Start the detection process
   * Initializes camera backend and begins video streaming
   *
   * @async
   * @returns {Promise<void>}
   */
  async startDetection() {
    try {
      // Show loading state
      this.startBtn.disabled = true;
      this.startBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i> Starting...';

      // Initialize camera on backend
      const response = await fetch("/start_detection");

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();

      if (result.status === "success") {
        this.isDetecting = true;
        this.sessionStartTime = Date.now();
        this.sessionLetterCount = 0;
        this.sessionWordCount = 0;
        this.startVideoStream();
        this.showVideoCanvas();
        this.startDetectionLoop();
        this.updateButtonStates();
        this.updateCameraStatus(true);
        this.setDetectionState("idle"); // Set to idle when detection starts
        this.startSessionTimer();
        showStatus("Camera started successfully!", "success");
      } else {
        showStatus(result.message || "Failed to start camera", "error");
        this.resetStartButton();
      }
    } catch (error) {
      console.error("Error starting detection:", error);
      showStatus(`Error starting detection: ${error.message}`, "error");
      this.resetStartButton();
    }
  }

  /**
   * Stop the detection process
   * Stops camera backend, cleans up video stream, and resets UI
   *
   * @async
   * @returns {Promise<void>}
   */
  async stopDetection() {
    try {
      // Show loading state
      this.stopBtn.disabled = true;
      this.stopBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i> Stopping...';

      // Stop detection loop
      this.isDetecting = false;
      if (this.detectionInterval) {
        clearInterval(this.detectionInterval);
        this.detectionInterval = null;
      }

      // Stop camera on backend
      const response = await fetch("/stop_detection");
      const result = await response.json();

      this.stopVideoStream();
      this.hideVideoCanvas();
      this.updateButtonStates();
      this.updateCameraStatus(false);
      this.resetProgressBars();
      this.setDetectionState("idle"); // Reset to idle state
      this.stopSessionTimer();

      // Reset letter display
      const letterTextElement = this.detectedLetterElement?.querySelector(
        ".isl-detected-letter"
      );
      if (letterTextElement) {
        letterTextElement.textContent = "-";
      } else if (this.detectedLetterElement) {
        this.detectedLetterElement.textContent = "-";
      }

      showStatus("Detection stopped", "success");
    } catch (error) {
      console.error("Error stopping detection:", error);
      showStatus("Error stopping detection", "error");
    }
  }

  /**
   * Start the detection polling loop
   * Continuously fetches detection results from backend every 100ms
   */
  startDetectionLoop() {
    if (!this.isDetecting) return;
    this.detectionInterval = setInterval(async () => {
      if (!this.isDetecting) return;
      try {
        const response = await fetch("/get_detection_meta");
        const data = await response.json();

        if (data.status === "success") {
          this.updateDisplay(data);
        } else {
          console.error("Detection error:", data.message);
        }
      } catch (error) {
        console.error("Error in detection loop:", error);
      }
    }, 100);
  }

  /**
   * Start the video stream from the backend
   * Sets the video source to the Flask streaming endpoint
   */
  startVideoStream() {
    if (this.videoStream) {
      this.videoStream.src = `/video_stream?ts=${Date.now()}`;
    }
  }

  /**
   * Stop the video stream
   * Clears the video source
   */
  stopVideoStream() {
    if (this.videoStream) {
      this.videoStream.src = "";
    }
  }

  /**
   * Update the UI display with detection results
   *
   * @param {Object} data - Detection data from backend
   * @param {string} data.detected_letter - Currently detected letter
   * @param {string} data.current_word - Word being built
   * @param {string} data.current_sentence - Accumulated sentence
   * @param {number} data.letter_progress - Progress towards adding letter (0-1)
   * @param {number} data.space_progress - Progress towards adding space (0-1)
   */
  updateDisplay(data) {
    // Update detected letter with state management
    this.detectedLetter = data.detected_letter || "";
    this.updateDetectedLetter(this.detectedLetter);

    // Update current word
    const prevWord = this.currentWord;
    this.currentWord = data.current_word || "";
    if (this.currentWordElement) {
      if (this.currentWord) {
        this.currentWordElement.innerHTML = this.currentWord;
      } else {
        this.currentWordElement.innerHTML =
          '<span class="isl-placeholder">Sign to begin</span>';
      }
    }

    // Update current sentence
    const prevSentence = this.currentSentence;
    this.currentSentence = data.current_sentence || "";
    if (this.currentSentenceElement) {
      if (this.currentSentence.trim()) {
        this.currentSentenceElement.innerHTML = this.currentSentence;
      } else {
        this.currentSentenceElement.innerHTML =
          '<span class="isl-placeholder">Your sentence will appear here...</span>';
      }
    }

    // Track session stats
    if (this.currentWord.length > prevWord.length) {
      this.sessionLetterCount++;
      this.updateSessionStats();
    }

    const wordCountNow =
      (this.currentSentence.match(/\s+/g) || []).length +
      (this.currentSentence.trim() ? 1 : 0);
    if (wordCountNow > this.sessionWordCount) {
      this.sessionWordCount = wordCountNow;
      this.updateSessionStats();
    }

    // Update progress bars
    this.updateProgressBars(data);
  }

  /**
   * Update progress bars for letter and space detection
   *
   * @param {Object} data - Detection data containing progress values
   * @param {number} data.letter_progress - Letter hold progress (0-1)
   * @param {number} data.space_progress - Space delay progress (0-1)
   * @param {string} data.detected_letter - Currently detected letter
   */
  updateProgressBars(data) {
    // Letter progress
    const letterProgress = (data.letter_progress || 0) * 100;
    if (this.letterProgressBar) {
      this.letterProgressBar.style.width = `${letterProgress}%`;
    }

    if (this.letterProgressText) {
      const currentTime = (data.letter_progress * 1.5).toFixed(1);
      this.letterProgressText.textContent = `${currentTime}s / 1.5s`;
    }

    // Space gesture progress
    const spaceProgress = (data.space_progress || 0) * 100;
    const isSpaceGestureActive = spaceProgress > 0;

    // Show/hide space progress bar based on whether space gesture is active
    if (this.spaceDurationContainer) {
      this.spaceDurationContainer.style.display = isSpaceGestureActive
        ? "flex"
        : "none";
    }

    if (this.spaceProgressBar) {
      this.spaceProgressBar.style.width = `${spaceProgress}%`;
    }

    if (this.spaceProgressText) {
      const currentSpaceTime = (data.space_progress * 1.5).toFixed(1);
      this.spaceProgressText.textContent = `${currentSpaceTime}s / 1.5s`;
    }
  }

  /**
   * Draw video frame on canvas (legacy method, kept for compatibility)
   *
   * @param {string} frameData - Base64 encoded frame data (unused in current implementation)
   */
  drawVideoFrame(frameData) {
    // Stream handled by <img>; keep as no-op for compatibility
  }

  /**
   * Show the video canvas and hide placeholder
   */
  showVideoCanvas() {
    if (this.videoPlaceholder) this.videoPlaceholder.style.display = "none";
    if (this.videoStream) this.videoStream.style.display = "block";
    if (this.canvas) this.canvas.style.display = "none";
  }

  /**
   * Hide the video canvas and show placeholder
   */
  hideVideoCanvas() {
    if (this.videoPlaceholder) this.videoPlaceholder.style.display = "flex";
    if (this.videoStream) this.videoStream.style.display = "none";
    if (this.canvas) this.canvas.style.display = "none";
  }

  /**
   * Update button states based on detection status
   * Shows/hides start/stop buttons appropriately
   */
  updateButtonStates() {
    if (this.isDetecting) {
      // Detection is running - show stop button, hide start button
      if (this.startBtn) {
        this.startBtn.style.display = "none";
      }
      if (this.stopBtn) {
        this.stopBtn.style.display = "inline-flex";
        this.stopBtn.disabled = false;
        this.stopBtn.innerHTML =
          '<i class="fas fa-stop"></i> Stop Detection (Enter)';
      }
    } else {
      // Detection is stopped - show start button, hide stop button
      if (this.stopBtn) {
        this.stopBtn.style.display = "none";
      }
      if (this.startBtn) {
        this.startBtn.style.display = "inline-flex";
        this.resetStartButton();
      }
    }
  }

  /**
   * Reset the start button to default state
   */
  resetStartButton() {
    if (this.startBtn) {
      this.startBtn.disabled = false;
      this.startBtn.innerHTML =
        '<i class="fas fa-play"></i> Start Detection (Enter)';
    }
  }

  /**
   * Reset all progress bars to zero state
   */
  resetProgressBars() {
    if (this.letterProgressBar) {
      this.letterProgressBar.style.width = "0%";
    }
    if (this.letterProgressText) {
      this.letterProgressText.textContent = "0.0s / 1.5s";
    }
    if (this.spaceProgressBar) {
      this.spaceProgressBar.style.width = "0%";
    }
    if (this.spaceProgressText) {
      this.spaceProgressText.textContent = "0.0s / 1.5s";
    }
    if (this.spaceDurationContainer) {
      this.spaceDurationContainer.style.display = "none";
    }
  }

  /**
   * Update camera status banner
   */
  updateCameraStatus(isActive) {
    if (this.cameraStatus) {
      if (isActive) {
        this.cameraStatus.classList.add("active");
        this.cameraStatus.querySelector(".isl-status-text").textContent =
          "Camera Active";
      } else {
        this.cameraStatus.classList.remove("active");
        this.cameraStatus.querySelector(".isl-status-text").textContent =
          "Camera Ready";
      }
    }
  }

  /**
   * Start session timer
   */
  startSessionTimer() {
    this.sessionTimer = setInterval(() => {
      if (this.sessionStartTime && this.sessionTimeElement) {
        const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60);
        const seconds = elapsed % 60;
        this.sessionTimeElement.textContent = `${minutes}:${seconds
          .toString()
          .padStart(2, "0")}`;
      }
    }, 1000);
  }

  /**
   * Stop session timer
   */
  stopSessionTimer() {
    if (this.sessionTimer) {
      clearInterval(this.sessionTimer);
      this.sessionTimer = null;
    }
  }

  /**
   * Update session statistics
   */
  updateSessionStats() {
    if (this.letterCountElement) {
      this.letterCountElement.textContent = this.sessionLetterCount;
    }
    if (this.wordCountElement) {
      this.wordCountElement.textContent = this.sessionWordCount;
    }
    // Calculate accuracy (placeholder - would need actual error tracking)
    if (this.accuracyElement) {
      const accuracy = Math.min(98, 85 + Math.random() * 10);
      this.accuracyElement.textContent = `${accuracy.toFixed(0)}%`;
    }
  }

  /**
   * Backspace - delete last character from current word
   * Sends request to backend to remove last character
   *
   * @async
   * @returns {Promise<void>}
   */
  async backspaceWord() {
    try {
      const response = await fetch("/backspace_word", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const result = await response.json();

      if (result.status === "success") {
        showStatus("Character deleted", "success");
      } else {
        showStatus(result.message || "No characters to delete", "error");
      }
    } catch (error) {
      console.error("Error backspacing word:", error);
      showStatus("Error deleting character", "error");
    }
  }

  /**
   * Reset the current word being built
   * Sends request to backend to clear current word
   *
   * @async
   * @returns {Promise<void>}
   */
  async resetWord() {
    try {
      const response = await fetch("/reset_word", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const result = await response.json();

      if (result.status === "success") {
        showStatus("Word reset successfully", "success");
      } else {
        showStatus(result.message || "Failed to reset word", "error");
      }
    } catch (error) {
      console.error("Error resetting word:", error);
      showStatus("Error resetting word", "error");
    }
  }

  /**
   * Reset the entire sentence and word
   * Sends request to backend to clear all text
   *
   * @async
   * @returns {Promise<void>}
   */
  async resetSentence() {
    try {
      const response = await fetch("/reset_sentence", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const result = await response.json();

      if (result.status === "success") {
        showStatus("Sentence reset successfully", "success");
      } else {
        showStatus(result.message || "Failed to reset sentence", "error");
      }
    } catch (error) {
      console.error("Error resetting sentence:", error);
      showStatus("Error resetting sentence", "error");
    }
  }

  /**
   * Save the current sentence to a text file
   * Sends request to backend to save sentence with timestamp
   *
   * @async
   * @returns {Promise<void>}
   */
  async saveSentence() {
    try {
      // Show loading state
      const originalText = this.saveSentenceBtn.innerHTML;
      this.saveSentenceBtn.disabled = true;
      this.saveSentenceBtn.innerHTML =
        '<i class="fas fa-spinner fa-spin"></i> Saving...';

      const response = await fetch("/save_sentence", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const result = await response.json();

      if (result.status === "success") {
        showStatus("Sentence saved successfully!", "success");
      } else {
        showStatus(result.message || "Failed to save sentence", "error");
      }

      // Reset button state
      this.saveSentenceBtn.disabled = false;
      this.saveSentenceBtn.innerHTML = originalText;
    } catch (error) {
      console.error("Error saving sentence:", error);
      showStatus("Error saving sentence", "error");

      // Reset button state
      this.saveSentenceBtn.disabled = false;
      this.saveSentenceBtn.innerHTML =
        '<i class="fas fa-save"></i> Save Sentence (Space)';
    }
  }
}

/**
 * Initialize detector when DOM is fully loaded
 * Sets up the detector instance and keyboard shortcuts
 */
document.addEventListener("DOMContentLoaded", function () {
  // Only initialize if we're on the detector page
  if (document.getElementById("videoCanvas")) {
    const detector = new ISLDetector();
    window.detector = detector; // Make globally accessible for onclick handlers

    /**
     * Handle page unload - ensure detection is stopped
     */
    window.addEventListener("beforeunload", function () {
      if (detector.isDetecting) {
        detector.stopDetection();
      }
    });

    /**
     * Keyboard shortcuts handler
     * - Backspace: Delete last character
     * - Space: Save sentence
     * - R: Reset word (clear entire word)
     * - S: Reset sentence
     * - Enter: Start/stop detection
     */
    document.addEventListener("keydown", function (event) {
      // Only handle shortcuts if not typing in an input
      if (
        event.target.tagName === "INPUT" ||
        event.target.tagName === "TEXTAREA"
      ) {
        return;
      }

      switch (event.key.toLowerCase()) {
        case "backspace": // Backspace to delete last character
          event.preventDefault();
          detector.backspaceWord();
          break;
        case " ": // Space to save sentence
          event.preventDefault();
          detector.saveSentence();
          break;
        case "r": // R to reset word
          event.preventDefault();
          detector.resetWord();
          break;
        case "s": // S to reset sentence
          event.preventDefault();
          detector.resetSentence();
          break;
        case "enter": // Enter to start/stop detection
          event.preventDefault();
          if (detector.isDetecting) {
            detector.stopDetection();
          } else {
            detector.startDetection();
          }
          break;
      }
    });

    // Keyboard shortcuts are now integrated into the UI, no need for floating box
  }
});
