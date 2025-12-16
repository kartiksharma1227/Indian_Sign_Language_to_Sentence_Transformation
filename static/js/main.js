/**
 * Main JavaScript Module for ISL Translator Application
 *
 * Provides common functionality across all pages including:
 * - Mobile navigation menu handling
 * - Smooth scrolling for anchor links
 * - Scroll-based animations
 * - Parallax effects
 * - Utility functions for status messages and timing
 *
 *  
 *  
 */

/**
 * Initialize all common functionality when DOM is fully loaded
 */
document.addEventListener("DOMContentLoaded", function () {
  // Mobile navigation toggle
  const navToggle = document.querySelector(".nav-toggle");
  const navMenu = document.querySelector(".nav-menu");

  /**
   * Toggle mobile navigation menu on hamburger button click
   * Animates the hamburger icon into an X shape when menu is open
   */
  if (navToggle && navMenu) {
    navToggle.addEventListener("click", function () {
      navMenu.classList.toggle("active");

      // Animate hamburger menu
      const bars = navToggle.querySelectorAll(".bar");
      bars.forEach((bar, index) => {
        if (navMenu.classList.contains("active")) {
          if (index === 0)
            bar.style.transform = "rotate(45deg) translate(5px, 5px)";
          if (index === 1) bar.style.opacity = "0";
          if (index === 2)
            bar.style.transform = "rotate(-45deg) translate(7px, -6px)";
        } else {
          bar.style.transform = "none";
          bar.style.opacity = "1";
        }
      });
    });
  }

  /**
   * Close mobile menu when clicking on any navigation link
   * Provides better UX on mobile devices
   */
  const navLinks = document.querySelectorAll(".nav-link");
  navLinks.forEach((link) => {
    link.addEventListener("click", () => {
      if (navMenu.classList.contains("active")) {
        navMenu.classList.remove("active");
        // Reset hamburger menu
        const bars = navToggle.querySelectorAll(".bar");
        bars.forEach((bar) => {
          bar.style.transform = "none";
          bar.style.opacity = "1";
        });
      }
    });
  });

  /**
   * Enable smooth scrolling for all anchor links on the page
   * Improves navigation experience for in-page links
   */
  const anchorLinks = document.querySelectorAll('a[href^="#"]');
  anchorLinks.forEach((link) => {
    link.addEventListener("click", function (e) {
      e.preventDefault();
      const targetId = this.getAttribute("href");
      const targetElement = document.querySelector(targetId);
      if (targetElement) {
        targetElement.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });

  /**
   * Setup Intersection Observer for scroll-based animations
   * Elements fade in and slide up when they enter the viewport
   *
   * @type {IntersectionObserverInit}
   */
  const observerOptions = {
    threshold: 0.1,
    rootMargin: "0px 0px -100px 0px",
  };

  /**
   * Callback for Intersection Observer
   * Triggers fade-in and slide-up animations when elements become visible
   *
   * @param {IntersectionObserverEntry[]} entries - Array of observed elements
   */
  const observer = new IntersectionObserver(function (entries) {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = "1";
        entry.target.style.transform = "translateY(0)";
      }
    });
  }, observerOptions);

  /**
   * Observe all animatable elements for scroll-based animations
   * Initializes elements with hidden state before animation
   */
  const animateElements = document.querySelectorAll(
    ".feature-card, .tech-card, .process-step, .stat-item"
  );
  animateElements.forEach((element) => {
    element.style.opacity = "0";
    element.style.transform = "translateY(30px)";
    element.style.transition = "opacity 0.6s ease, transform 0.6s ease";
    observer.observe(element);
  });

  /**
   * Add parallax scrolling effect to hero section
   * Creates depth by moving hero visual at different speed than scroll
   */
  const hero = document.querySelector(".hero");
  if (hero) {
    window.addEventListener("scroll", function () {
      const scrolled = window.pageYOffset;
      const parallax = hero.querySelector(".hero-visual");
      if (parallax) {
        const speed = scrolled * -0.5;
        parallax.style.transform = `translateY(${speed}px)`;
      }
    });
  }

  /**
   * Animate gesture indicator dots in sequence
   * Creates a pulsing animation effect cycling through indicators
   */
  const gestureIndicators = document.querySelectorAll(".gesture-dot");
  if (gestureIndicators.length > 0) {
    let currentDot = 0;
    setInterval(() => {
      gestureIndicators.forEach((dot) => dot.classList.remove("active"));
      gestureIndicators[currentDot].classList.add("active");
      currentDot = (currentDot + 1) % gestureIndicators.length;
    }, 2000);
  }

  /**
   * Add visual feedback (scale animation) to all buttons on click
   * Provides tactile feedback for better UX
   */
  const buttons = document.querySelectorAll(".btn");
  buttons.forEach((button) => {
    button.addEventListener("click", function () {
      if (!this.disabled) {
        this.style.transform = "scale(0.98)";
        setTimeout(() => {
          this.style.transform = "scale(1)";
        }, 150);
      }
    });
  });
});

/**
 * Display a status message to the user
 *
 * @param {string} message - The message text to display
 * @param {string} [type='success'] - Message type: 'success', 'error', 'warning', or 'info'
 * @param {number} [duration=3000] - How long to display the message in milliseconds
 *
 * @example
 * showStatus('Camera started successfully!', 'success', 3000);
 * showStatus('Error loading camera', 'error', 5000);
 */
function showStatus(message, type = "success", duration = 3000) {
  // Remove existing status messages
  const existingMessages = document.querySelectorAll(".status-message");
  existingMessages.forEach((msg) => msg.remove());

  // Create new status message
  const statusElement = document.createElement("div");
  statusElement.className = `status-message ${type}`;
  statusElement.textContent = message;

  document.body.appendChild(statusElement);

  // Show the message
  setTimeout(() => {
    statusElement.classList.add("show");
  }, 100);

  // Hide and remove the message
  setTimeout(() => {
    statusElement.classList.remove("show");
    setTimeout(() => {
      statusElement.remove();
    }, 300);
  }, duration);
}

/**
 * Format seconds into MM:SS display format
 *
 * @param {number} seconds - Total seconds to format
 * @returns {string} Formatted time string in MM:SS format
 *
 * @example
 * formatTime(125) // Returns "2:05"
 * formatTime(59)  // Returns "0:59"
 */
function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = Math.floor(seconds % 60);
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}

/**
 * Create a debounced version of a function to limit call frequency
 * Useful for optimizing expensive operations triggered by frequent events
 *
 * @param {Function} func - The function to debounce
 * @param {number} wait - Milliseconds to wait before executing function
 * @returns {Function} Debounced version of the original function
 *
 * @example
 * const debouncedSearch = debounce(performSearch, 300);
 * searchInput.addEventListener('input', debouncedSearch);
 */
function debounce(func, wait) {
  let timeout;
  return function executedFunction(...args) {
    const later = () => {
      clearTimeout(timeout);
      func(...args);
    };
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}
