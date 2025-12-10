#!/usr/bin/env python3
"""
NailBite Detector prototype for Ubuntu using MediaPipe + OpenCV.

Press 's' to toggle monitoring, 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import subprocess
import shutil
import threading
import argparse

# --- Constants ---
FINGERTIP_INDICES = [8, 12]  # Index and middle fingertips
ALERT_MESSAGE = "Stop doing this!"
STATUS_MONITORING_ON = "Monitoring: ON"
STATUS_MONITORING_OFF = "Monitoring: OFF"
STATUS_ALERT = "ALERT: hand near face"
WINDOW_TITLE = 'NailBite Detector (press s to toggle, q to quit)'


def play_alert_sound(volume: float = 0.5, sound_type: str = 'beep'):
    """Plays an alert sound using simpleaudio or fallback system commands."""
    def _play_with_simpleaudio():
        try:
            import simpleaudio as sa
            fs = 44100
            duration = 0.2
            num_samples = int(fs * duration)
            t = np.linspace(0, duration, num_samples, False)

            note = None
            if sound_type == 'beep':
                # A simple 440 Hz sine wave
                note = np.sin(440 * t * 2 * np.pi)
            elif sound_type == 'sweep':
                # A frequency sweep from 880 Hz down to 440 Hz
                start_freq, end_freq = 880, 440
                freqs = np.linspace(start_freq, end_freq, num_samples)
                note = np.sin(freqs * t * 2 * np.pi)
            elif sound_type == 'noise':
                # A burst of white noise
                note = np.random.uniform(-1, 1, num_samples)

            if note is None:
                return False

            note *= 0.5  # Use 0.5 to avoid clipping at max volume
            audio = (note * volume * (2 ** 15 - 1)).astype(np.int16)
            sa.play_buffer(audio, 1, 2, fs).wait_done()
            return True
        except (ImportError, Exception) as e:
            print(f"Failed to play sound with simpleaudio: {e}")
            return False

    # Run in a separate thread to not block the main loop
    if _play_with_simpleaudio():
        return

    # Fallback to system commands if simpleaudio fails or is not installed
    try:
        if shutil.which("paplay"):
            pa_volume = int(np.clip(volume, 0.0, 1.0) * 65536)
            subprocess.Popen(["paplay", f"--volume={pa_volume}", "/usr/share/sounds/freedesktop/stereo/complete.oga"])
        elif shutil.which("aplay"):
            subprocess.Popen(["aplay", "/usr/share/sounds/alsa/Front_Center.wav"])
    except Exception as e:
        print(f"Failed to play fallback sound: {e}")


def send_notification(text: str):
    """Sends a desktop notification if notify-send is available."""
    if shutil.which("notify-send"):
        subprocess.Popen(["notify-send", text])


class NailBiteDetector:
    def __init__(self, camera_id=0, distance_threshold_ratio=0.5, frames_to_trigger=6, cooldown_seconds=3, volume=0.5, alert_sound='beep', headless=False):
        self.cap = cv2.VideoCapture(camera_id)
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(static_image_mode=False,
                                                  model_complexity=1,
                                                  min_detection_confidence=0.5,
                                                  min_tracking_confidence=0.5)
        self.distance_threshold_ratio = distance_threshold_ratio
        self.frames_to_trigger = frames_to_trigger
        self.cooldown_seconds = cooldown_seconds
        self.near_count = 0
        self.last_alert_time = 0
        self.monitoring = True
        self.volume = volume
        self.alert_sound = alert_sound
        self.headless = headless

    @staticmethod
    def _normalized_to_pixel(lm, w, h):
        return int(lm.x * w), int(lm.y * h)

    def run(self):
        if not self.cap.isOpened():
            print("Cannot open camera")
            return

        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.holistic.process(image)

                nose_px = None
                face_height = None

                if results.pose_landmarks:
                    nose = results.pose_landmarks.landmark[0]
                    nose_px = self._normalized_to_pixel(nose, w, h)

                if results.face_landmarks:
                    ys = [lm.y for lm in results.face_landmarks.landmark]
                    min_y = min(ys)
                    max_y = max(ys)
                    face_height = (max_y - min_y) * h

                is_hand_near = False

                # Check both hands fingertips
                if nose_px is not None:
                    threshold = (face_height if face_height else h * 0.2) * self.distance_threshold_ratio
                    for hand_landmarks in [results.right_hand_landmarks, results.left_hand_landmarks]:
                        if hand_landmarks:
                            for idx in FINGERTIP_INDICES:
                                lm = hand_landmarks.landmark[idx]
                                tip_px = self._normalized_to_pixel(lm, w, h)
                                dist = np.hypot(tip_px[0] - nose_px[0], tip_px[1] - nose_px[1])
                                if dist < threshold:
                                    is_hand_near = True

                status_text = STATUS_MONITORING_ON if self.monitoring else STATUS_MONITORING_OFF

                if self.monitoring:
                    if is_hand_near:
                        self.near_count += 1
                    else:
                        self.near_count = 0

                    if self.near_count >= self.frames_to_trigger and (time.time() - self.last_alert_time) > self.cooldown_seconds:
                        self.last_alert_time = time.time()
                        threading.Thread(target=play_alert_sound, args=(self.volume, self.alert_sound), daemon=True).start() # daemon=True to not block exit
                        send_notification(ALERT_MESSAGE)
                        status_text = STATUS_ALERT
                        self.near_count = 0

                if not self.headless:
                    # --- Visualization ---
                    vis = frame.copy()
                    if nose_px:
                        cv2.circle(vis, nose_px, 6, (255, 0, 0), -1)
                    
                    if results.right_hand_landmarks:
                        for idx in FINGERTIP_INDICES:
                            lm = results.right_hand_landmarks.landmark[idx]
                            cv2.circle(vis, self._normalized_to_pixel(lm, w, h), 6, (0, 0, 255), -1)
                    if results.left_hand_landmarks:
                        for idx in FINGERTIP_INDICES:
                            lm = results.left_hand_landmarks.landmark[idx]
                            cv2.circle(vis, self._normalized_to_pixel(lm, w, h), 6, (0, 0, 255), -1)

                    cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow(WINDOW_TITLE, vis)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    if key == ord('s'):
                        self.monitoring = not self.monitoring
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            self.cap.release()
            if not self.headless:
                cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="NailBite Detector using MediaPipe.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--camera_id", type=int, default=0,
                        help="ID of the camera to use.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Distance threshold ratio relative to face height.")
    parser.add_argument("--frames", type=int, default=6,
                        help="Number of consecutive frames to trigger an alert.")
    parser.add_argument("--cooldown", type=int, default=3,
                        help="Cooldown in seconds between alerts.")
    parser.add_argument("--volume", type=float, default=0.5,
                        help="Alert volume, from 0.0 (mute) to 1.0 (max).")
    parser.add_argument("--alert-sound", type=str, default='beep',
                        choices=['beep', 'sweep', 'noise'],
                        help="Type of the alert sound.")
    parser.add_argument("--headless", action='store_true',
                        help="Run without showing the camera preview window.")
    args = parser.parse_args()

    detector = NailBiteDetector(camera_id=args.camera_id,
                                distance_threshold_ratio=args.threshold,
                                frames_to_trigger=args.frames,
                                cooldown_seconds=args.cooldown,
                                volume=args.volume,
                                alert_sound=args.alert_sound,
                                headless=args.headless)
    detector.run()
