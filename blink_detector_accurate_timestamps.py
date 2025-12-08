"""
Blink Detector with Accurate Timestamp Logging
Uses MIDPOINT method for most accurate blink timing
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from collections import deque
from datetime import timedelta


class AccurateBlinkDetector:
    """
    Enhanced blink detector that logs accurate timestamps at blink midpoint.
    """
    
    def __init__(self, ear_threshold=0.18, consecutive_frames=2):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        
        # MediaPipe initialization
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Eye landmarks
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Tracking
        self.blink_counter = 0
        self.total_blinks = 0
        self.frame_counter = 0
        self.fps = 30
        
        # Blink tracking for accurate timestamps
        self.blink_start_frame = None
        self.blink_frames = []  # Store all frames during blink
        self.min_ear_in_blink = 1.0
        self.min_ear_frame = None
        
        # Logging
        self.blink_log = []
        self.ear_history = deque(maxlen=100)
    
    def set_fps(self, fps):
        """Set video FPS."""
        self.fps = fps
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio."""
        if len(eye_landmarks) < 6:
            return 0
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0
    
    def get_landmarks(self, face_landmarks, indices, width, height):
        """Extract landmark coordinates."""
        coords = []
        for idx in indices:
            if idx < len(face_landmarks.landmark):
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                coords.append((x, y))
        return coords
    
    def format_timestamp(self, frame_num):
        """Convert frame to timestamp."""
        seconds = frame_num / self.fps
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        minutes = total_seconds // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - total_seconds) * 1000)
        return f"{minutes:02d}:{secs:02d}.{milliseconds:03d}", seconds
    
    def log_blink(self):
        """
        Log blink with accurate midpoint timestamp.
        Called when blink is complete.
        """
        if not self.blink_frames:
            return
        
        # Get start and end frames
        start_frame = self.blink_frames[0]['frame']
        end_frame = self.blink_frames[-1]['frame']
        
        # Calculate midpoint (can be decimal)
        midpoint_frame = (start_frame + end_frame) / 2.0
        
        # Get timestamps
        start_ts_str, start_ts_sec = self.format_timestamp(start_frame)
        end_ts_str, end_ts_sec = self.format_timestamp(end_frame)
        mid_ts_str, mid_ts_sec = self.format_timestamp(midpoint_frame)
        
        # Calculate duration
        duration = (end_frame - start_frame) / self.fps
        
        # Get minimum EAR during blink
        min_ear = min(frame['ear'] for frame in self.blink_frames)
        min_ear_frame = [f for f in self.blink_frames if f['ear'] == min_ear][0]['frame']
        
        # Log to CSV
        self.total_blinks += 1
        self.blink_log.append({
            'Blink_Number': self.total_blinks,
            'Start_Frame': start_frame,
            'End_Frame': end_frame,
            'Midpoint_Frame': f"{midpoint_frame:.1f}",
            'Start_Time_Seconds': f"{start_ts_sec:.3f}",
            'End_Time_Seconds': f"{end_ts_sec:.3f}",
            'Midpoint_Time_Seconds': f"{mid_ts_sec:.3f}",
            'Start_Timestamp': start_ts_str,
            'End_Timestamp': end_ts_str,
            'Midpoint_Timestamp': mid_ts_str,
            'Duration_Seconds': f"{duration:.3f}",
            'Min_EAR': f"{min_ear:.3f}",
            'Min_EAR_Frame': min_ear_frame,
            'Num_Frames_Closed': len(self.blink_frames)
        })
        
        print(f"👁️ BLINK #{self.total_blinks}: {start_ts_str} to {end_ts_str} "
              f"(midpoint: {mid_ts_str}, duration: {duration*1000:.0f}ms)")
    
    def process_frame(self, frame):
        """Process frame and detect blinks with accurate timing."""
        self.frame_counter += 1
        h, w = frame.shape[:2]
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        blink_detected = False
        avg_ear = 0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye landmarks and calculate EAR
                left_eye = self.get_landmarks(face_landmarks, self.LEFT_EYE, w, h)
                right_eye = self.get_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
                
                if left_eye and right_eye:
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Draw landmarks
                    for coord in left_eye + right_eye:
                        cv2.circle(frame, coord, 2, (0, 255, 0), -1)
                
                self.ear_history.append(avg_ear)
                
                # Blink detection with accurate tracking
                if avg_ear < self.ear_threshold and avg_ear > 0:
                    # Eye is closing/closed
                    if self.blink_counter == 0:
                        # First frame of blink
                        self.blink_start_frame = self.frame_counter
                        self.blink_frames = []
                    
                    # Track this frame
                    self.blink_frames.append({
                        'frame': self.frame_counter,
                        'ear': avg_ear
                    })
                    
                    self.blink_counter += 1
                else:
                    # Eye is open
                    if self.blink_counter >= self.consecutive_frames:
                        # Log with accurate midpoint
                        self.log_blink()
                        blink_detected = True
                    
                    # Reset tracking
                    self.blink_counter = 0
                    self.blink_start_frame = None
                    self.blink_frames = []
        
        # Draw info
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if blink_detected or avg_ear < self.ear_threshold:
            cv2.putText(frame, "BLINKING", (w - 200, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        return frame, blink_detected, avg_ear
    
    def export_csv(self, filename='blink_timestamps_accurate.csv'):
        """Export detailed blink log to CSV."""
        if not self.blink_log:
            print("⚠️ No blinks detected.")
            return None
        
        df = pd.DataFrame(self.blink_log)
        df.to_csv(filename, index=False)
        print(f"\n✅ CSV exported: {filename}")
        print(f"   Total blinks: {len(self.blink_log)}")
        print(f"   Columns: {', '.join(df.columns)}")
        return filename


# Example usage
if __name__ == "__main__":
    print("""
    Accurate Blink Detection with Midpoint Timestamps
    =================================================
    
    This detector logs blinks at their MIDPOINT for accuracy.
    
    Example:
        Blink: 6.00s to 6.06s
        Logged: 6.03s (midpoint) ✅
    
    CSV includes:
    - Start/End frames and timestamps
    - Midpoint timestamp (main timestamp)
    - Duration
    - Minimum EAR
    - Number of frames eye was closed
    """)
