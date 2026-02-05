#!/usr/bin/env python3
"""
Runtime Data Recorder for Autonomous Deployment

This module provides button-triggered data recording during autonomous runs.
It mirrors the data collection format from collect.py while remaining real-time safe.

Usage in deploy.py:
    from deploy_recorder import RuntimeRecorder
    
    recorder = RuntimeRecorder(folder_name='runtime_data')
    
    while True:
        image = bot.getImage()
        delta = model(image)
        recorder.update(image, delta)
        ...
"""

import os
import cv2
import time
import numpy as np
from pynput import keyboard
from pynput.keyboard import Key
from threading import Lock


class RuntimeRecorder:
    """
    Button-triggered data recorder for autonomous deployment.
    
    Saves images and steering labels in the same format as collect.py:
    - Filename: {frame_number:06d}{angle:.2f}.jpg
    - Angle range: [-0.5, 0.5]
    - Auto-increments frame counter
    """
    
    def __init__(self, 
                 folder_name: str = 'runtime_recorded',
                 base_path: str = None,
                 record_button: str = 'r',
                 toggle_mode: bool = False,
                 autonomous_button: str = 'a'):
        """
        Initialize the runtime recorder.
        
        Args:
            folder_name: Name of the data folder (default: 'runtime_recorded')
            base_path: Base path for data directory. If None, auto-detects from script location
            record_button: Keyboard key to trigger recording (default: 'r')
            toggle_mode: If True, button toggles recording on/off. 
                        If False, must hold button to record (default: True)
            autonomous_button: Keyboard key to return to autonomous mode (default: 'a')
        """
        # === Path setup (mirrors collect.py) ===
        if base_path is None:
            script_path = os.path.dirname(os.path.realpath(__file__))
            base_path = os.path.join(script_path, '..', 'data')
        
        self.data_folder = os.path.join(base_path, folder_name)
        self.folder_name = folder_name
        self.ticks = 0
        # Create folder if it doesn't exist
        os.makedirs(self.data_folder, exist_ok=True)
        
        # === Frame counter (mirrors collect.py logic) ===
        self.frame_number = self._get_next_frame_number()
        
        # === Recording state ===
        self.is_recording = False
        self.toggle_mode = toggle_mode
        self.record_button = record_button
        self.autonomous_button = autonomous_button
        self._state_lock = Lock()  # Thread-safe state access
        
        # === Manual control state ===
        self.is_manual_control = False
        self.manual_steering = 0.0  # Range: [-0.5, 0.5]
        
        # === Button handling ===
        self._setup_keyboard_listener()
        
        # === Statistics ===
        self.frames_recorded = 0
        self.session_start_time = None
        
        print(f"[RuntimeRecorder] Initialized:")
        print(f"  Data folder: {self.data_folder}")
        print(f"  Starting frame: {self.frame_number}")
        print(f"  Record button: '{self.record_button}'" + 
              (" (toggle mode)" if toggle_mode else " (hold mode)"))
        print(f"  Manual control: LEFT/RIGHT arrows (press to activate)")
        print(f"  Autonomous mode: '{self.autonomous_button}'")
        print(f"  Ready to record!\\n")
    
    def _get_next_frame_number(self) -> int:
        """
        Find the next available frame number by scanning existing files.
        This mirrors the logic in collect.py (lines 24-32).
        
        Returns:
            Next frame number to use
        """
        try:
            filenames = os.listdir(self.data_folder)
            if len(filenames) == 0:
                return 0
            
            # Extract frame numbers from filenames (format: NNNNNN{angle}.jpg)
            frame_numbers = []
            for f in filenames:
                if f.endswith('.jpg'):
                    try:
                        # Get first 6 characters (frame number)
                        frame_num = int(f[:6])
                        frame_numbers.append(frame_num)
                    except ValueError:
                        continue
            
            if frame_numbers:
                return max(frame_numbers) + 1
            else:
                return 0
                
        except Exception as e:
            print(f"[RuntimeRecorder] Warning: Could not scan existing files: {e}")
            return 0
    
    def _setup_keyboard_listener(self):
        """
        Setup non-blocking keyboard listener for record button and manual control.
        Uses pynput (same as collect.py) for cross-platform compatibility.
        """
        def on_press(key):
            try:
                # Check for autonomous mode button
                if hasattr(key, 'char') and key.char == self.autonomous_button:
                    with self._state_lock:
                        if self.is_manual_control:
                            self.is_manual_control = False
                            self.manual_steering = 0.0
                            print(f"\\n[RuntimeRecorder] Switched to AUTONOMOUS mode")
                
                # Check if the pressed key matches our record button
                elif hasattr(key, 'char') and key.char == self.record_button:
                    with self._state_lock:
                        if self.toggle_mode:
                            # Toggle mode: flip recording state
                            self.is_recording = not self.is_recording
                            if self.is_recording:
                                self.session_start_time = time.time()
                                print(f"\\n[RuntimeRecorder] Recording STARTED (press '{self.record_button}' to stop)")
                            else:
                                duration = time.time() - self.session_start_time if self.session_start_time else 0
                                print(f"\\n[RuntimeRecorder] Recording STOPPED")
                                print(f"  Frames recorded this session: {self.frames_recorded}")
                                print(f"  Session duration: {duration:.1f}s\\n")
                                self.frames_recorded = 0
                        else:
                            # Hold mode: record while button is pressed
                            if not self.is_recording:
                                self.is_recording = True
                                self.session_start_time = time.time()
                                print(f"\\n[RuntimeRecorder] Recording... (release '{self.record_button}' to stop)")
                
                # Check for arrow keys (manual control - steering only)
                elif key == Key.left:
                    with self._state_lock:
                        if not self.is_manual_control:
                            self.is_manual_control = True
                            print(f"\\n[RuntimeRecorder] Switched to MANUAL CONTROL mode")
                            print(f"  UP/DOWN: Speed, LEFT/RIGHT: Steering")
                            print(f"  Press '{self.autonomous_button}' to return to autonomous\\n")
                        self.manual_steering += -0.25  # Turn left
                
                elif key == Key.right:
                    with self._state_lock:
                        if not self.is_manual_control:
                            self.is_manual_control = True
                            print(f"\\n[RuntimeRecorder] Switched to MANUAL STEERING mode")
                            print(f"  LEFT/RIGHT: Steering control")
                            print(f"  Press '{self.autonomous_button}' to return to autonomous\\n")
                        self.manual_steering += 0.25  # Turn right
                        
            except Exception as e:
                print(f"[RuntimeRecorder] Keyboard handler error: {e}")
        
        def on_release(key):
            try:
                # In hold mode, stop recording when button is released
                if not self.toggle_mode and hasattr(key, 'char') and key.char == self.record_button:
                    with self._state_lock:
                        if self.is_recording:
                            self.is_recording = False
                            duration = time.time() - self.session_start_time if self.session_start_time else 0
                            print(f"\\n[RuntimeRecorder] Recording STOPPED")
                            print(f"  Frames recorded this session: {self.frames_recorded}")
                            print(f"  Session duration: {duration:.1f}s\\n")
                            self.frames_recorded = 0
                            self.ticks = 0
                
                # Reset manual steering when arrow keys are released
                elif key == Key.left or key == Key.right:
                    with self._state_lock:
                        if self.is_manual_control:
                            self.manual_steering = 0.0
                            
            except Exception as e:
                print(f"[RuntimeRecorder] Keyboard handler error: {e}")
        
        # Start non-blocking keyboard listener
        self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.listener.start()
    
    def update(self, image: np.ndarray, delta: float) -> None:
        """
        Main update method called each control cycle.
        
        If recording is active, saves the image and steering label
        using the exact format from collect.py (line 119).
        
        This method is designed to be real-time safe:
        - Quick check of recording flag
        - Non-blocking file write (could be optimized further with queue)
        
        Args:
            image: Raw camera image (numpy array, any format)
            delta: Steering command in range [-0.5, 0.5] (or will be clamped)
        """
        # Quick check without lock (reading bool is atomic in Python)
        angle = np.clip(delta, -0.5, 0.5)
        if self.manual_steering is not None:
            self.manual_steering= np.clip(self.manual_steering, -0.5,0.5)
        if self.is_recording:
            
            
            try:
                if self.is_manual_control:
                    angle = self.manual_steering
                # Clamp angle to valid range (mirrors collect.py line 90)
                
                # Generate filename (mirrors collect.py line 119)
                # Format: {frame_number:06d}{angle:.2f}.jpg
                # Example: 000042-0.35.jpg or 001337+0.12.jpg
                filename = f"{str(self.frame_number).zfill(6)}{angle:+.2f}.jpg"
                filepath = os.path.join(self.data_folder, filename)
                
                if self.ticks % 4 == 0:
                # Save image (mirrors collect.py line 119)
                    cv2.imwrite(filepath, image)
                
                # Increment frame counter
                self.frame_number += 1
                self.frames_recorded += 1
                self.ticks+=1
                
                # Optional: Print progress every N frames
                if self.frames_recorded % 100 == 0:
                    print(f"[RuntimeRecorder] Recorded {self.frames_recorded} frames...")
                        
            except Exception as e:
                print(f"[RuntimeRecorder] Error saving frame: {e}")

        if self.is_manual_control:
            return self.manual_steering
        else :
            return None

    
    def stop(self):
        """
        Cleanup method to stop keyboard listener.
        Call this when shutting down the robot.
        """
        if hasattr(self, 'listener'):
            self.listener.stop()
        
        if self.is_recording and self.frames_recorded > 0:
            print(f"\\n[RuntimeRecorder] Final stats:")
            print(f"  Total frames recorded: {self.frames_recorded}")
            print(f"  Data saved to: {self.data_folder}")
    
    def get_stats(self) -> dict:
        """
        Get current recording statistics.
        
        Returns:
            Dictionary with frame count, folder, and recording state
        """
        return {
            'is_recording': self.is_recording,
            'frames_recorded': self.frames_recorded,
            'next_frame_number': self.frame_number,
            'data_folder': self.data_folder,
            'is_manual_control': self.is_manual_control
        }
    
    def get_manual_command(self) -> tuple:
        """
        Get the current manual control command.
        
        Returns:
            Tuple of (is_manual, steering):
                - is_manual: True if in manual control mode
                - steering: Manual steering command [-0.5, 0.5]
        """
        with self._state_lock:
            return (self.is_manual_control, self.manual_steering)


# === Example usage (for testing) ===
if __name__ == "__main__":
    print("RuntimeRecorder Test Mode")
    print("=" * 50)
    
    # Create test recorder
    recorder = RuntimeRecorder(
        folder_name='test_runtime_recording',
        toggle_mode=True
    )
    
    print("Simulating autonomous run...")
    print("Press 'r' to start/stop recording")
    print("Press Ctrl+C to exit\\n")
    
    try:
        frame_count = 0
        while True:
            # Simulate camera image (random noise)
            fake_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            
            # Simulate steering angle (sine wave)
            fake_angle = 0.3 * np.sin(frame_count * 0.1)
            
            # Call update (will only save if recording is active)
            recorder.update(fake_image, fake_angle)
            
            frame_count += 1
            time.sleep(0.1)  # 20 FPS simulation

    
            
    except KeyboardInterrupt:
        print("\\n\\nTest stopped by user")
        recorder.stop()
        stats = recorder.get_stats()
        print(f"\\nFinal statistics: {stats}")

