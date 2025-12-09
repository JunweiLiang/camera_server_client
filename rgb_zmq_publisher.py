# coding=utf-8
"""
    ZeroMQ Publisher for Webcam Frames with Pygame Visualization

    Uses OpenCV to read frames and publish them via ZeroMQ PUB socket.
    Employs multithreading and a deque to ensure the latest frames are published
    at a controlled FPS. Pygame is used for robust, thread-safe GUI visualization.
    pip install pyzmq pygame
"""
import cv2
import threading
import argparse
from collections import deque
import time
import numpy as np
import zmq
import base64
import pygame # Import Pygame
import json # Import json for packaging data

# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="ZeroMQ Webcam Publisher with Pygame GUI")
parser.add_argument('--cam_num', type=int, default=0,
                    help='Camera device number (e.g., 0 for default webcam).')
parser.add_argument('--fps', type=int, default=5,
                    help='Target publishing FPS (for ZMQ stream).')
parser.add_argument('--h', type=int, default=720,
                    help='Frame height.')
parser.add_argument('--w', type=int, default=1280,
                    help='Frame width.')
parser.add_argument('--show_video', action='store_true',
                    help='Display the published video stream locally using Pygame.')
parser.add_argument('--jpeg_quality', type=int, default=70,
                    help='JPEG compression quality (0-100). Lower is smaller size, higher is better quality.')
parser.add_argument('--port', type=int, default=5555,
                    help='ZeroMQ port to bind to.')


class WebcamStreamZMQ:
    def __init__(self, src=0, fps=30, h=480, w=640, show_gui=False, jpeg_quality=70, port=5555):

        self.port = port
        self.jpeg_quality = jpeg_quality
        self.width = w
        self.height = h

        # --- ZeroMQ Context and Socket Setup ---
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        try:
            self.zmq_socket.bind(f'tcp://*:{self.port}')
            print(f"ZeroMQ Publisher bound to tcp://*:{self.port}")
        except zmq.error.ZMQError as e:
            print(f"Error binding ZeroMQ socket: {e}")
            print(f"Is port {self.port} already in use?")
            raise

        # --- Webcam Initialization ---
        self.stream = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not self.stream.isOpened():
            print(f"Error: Failed to open camera with src: {src}")
            raise IOError("Cannot open webcam")

        # Set camera properties
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Verify actual camera settings (often different from requested)
        actual_fps = self.stream.get(cv2.CAP_PROP_FPS)
        actual_width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Camera opened: {src}. "
              f"Actual Resolution: {int(actual_width)}x{int(actual_height)}, "
              f"Actual FPS: {actual_fps:.2f}")

        # --- Threading and Frame Queue Setup ---
        # The deque will now store tuples of (frame, timestamp)
        self.frame_queue = deque(maxlen=1)
        self.queue_lock = threading.Lock() # Protects access to the deque

        self.stopped = False # Flag to control thread termination

        # Initial read to populate self.frame and self.frame_queue
        ret, frame = self.stream.read()
        if ret:
            with self.queue_lock:
                # Store frame and its capture timestamp
                #self.frame_queue.append((frame, time.perf_counter())) # perf_counter cannot be used across machine
                self.frame_queue.append((frame, time.time()))
            self.frame = frame.copy() # For GUI display, updated by read thread
        else:
            print("Warning: Failed to read initial frame from camera. GUI might be blank initially.")
            self.frame = None

        self.frame_count = 0 # Counts frames successfully read from camera

        # Publisher FPS control
        self.target_fps = fps
        self.target_dt = 1.0 / self.target_fps

        # --- Start Threads ---
        self.read_thread = threading.Thread(target=self._update, daemon=True)
        self.publish_thread = threading.Thread(target=self._publish, daemon=True)
        self.read_thread.start()
        self.publish_thread.start()

        # Optional Pygame GUI display thread
        self.show_gui = show_gui
        if self.show_gui:
            self.gui_thread = threading.Thread(target=self._show_frames, daemon=True)
            self.gui_thread.start()

        print("WebcamStreamZMQ publisher initialized.")

    def _update(self):
        """
        Continuously reads frames from the camera.
        This thread runs as fast as the camera provides frames, keeping the deque
        updated with the absolute latest frame and preventing camera buffer buildup.
        """
        print("Starting camera read thread...")
        while not self.stopped:
            ret, frame = self.stream.read()
            if ret:
                self.frame_count += 1
                with self.queue_lock:
                    # Store frame and its capture timestamp
                    #self.frame_queue.append((frame, time.perf_counter())) # perf_counter cannot be used across machine
                    self.frame_queue.append((frame, time.time()))
                self.frame = frame.copy() # Update the frame used by GUI
            else:
                print("Warning: Failed to read frame from camera. Is it still connected?")
                time.sleep(0.1)
        print("Camera read thread stopped.")

    def _publish(self):
        """
        Publishes the latest frame from the deque via ZeroMQ PUB socket at the target FPS.
        The message now includes a timestamp for latency calculation.
        """
        print(f"Starting ZMQ publish thread at {self.target_fps} FPS...")
        while not self.stopped:
            start_time = time.time()

            frame_data_to_publish = None
            with self.queue_lock:
                if self.frame_queue:
                    # Retrieve the tuple (frame, timestamp)
                    frame_data_to_publish = self.frame_queue[0]

            if frame_data_to_publish is not None:
                frame_to_publish, timestamp = frame_data_to_publish
                try:
                    ret, buf = cv2.imencode(".jpg", frame_to_publish,
                                            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                    if ret:
                        jpg_as_text = base64.b64encode(buf).decode('utf-8')

                        # Create a dictionary to hold the image and timestamp
                        message = {
                            "timestamp": timestamp,
                            "image": jpg_as_text
                        }

                        # Serialize the dictionary to a JSON string and send
                        self.zmq_socket.send_string(json.dumps(message))
                        del buf # 感觉有用，手动尽快删除缓存
                    else:
                        print("Error: Failed to encode frame to JPEG.")
                except Exception as e:
                    print(f"Error publishing frame: {e}")
            else:
                time.sleep(0.005)

            elapsed = time.time() - start_time
            sleep_time = max(0.0, self.target_dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        print("ZMQ publish thread stopped.")

    def _show_frames(self):
        """
        Pygame GUI thread to display the latest frame.
        All Pygame operations happen within this thread.
        """
        print("Starting Pygame GUI display thread...")
        pygame.init() # Initialize Pygame here in the GUI thread
        try:
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Webcam Stream (ZMQ Publisher)")
            self.pygame_font = pygame.font.Font(None, 30) # Font for FPS display
            self.pygame_clock = pygame.time.Clock() # For controlling GUI FPS

            start_time_gui = time.time()
            displayed_frame_count = 0

            while not self.stopped:
                # Event handling for quitting the Pygame window
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Pygame window closed. Signaling shutdown.")
                        self.stop() # Signal all threads to stop
                        break # Break from event loop

                if self.stopped: # Check again after event loop
                    break

                current_frame_for_display = self.frame

                if current_frame_for_display is not None:
                    displayed_frame_count += 1
                    current_time = time.time()

                    if (current_time - start_time_gui) > 0:
                        camera_read_fps = self.frame_count / (current_time - start_time_gui)
                        gui_fps = int(displayed_frame_count / (current_time - start_time_gui))
                    else:
                        camera_read_fps = 0
                        gui_fps = 0

                    # Convert OpenCV BGR image (numpy array) to Pygame Surface (RGB)
                    # OpenCV uses BGR, Pygame expects RGB, so swap channels
                    frame_rgb = cv2.cvtColor(current_frame_for_display, cv2.COLOR_BGR2RGB)
                    pygame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)) # Transpose for Pygame


                    self.screen.fill((0, 0, 0)) # Clear screen (optional, but good practice)
                    self.screen.blit(pygame_surface, (0, 0)) # Draw the image

                    # Render FPS text
                    fps_text = self.pygame_font.render(
                        f"Cam Read FPS: {camera_read_fps:.1f} | GUI FPS: {gui_fps} | ZMQ Target FPS: {self.target_fps}",
                        True, (0, 255, 0) # Green color
                    )
                    self.screen.blit(fps_text, (10, 10))

                    pygame.display.flip() # Update the full display Surface to the screen
                else:
                    time.sleep(0.01) # Wait if no frame is available yet

                self.pygame_clock.tick(60) # Limit GUI FPS to 60 or less
        except Exception as e:
            print(f"Error in Pygame GUI thread: {e}")
        finally:
            pygame.quit() # Deinitialize Pygame here
            print("Pygame GUI display thread stopped.")


    def stop(self):
        """
        Signals all threads to stop and performs cleanup.
        """
        print("Stopping ZeroMQ webcam publisher...")
        self.stopped = True
        # Join threads to ensure they complete their execution
        if self.read_thread.is_alive():
            self.read_thread.join()
        if self.publish_thread.is_alive():
            self.publish_thread.join()
        if self.show_gui and self.gui_thread.is_alive():
            self.gui_thread.join() # Wait for GUI thread to finish its pygame.quit()

        self.stream.release() # Release the camera resource
        self.zmq_socket.close() # Close the ZMQ socket
        self.zmq_context.term() # Terminate the ZMQ context
        print("ZeroMQ webcam publisher stopped cleanly.")


if __name__ == "__main__":
    args = parser.parse_args()
    cam_stream_zmq = None
    try:
        cam_stream_zmq = WebcamStreamZMQ(
            args.cam_num, fps=args.fps, h=args.h, w=args.w,
            show_gui=args.show_video, jpeg_quality=args.jpeg_quality, port=args.port
        )
        print("ZeroMQ WebcamStreamZMQ instance created. Running...")

        # Keep the main thread alive until program is interrupted
        # The GUI thread will also signal 'stopped' if its window is closed
        while not cam_stream_zmq.stopped:
            time.sleep(0.1) # Small sleep to prevent busy-waiting

    except IOError as e:
        print(f"Error initializing camera: {e}")
    except zmq.error.ZMQError as e:
        print(f"ZeroMQ error during initialization: {e}")
    except KeyboardInterrupt:
        print("Publisher interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if cam_stream_zmq:
            cam_stream_zmq.stop()
        # pygame.quit() is called in the _show_frames thread's finally block,
        # so no need for cv2.destroyAllWindows() here.
