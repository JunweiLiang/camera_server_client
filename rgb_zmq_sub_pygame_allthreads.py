# coding=utf-8
"""
    ZeroMQ CompressedImage Subscriber with Pygame Visualization

    Connects to a ZeroMQ PUB socket, receives compressed image frames,
    decompresses them, and displays them using a thread-safe Pygame GUI.
    Uses a deque(maxlen=1) to ensure the GUI always shows the latest frame,
    skipping old ones if processing is slower than reception.
    Now includes latency calculation based on publisher's timestamp.
    pip install pyzmq pygame
"""
import cv2
import argparse
import threading
from collections import deque
import time
import numpy as np
import zmq # Import ZeroMQ
import base64 # For decoding base64 image data
import json # For parsing JSON messages

import pygame # Import Pygame

# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser(description="ZeroMQ CompressedImage Subscriber with Pygame GUI")
parser.add_argument('--publisher_ip', type=str, default='127.0.0.1',
                    help='IP address of the ZeroMQ publisher.')
parser.add_argument('--publisher_port', type=int, default=5555,
                    help='Port of the ZeroMQ publisher.')
parser.add_argument('--show_video', action='store_true',
                    help='Display the received video stream locally using Pygame.')
parser.add_argument('--display_fps_limit', type=int, default=60,
                    help='Maximum FPS for the Pygame GUI display.')


class WebcamStreamSubscriberZMQ:
    def __init__(self, publisher_ip, publisher_port, display_fps_limit=60):
        self.publisher_ip = publisher_ip
        self.publisher_port = publisher_port
        print(f"WebcamStreamSubscriberZMQ initializing, connecting to tcp://{self.publisher_ip}:{self.publisher_port}...")

        # --- Frame Queue for Decoupling ---
        # deque(maxlen=1) ensures we only ever store the latest received frame.
        # This means the GUI thread always gets the freshest data.
        # Now stores tuples of (frame, latency_ms)
        self.frame_queue = deque(maxlen=1)
        self.queue_lock = threading.Lock() # Protects access to the deque

        self.received_frame_count = 0 # Counts frames received via ZMQ
        self.stopped = False # Flag to control thread termination

        # Store the last decoded frame for GUI to access consistently
        self.current_display_frame = None
        self.current_latency_ms = 0.0 # Store the latest calculated latency
        self.display_width = 640 # Default resolution, will be adjusted by first frame size
        self.display_height = 480

        # --- ZeroMQ Socket Setup ---
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        try:
            self.zmq_socket.connect(f'tcp://{self.publisher_ip}:{self.publisher_port}')
            self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, '') # Subscribe to all messages (empty topic)
            print(f"ZeroMQ Subscriber connected to tcp://{self.publisher_ip}:{self.publisher_port}.")
        except zmq.error.ZMQError as e:
            print(f"Error connecting ZeroMQ socket: {e}")
            print(f"Please ensure the publisher is running and accessible at tcp://{self.publisher_ip}:{self.publisher_port}.")
            raise

        # --- ZMQ Frame Receiving Thread ---
        self.receive_thread = threading.Thread(target=self._receive_frames_zmq, daemon=True)
        self.receive_thread.start()
        print("ZMQ frame receiving thread started.")

        # --- Pygame GUI Thread Setup ---
        self.show_gui = False # Default, set by main function
        self.display_fps_limit = display_fps_limit
        self.gui_thread = None # Will be initialized and started if show_gui is True

        print("WebcamStreamSubscriberZMQ initialized.")


    def set_show_gui(self, show_gui_flag):
        """Method to set the show_gui flag from outside the class and start GUI thread."""
        self.show_gui = show_gui_flag
        if self.show_gui and self.gui_thread is None:
            self.gui_thread = threading.Thread(target=self._pygame_gui_loop, daemon=True)
            self.gui_thread.start()
            print("Pygame GUI thread started.")


    def _receive_frames_zmq(self):
        """
        Thread function to continuously receive frames from ZeroMQ.
        Decodes the JSON message, extracts the timestamp and image,
        calculates latency, and pushes the image and latency to the deque.
        """
        print("Starting ZMQ frame receiving loop...")
        while not self.stopped:
            try:
                # Receive the JSON string
                message_str = self.zmq_socket.recv_string()

                # Record reception timestamp as early as possible
                #received_at_timestamp = time.perf_counter() # perf_counter cannot be used across machine
                received_at_timestamp = time.time()

                # Parse the JSON string
                message = json.loads(message_str)
                publisher_timestamp = message['timestamp']
                jpg_as_text = message['image']

                # Calculate latency
                latency_ms = (received_at_timestamp - publisher_timestamp) * 1000

                img_bytes = base64.b64decode(jpg_as_text)

                # Convert bytes to numpy array and decode JPEG
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                del np_arr # Explicitly free memory if numpy array is large

                if frame is not None:
                    self.received_frame_count += 1
                    # Update global display dimensions for Pygame if not set
                    if self.current_display_frame is None: # Only set once after first valid frame
                        self.display_height, self.display_width, _ = frame.shape
                        print(f"Detected frame resolution: {self.display_width}x{self.display_height}")

                    # Push the latest frame and latency into the deque.
                    with self.queue_lock:
                        self.frame_queue.append(frame) # For the deque's maxlen=1 behavior
                        self.current_display_frame = frame.copy() # Update for GUI thread
                        self.current_latency_ms = latency_ms # Store latest latency for GUI
                else:
                    print("Warning: Failed to decode received ZMQ image, skipping.")

            except zmq.Again: # Timeout occurred, no message
                continue
            except zmq.error.ZMQError as e:
                if self.stopped: # If stopping, gracefully exit
                    break
                print(f"ZMQ error in receive thread: {e}. Retrying connection...")
                time.sleep(1) # Wait before retrying ZMQ operation
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}. Received invalid message format.")
            except Exception as e:
                print(f"Error decoding or processing ZMQ message: {e}")
                self.stopped = True # Signal for shutdown on unexpected error
                break

        print("ZMQ frame receiving thread stopped.")


    def _pygame_gui_loop(self):
        """
        Dedicated thread for Pygame GUI operations.
        All Pygame calls must happen within this single thread.
        Displays latency.
        """
        print("Starting Pygame GUI loop...")
        pygame.init()
        try:
            # Set up the Pygame display based on the first received frame's resolution
            # If no frame received yet, use default, it will be resized later
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            pygame.display.set_caption("ZMQ Subscriber Display (Pygame)")
            self.pygame_font = pygame.font.Font(None, 30) # Font for FPS and latency text
            self.pygame_clock = pygame.time.Clock() # To control GUI FPS

            gui_frame_count = 0
            start_time_gui = time.time()

            while not self.stopped:
                # --- Pygame Event Handling ---
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("Pygame window closed. Signaling subscriber shutdown.")
                        self.stop() # Signal the main loop to stop
                        break # Exit event loop

                if self.stopped: # Check if stop was signaled by event
                    break

                # --- Get Latest Frame for Display ---
                frame_to_display = None
                current_latency_for_display = 0.0
                with self.queue_lock:
                    # Access the last frame and its associated latency
                    if self.current_display_frame is not None:
                        frame_to_display = self.current_display_frame
                        current_latency_for_display = self.current_latency_ms

                if frame_to_display is not None:
                    # Update screen dimensions if they changed (e.g., from default to actual frame size)
                    current_frame_height, current_frame_width, _ = frame_to_display.shape
                    if (current_frame_width, current_frame_height) != self.screen.get_size():
                        self.display_width = current_frame_width
                        self.display_height = current_frame_height
                        self.screen = pygame.display.set_mode((self.display_width, self.display_height))
                        print(f"Resizing Pygame window to {self.display_width}x{self.display_height}")

                    # Convert OpenCV BGR image (numpy array) to Pygame Surface (RGB)
                    frame_rgb = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
                    pygame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)) # Transpose for Pygame

                    self.screen.fill((0, 0, 0)) # Clear screen with black
                    self.screen.blit(pygame_surface, (0, 0)) # Draw the image at top-left

                    # --- Display FPS and Latency Info ---
                    gui_frame_count += 1
                    current_time_gui = time.time()
                    if (current_time_gui - start_time_gui) > 0:
                        received_fps = self.received_frame_count / (current_time_gui - start_time_gui)
                        gui_fps = int(gui_frame_count / (current_time_gui - start_time_gui))
                    else:
                        received_fps = 0
                        gui_fps = 0

                    # Render FPS text
                    fps_text_line1 = self.pygame_font.render(
                        f"Received FPS: {received_fps:.1f}",
                        True, (0, 255, 0) # Green
                    )
                    fps_text_line2 = self.pygame_font.render(
                        f"GUI FPS: {gui_fps} (Limit: {self.display_fps_limit})",
                        True, (255, 255, 0) # Yellow
                    )
                    # Render Latency text
                    latency_text = self.pygame_font.render(
                        f"Latency: {current_latency_for_display:.2f} ms",
                        True, (255, 0, 0) # Red
                    )

                    self.screen.blit(fps_text_line1, (10, 10))
                    self.screen.blit(fps_text_line2, (10, 40))
                    self.screen.blit(latency_text, (10, 70)) # Position below FPS info

                    pygame.display.flip() # Update the entire screen
                else:
                    # If no frame available, pause briefly to avoid busy-waiting
                    time.sleep(0.005)

                # Control the GUI display FPS
                self.pygame_clock.tick(self.display_fps_limit)

        except Exception as e:
            print(f"Error in Pygame GUI loop: {e}")
        finally:
            pygame.quit() # Deinitialize Pygame properly
            print("Pygame GUI thread stopped.")


    def stop(self):
        """Signals all threads to stop and performs cleanup."""
        print("Stopping WebcamStreamSubscriberZMQ...")
        self.stopped = True
        # Join threads to ensure they complete their execution
        if self.receive_thread.is_alive():
            self.receive_thread.join()
        if self.gui_thread and self.gui_thread.is_alive():
            self.gui_thread.join()

        self.zmq_socket.close() # Close the ZMQ socket
        self.zmq_context.term() # Terminate the ZMQ context
        print("WebcamStreamSubscriberZMQ stopped cleanly.")


if __name__ == '__main__':
    args = parser.parse_args()
    cam_sub = None
    try:
        cam_sub = WebcamStreamSubscriberZMQ(
            publisher_ip=args.publisher_ip,
            publisher_port=args.publisher_port,
            display_fps_limit=args.display_fps_limit
        )

        # Pass the show_video argument to the subscriber instance
        cam_sub.set_show_gui(args.show_video)

        # Keep the main thread alive while background threads run
        # The 'stopped' flag will be set to True by the GUI thread on window close,
        # or by errors in ZMQ receiving thread.
        print("ZMQ Subscriber running... Press Ctrl+C or close Pygame window to stop.")
        while not cam_sub.stopped:
            time.sleep(0.1) # Small sleep to prevent busy-waiting

    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")
    finally:
        if cam_sub:
            cam_sub.stop() # Ensure all resources are cleaned up
