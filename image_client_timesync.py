import cv2
import zmq
import numpy as np
import time
import struct
from collections import deque
from multiprocessing import shared_memory
import logging_mp
import argparse

# Configure logger
logger_mp = logging_mp.get_logger(__name__)
logging_mp.basic_config(level=logging_mp.INFO)

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="192.168.123.164", help="ip address for the server")


class ImageClient:
    """
    A client that connects to the ImageServer, performs a time-synchronization
    handshake, and then receives and processes a timestamped image stream.
    """
    def __init__(self, tv_img_shape=None,
                 tv_img_shm_name=None, wrist_img_shape=None, wrist_img_shm_name=None,
                 delay_shm_name=None,
                 image_show=False, server_address="192.168.123.164", port=5555, handshake_port=5556):
        """
        Args:
            tv_img_shape (tuple): Head camera resolution (H, W, C).
            tv_img_shm_name (str): Shared memory name for the head camera image.
            wrist_img_shape (tuple): Wrist camera resolution (H, W, C).
            wrist_img_shm_name (str): Shared memory name for the wrist camera image.
            image_show (bool): Whether to display received images in real time.
            server_address (str): The IP address of the image server.
            port (int): The port for the image stream.
            handshake_port (int): The port for the time synchronization handshake.
        """
        self.running = True
        self._image_show = image_show
        self._server_address = server_address
        self._port = port
        self._handshake_port = handshake_port

        self.tv_img_shape = tv_img_shape
        self.wrist_img_shape = wrist_img_shape

        # --- ZMQ and Time Sync Initialization ---
        self._context = zmq.Context()
        self._socket = None
        self.time_offset = 0  # Difference between server clock and client clock

        # --- Shared Memory Setup ---
        self.tv_enable_shm = self._setup_shm(tv_img_shm_name, tv_img_shape, 'tv')
        self.wrist_enable_shm = self._setup_shm(wrist_img_shm_name, wrist_img_shape, 'wrist')

        self.delay_enable_shm = False
        if delay_shm_name is not None:
            try:
                # A float (double in C) is 8 bytes
                self.delay_shm = shared_memory.SharedMemory(name=delay_shm_name)
                # Use a 0-dim NumPy array view for easy access
                self.delay_array = np.ndarray((), dtype=np.float64, buffer=self.delay_shm.buf)
                self.delay_enable_shm = True
                logger_mp.info(f"Successfully attached to shared memory '{delay_shm_name}' for delay.")
            except FileNotFoundError:
                logger_mp.error(f"Shared memory block '{delay_shm_name}' not found. Cannot share delay.")


        # --- Moving Average and Logging Setup ---
        # deque(maxlen) is an efficient way to store a rolling list of values.
        self._delay_history = deque(maxlen=500) # Adjust the maxlen as needed for your application
        self._last_log_time = time.time()
        self._log_interval_sec = 20.0 # Log every 10 seconds

    def _setup_shm(self, shm_name, img_shape, prefix):
        """Helper to initialize shared memory segments."""
        if img_shape is not None and shm_name is not None:
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                setattr(self, f"{prefix}_image_shm", shm)
                setattr(self, f"{prefix}_img_array", np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf))
                logger_mp.info(f"Successfully attached to shared memory '{shm_name}'.")
                return True
            except FileNotFoundError:
                logger_mp.error(f"Shared memory block '{shm_name}' not found. Please ensure the consumer process has created it.")
                return False
        return False

    def _perform_handshake(self, num_samples=100):
        """
        Connects to the handshake server to calculate network delay and clock offset.
        Performs multiple samples for better accuracy.
        """
        logger_mp.info(f"Performing handshake with server at {self._server_address}:{self._handshake_port}...")
        handshake_socket = self._context.socket(zmq.REQ)
        handshake_socket.connect(f"tcp://{self._server_address}:{self._handshake_port}")

        offsets = []
        rtts = []

        try:
            handshake_socket.setsockopt(zmq.RCVTIMEO, 2000) # 2 seconds
            handshake_socket.setsockopt(zmq.LINGER, 0)

            for i in range(num_samples):
                t0 = time.time()
                handshake_socket.send(b"sync")

                try:
                    packed_server_time = handshake_socket.recv()
                    t1 = time.time()
                    server_time = struct.unpack('d', packed_server_time)[0]

                    rtt = t1 - t0
                    # Standard NTP-like offset calculation
                    offset = server_time - (t0 + rtt / 2)

                    rtts.append(rtt)
                    offsets.append(offset)
                    # Small delay to avoid flooding the server or hitting rate limits
                    time.sleep(0.05)

                except zmq.Again:
                    logger_mp.warning(f"Handshake sample {i+1}/{num_samples} failed: No response.")
                    continue
                except Exception as e:
                    logger_mp.error(f"An error occurred during handshake sample {i+1}/{num_samples}: {e}")
                    continue

            if not offsets:
                logger_mp.error("Handshake failed: No successful samples obtained.")
                return False

            self.time_offset = sum(offsets) / len(offsets)
            average_rtt = sum(rtts) / len(rtts)

            logger_mp.info(f"Handshake successful after {len(offsets)} samples!")
            logger_mp.info(f"  - Average Network RTT: {average_rtt * 1000:.2f} ms")
            logger_mp.info(f"  - Estimated clock offset: {self.time_offset:.4f} s (Server is ahead if > 0)")

        except Exception as e:
            logger_mp.error(f"An error occurred during handshake: {e}")
            return False
        finally:
            handshake_socket.close()

        return True

    def _close(self):
        """Cleans up resources."""
        if self._socket:
            self._socket.close()
        self._context.term()
        if self._image_show:
            cv2.destroyAllWindows()
        logger_mp.info("Image client has been closed.")
    def stop(self):
        self.running = False

    def receive_process(self):
        """
        Main loop to connect, perform handshake, and receive image data.
        """
        # --- 1. Handshake ---
        if not self._perform_handshake():
            self._close()
            return

        # --- 2. Connect to Image Stream ---
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{self._server_address}:{self._port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        logger_mp.info("Connected to image stream. Waiting to receive data...")

        try:
            start_time = time.time()
            frame_count = 0
            while self.running:
                # Receive message
                message = self._socket.recv()
                receive_time = time.time()

                # --- 3. Unpack Timestamp and Image ---
                # First 8 bytes are the packed double timestamp
                server_timestamp = struct.unpack('d', message[:8])[0]
                jpg_bytes = message[8:]

                # --- 4. Calculate and Log Delay ---
                # Adjust server time with the calculated offset to sync clocks
                # The adjusted_server_time represents when the server *sent* the image
                # as measured by the client's clock.
                adjusted_server_time = server_timestamp - self.time_offset
                delay = receive_time - adjusted_server_time

                # Enforce non-negative delay: a received image cannot logically arrive before it's sent.
                # Small negative values (e.g., due to measurement inaccuracies) are set to zero.
                #if delay < 0:
                # this might happend due to clock drift?
                #    logger_mp.debug(f"Negative delay detected: {delay * 1000:.2f} ms. Clamping to 0.")
                #    delay = 0.0

                # Add the current delay to the history
                self._delay_history.append(delay)

                # add global average FPS
                frame_count += 1
                fps = frame_count / float(time.time() - start_time)

                # Check if it's time to log the moving average
                current_time = time.time()
                if current_time - self._last_log_time >= self._log_interval_sec:
                    if self._delay_history:
                        average_delay = sum(self._delay_history) / len(self._delay_history)
                        logger_mp.info(f"[Image Client] Average Frame Latency: {average_delay * 1000:.2f} ms")
                        logger_mp.info("[Image Client] Global FPS: %.2f" % fps)
                    self._last_log_time = current_time

                # Decode image
                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                current_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                if current_image is None:
                    logger_mp.warning("[Image Client] Failed to decode image.")
                    continue

                # --- 5. Process Image (Shared Memory and Display) ---
                if self.tv_enable_shm:
                    # Ensure the slice matches the expected dimensions
                    if current_image.shape[1] >= self.tv_img_shape[1]:
                        np.copyto(self.tv_img_array, current_image[:, :self.tv_img_shape[1]])
                    else:
                        logger_mp.warning("[Image Client] TV image part is smaller than expected shape.")

                if self.wrist_enable_shm:
                    # Ensure the slice matches the expected dimensions
                    if current_image.shape[1] >= self.wrist_img_shape[1]:
                        np.copyto(self.wrist_img_array, current_image[:, -self.wrist_img_shape[1]:])
                    else:
                        logger_mp.warning("[Image Client] Wrist image part is smaller than expected shape.")

                # Write the delay to shared memory
                if self.delay_enable_shm:
                    try:
                        # Use the 0-dim NumPy array to write the scalar value
                        self.delay_array[()] = delay
                    except Exception as e:
                        logger_mp.warning(f"Failed to write delay to shared memory: {e}")



                if self._image_show:
                    height, width = current_image.shape[:2]
                    # Resize for better display if the image is large
                    display_width = 800
                    scale = display_width / width
                    resized_image = cv2.resize(current_image, (display_width, int(height * scale)))

                    # Add delay text to the resized image
                    delay_text = f"Delay: {delay * 1000:.2f} ms"
                    cv2.putText(
                        resized_image,
                        delay_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )


                    fps_text = f"FPS: {fps:.1f}"
                    cv2.putText(
                        resized_image,
                        fps_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.imshow('Image Client Stream', resized_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

        except KeyboardInterrupt:
            logger_mp.info("Image client interrupted by user.")
        except Exception as e:
            logger_mp.warning(f"[Image Client] An error occurred while receiving data: {e}")
        finally:
            self._close()


if __name__ == "__main__":
    # Example usage:
    # Make sure to replace '127.0.0.1' with the actual IP of your server if it's on another machine.
    # Dummy shapes for testing if you don't have actual camera setups:
    # tv_img_shape = (480, 640, 3) # Height, Width, Channels
    # wrist_img_shape = (480, 640, 3)

    # client = ImageClient(image_show=True, server_address='192.168.123.164',
    #                      tv_img_shape=tv_img_shape, tv_img_shm_name='tv_img_shm',
    #                      wrist_img_shape=wrist_img_shape, wrist_img_shm_name='wrist_img_shm',
    #                      delay_shm_name='image_delay_shm')

    # For a simple test without shared memory or specific camera splits, uncomment this:
    args = parser.parse_args()
    client = ImageClient(image_show=True, server_address=args.ip)
    client.receive_process()
