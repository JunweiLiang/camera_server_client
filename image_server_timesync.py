import cv2
import zmq
import time
import struct
from collections import deque
import numpy as np
import pyrealsense2 as rs
import logging_mp
import threading
import queue
import argparse

# It's good practice to have a centralized logger configuration.
# Assuming logging_mp is a custom module for multiprocessing logging.
logger_mp = logging_mp.get_logger(__name__, level=logging_mp.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument("--bino", action="store_true", help="use binocular camera, 2560x720, 1280x480")
parser.add_argument("--bino_camera_id", default=0, type=int, help="bino camera ID, v4l2-ctl --list-devices")
parser.add_argument("--bino_res", default="2560x720", choices=["2560x720", "1280x480", "3840x1080"])
parser.add_argument("--bino_fps", default=30, type=int, choices=[60, 30])
parser.add_argument("--rs_serial_id", default=None)

class RealSenseCamera(object):
    """
    A wrapper class for the Intel RealSense camera.
    """
    def __init__(self, img_shape, fps, serial_number=None, enable_depth=False) -> None:
        """
        Initializes the RealSense camera.
        Args:
            img_shape (list): The desired image shape [height, width].
            fps (int): The desired frames per second.
            serial_number (str, optional): The serial number of the device. Defaults to None.
            enable_depth (bool, optional): Whether to enable the depth stream. Defaults to False.
        """
        self.img_shape = img_shape
        self.fps = fps
        self.serial_number = serial_number
        self.enable_depth = enable_depth
        self.pipeline = None
        self.align = None
        self._device = None
        self.g_depth_scale = None
        self.intrinsics = None
        self.init_realsense()

    def init_realsense(self):
        """
        Configures and starts the RealSense pipeline.
        """
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number:
            config.enable_device(self.serial_number)

        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        try:
            profile = self.pipeline.start(config)
            self._device = profile.get_device()

            # Align depth frames to color frames
            align_to = rs.stream.color
            self.align = rs.align(align_to)

            if self.enable_depth:
                depth_sensor = self._device.first_depth_sensor()
                self.g_depth_scale = depth_sensor.get_depth_scale()

            self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        except RuntimeError as e:
            logger_mp.error(f"[Image Server] Failed to start RealSense camera {self.serial_number}: {e}")
            self.pipeline = None # Ensure pipeline is not used if it failed to start

    def get_frame(self):
        """
        Waits for and retrieves the next frame set from the camera.
        Returns:
            tuple: A tuple containing the color image and depth image (or None).
                   Returns (None, None) if frames are not available.
        """
        if not self.pipeline:
            return None, None

        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame() if self.enable_depth else None

            if not color_frame:
                return None, None

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) if depth_frame else None

            return color_image, depth_image
        except RuntimeError as e:
            logger_mp.error(f"[Image Server] Error getting frame from RealSense {self.serial_number}: {e}")
            return None, None

    def release(self):
        """
        Stops the RealSense pipeline.
        """
        if self.pipeline:
            self.pipeline.stop()
            logger_mp.info(f"[Image Server] RealSense camera {self.serial_number} released.")


class OpenCVCamera():
    """
    A wrapper class for standard OpenCV-compatible cameras.
    """
    def __init__(self, device_id, img_shape, fps):
        """
        Initializes an OpenCV camera.
        Args:
            device_id (int or str): The device ID (e.g., 0) or path (e.g., /dev/video0).
            img_shape (list): The desired image shape [height, width].
            fps (int): The desired frames per second.
        """
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            logger_mp.error(f"[Image Server] Camera {self.id} Error: Failed to open camera.")
            return

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        if not self._can_read_frame():
            logger_mp.error(f"[Image Server] Camera {self.id} Error: Failed to read initial frame.")
            self.release()

    def _can_read_frame(self):
        """
        Checks if a frame can be read from the camera.
        Returns:
            bool: True if a frame can be read, False otherwise.
        """
        if self.cap and self.cap.isOpened():
            success, _ = self.cap.read()
            return success
        return False

    def release(self):
        """
        Releases the camera capture object.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger_mp.info(f"[Image Server] OpenCV camera {self.id} released.")

    def get_frame(self):
        """
        Retrieves the next frame from the camera.
        Returns:
            numpy.ndarray or None: The color image, or None if the frame could not be read.
        """
        if not (self.cap and self.cap.isOpened()):
            return None
        ret, color_image = self.cap.read()
        return color_image if ret else None


class ImageServer:
    """
    A multi-threaded server that captures frames from multiple cameras
    and publishes them over a ZMQ socket. It also provides a handshake
    server for time synchronization.
    """
    def __init__(self, config, port=5555, handshake_port=5556):
        logger_mp.info(f"Initializing ImageServer with config: {config}")
        self.config = config
        self.port = port
        self.handshake_port = handshake_port
        self.fps = config.get('fps', 30)

        # --- Camera Initialization ---
        self.head_cameras = self._initialize_cameras('head')
        self.wrist_cameras = self._initialize_cameras('wrist')

        # --- Threading and Communication Setup ---
        self.frame_queue = queue.Queue(maxsize=5) # Bounded queue to prevent memory overflow
        self.stop_event = threading.Event()
        self.extractor_thread = None
        self.sender_thread = None
        self.handshake_thread = None

        # --- ZeroMQ Setup ---
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        self._log_camera_info()
        logger_mp.info(f"[Image Server] Image stream ready on port {self.port}.")
        logger_mp.info(f"[Image Server] Handshake server ready on port {self.handshake_port}.")

    def _initialize_cameras(self, camera_group):
        """
        A helper function to initialize a group of cameras based on config.
        Args:
            camera_group (str): The key for the camera group in the config (e.g., 'head', 'wrist').
        Returns:
            list: A list of initialized camera objects.
        """
        cameras = []
        cam_type = self.config.get(f'{camera_group}_camera_type')
        cam_ids = self.config.get(f'{camera_group}_camera_id_numbers')
        img_shape = self.config.get(f'{camera_group}_camera_image_shape')

        if not all([cam_type, cam_ids, img_shape]):
            logger_mp.info(f"[Image Server] No configuration for '{camera_group}' cameras.")
            return cameras

        for cam_id in cam_ids:
            camera = None
            if cam_type == 'opencv':
                camera = OpenCVCamera(device_id=cam_id, img_shape=img_shape, fps=self.fps)
            elif cam_type == 'realsense':
                camera = RealSenseCamera(img_shape=img_shape, fps=self.fps, serial_number=str(cam_id))
            else:
                logger_mp.warning(f"[Image Server] Unsupported camera type: {cam_type}")
                continue

            if isinstance(camera, OpenCVCamera) and (not camera.cap or not camera.cap.isOpened()):
                logger_mp.error(f"Failed to initialize OpenCV camera {cam_id}. Skipping.")
                continue
            if isinstance(camera, RealSenseCamera) and not camera.pipeline:
                logger_mp.error(f"Failed to initialize RealSense camera {cam_id}. Skipping.")
                continue

            cameras.append(camera)
        return cameras

    def _log_camera_info(self):
        """Logs resolution and other info for all initialized cameras."""
        for cam_group_name, cam_group in [("Head", self.head_cameras), ("Wrist", self.wrist_cameras)]:
            for cam in cam_group:
                if isinstance(cam, OpenCVCamera):
                    res = f"{int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}x{int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
                    logger_mp.info(f"[Image Server] {cam_group_name} camera {cam.id} resolution: {res}")
                elif isinstance(cam, RealSenseCamera):
                    res = f"{cam.img_shape[0]}x{cam.img_shape[1]}"
                    logger_mp.info(f"[Image Server] {cam_group_name} camera {cam.serial_number} resolution: {res}")

    def _frame_extractor_thread(self):
        """
        Thread target: Continuously extracts frames from all cameras and puts them in a queue.
        """
        logger_mp.info("[Image Server] Frame extractor thread started.")
        while not self.stop_event.is_set():
            head_frames = []
            for cam in self.head_cameras:
                frame = cam.get_frame()
                color_image = frame[0] if isinstance(cam, RealSenseCamera) else frame
                if color_image is None:
                    logger_mp.warning(f"[Image Server] Failed to get frame from a head camera.")
                    continue
                head_frames.append(color_image)

            wrist_frames = []
            for cam in self.wrist_cameras:
                frame = cam.get_frame()
                color_image = frame[0] if isinstance(cam, RealSenseCamera) else frame
                if color_image is None:
                    logger_mp.warning(f"[Image Server] Failed to get frame from a wrist camera.")
                    continue
                wrist_frames.append(color_image)

            if head_frames or wrist_frames:
                try:
                    self.frame_queue.put_nowait({'head': head_frames, 'wrist': wrist_frames})
                except queue.Full:
                    logger_mp.warning("[Image Server] Frame queue is full. Dropping a frame.")

            #time.sleep(1 / (self.fps * 2)) # prevent CPU high usage??
            time.sleep(0.001)
        logger_mp.info("[Image Server] Frame extractor thread stopped.")

    def _frame_sender_thread(self):
        """
        Thread target: Gets frames from queue, adds timestamp, processes, and sends via ZMQ.
        """
        logger_mp.info("[Image Server] Frame sender thread started.")
        while not self.stop_event.is_set():
            try:
                frames_dict = self.frame_queue.get(timeout=1.0)

                head_frames = frames_dict['head']
                wrist_frames = frames_dict['wrist']

                full_color = None
                head_color = cv2.hconcat(head_frames) if head_frames else None
                wrist_color = cv2.hconcat(wrist_frames) if wrist_frames else None

                if head_color is not None and wrist_color is not None:
                    full_color = cv2.hconcat([head_color, wrist_color])
                elif head_color is not None:
                    full_color = head_color
                elif wrist_color is not None:
                    full_color = wrist_color
                else:
                    continue

                ret, buffer = cv2.imencode('.jpg', full_color, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ret:
                    logger_mp.error("[Image Server] Frame imencode failed.")
                    continue

                # --- Prepend timestamp to the message ---
                # 'd' is for a double-precision float (8 bytes)
                timestamp = time.time()
                packed_timestamp = struct.pack('d', timestamp)
                message = packed_timestamp + buffer.tobytes()

                self.socket.send(message)
                self.frame_queue.task_done()

            except queue.Empty:
                continue
        logger_mp.info("[Image Server] Frame sender thread stopped.")

    def _handshake_thread(self):
        """
        Thread target: Runs a REP socket to respond to client pings with the server time.
        This allows clients to calculate network delay and clock offset.
        """
        logger_mp.info("[Image Server] Handshake thread started.")
        handshake_socket = self.context.socket(zmq.REP)
        handshake_socket.bind(f"tcp://*:{self.handshake_port}")

        poller = zmq.Poller()
        poller.register(handshake_socket, zmq.POLLIN)

        while not self.stop_event.is_set():
            try:
                # Poll for incoming messages with a timeout
                socks = dict(poller.poll(timeout=500))
                if handshake_socket in socks and socks[handshake_socket] == zmq.POLLIN:
                    # Receive the request, its content doesn't matter for this simple handshake
                    _ = handshake_socket.recv()
                    # Reply with the current server time
                    server_time = time.time()
                    handshake_socket.send(struct.pack('d', server_time))
            except zmq.ZMQError as e:
                logger_mp.error(f"[Image Server] ZMQ error in handshake thread: {e}")
                break

        handshake_socket.close()
        logger_mp.info("[Image Server] Handshake thread stopped.")

    def start(self):
        """
        Starts all the server threads.
        """
        self.extractor_thread = threading.Thread(target=self._frame_extractor_thread)
        self.sender_thread = threading.Thread(target=self._frame_sender_thread)
        self.handshake_thread = threading.Thread(target=self._handshake_thread)

        self.extractor_thread.daemon = True
        self.sender_thread.daemon = True
        self.handshake_thread.daemon = True

        self.extractor_thread.start()
        self.sender_thread.start()
        self.handshake_thread.start()
        logger_mp.info("[Image Server] All threads have been started.")

    def stop(self):
        """
        Signals threads to stop and cleans up resources.
        """
        logger_mp.info("[Image Server] Stopping server...")
        self.stop_event.set()

        # Wait for threads to finish
        if self.extractor_thread: self.extractor_thread.join(timeout=2)
        if self.sender_thread: self.sender_thread.join(timeout=2)
        if self.handshake_thread: self.handshake_thread.join(timeout=2)

        # Release camera resources
        for cam in self.head_cameras + self.wrist_cameras:
            cam.release()

        # Close ZMQ resources
        self.socket.close()
        self.context.term()
        logger_mp.info("[Image Server] Server has been closed gracefully.")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.bino:
        # Example configuration
        width, height = args.bino_res.split("x")
        width, height = int(width), int(height)
        config = {
            'fps': args.bino_fps,
            'head_camera_type': 'opencv',
            #'head_camera_image_shape': [720, 2560],  # [height, width]
            'head_camera_image_shape': [height, width],
            'head_camera_id_numbers': [args.bino_camera_id], # Use strings for serial numbers

            #'wrist_camera_type': 'opencv',
            #'wrist_camera_image_shape': [480, 640],
            #'wrist_camera_id_numbers': [0, 2], # Use integers for /dev/video*
        }
    else:
        # Example configuration
        config = {
            'fps': 30,
            'head_camera_type': 'realsense',
            #'head_camera_image_shape': [720, 1280],  # [height, width]
            # 模型实际使用640x480够了
            'head_camera_image_shape': [480, 640],  # [height, width]
            'head_camera_id_numbers': ["243222072371"], # Use strings for serial numbers

            #'wrist_camera_type': 'opencv',
            #'wrist_camera_image_shape': [480, 640],
            #'wrist_camera_id_numbers': [0, 2], # Use integers for /dev/video*
        }
        # 1号机的realsense: 242222070727
        # 2号机的realsense: 243222072371
        # 3号机：337122070060
        if args.rs_serial_id is not None:
            config["head_camera_id_numbers"] = [args.rs_serial_id]

    # Initialize the server with specific ports
    server = ImageServer(config, port=5555, handshake_port=5556)
    server.start()

    try:
        # Keep the main thread alive to handle signals like KeyboardInterrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger_mp.warning("[Image Server] Interrupted by user.")
    finally:
        server.stop()
