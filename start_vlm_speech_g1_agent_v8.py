# coding=utf-8
from langchain_community.utilities import BraveSearchWrapper
import sys
import os
# TODO add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from web_search.web_search_v2 import format_search_result_multithreaded

import cv2
import argparse
import queue
import threading
import time
import numpy as np
import zmq  # Import ZeroMQ
import base64  # For decoding base64 image data
import json  # For parsing JSON messages from ZMQ
import random

# documentation: https://python-sounddevice.readthedocs.io/en/0.5.1/usage.html#playback
import sounddevice as sd
import soundfile as sf

# 对silero-vad模型的封装，
from pysilero import VADIterator
import sys

sys.path.append("third_party/Matcha-TTS")

from streaming_sensevoice import StreamingSenseVoice

from openai import OpenAI
import base64
import time


import re

import pygame  # 替换opencv imshow
import requests


# --- Command Line Argument Parser ---
parser = argparse.ArgumentParser()

# ZMQ related arguments for video stream
parser.add_argument("--publisher_ip", type=str, default="127.0.0.1", help="IP address of the ZeroMQ publisher.")
parser.add_argument("--publisher_port", type=int, default=5555, help="Port of the ZeroMQ publisher for video frames.")

# camera related (now mostly for default dimensions, ZMQ will provide actual)
parser.add_argument("--cam_num", type=int, default=0, help="DEPRECATED: camera num, now using ZMQ for frames.")
parser.add_argument("--mic_id", type=int, default=0, help="mic device id")
parser.add_argument("--speaker_id", type=int, default=0, help="speaker device id")

parser.add_argument("--display_fps_limit", type=int, default=60, help="Target display FPS for GUI, ZMQ stream FPS might differ. ")

parser.add_argument("--h", type=int, default=1080, help="Default image height, will be updated by ZMQ stream. ")
parser.add_argument("--w", type=int, default=1920, help="Default image width, will be updated by ZMQ stream. ")
parser.add_argument("--show_video", action="store_true", help="show GUI during running")
parser.add_argument("--save_video", action="store_true")
parser.add_argument("--write_video_path", default="output.avi")


parser.add_argument("--tts_api_url_port", default="m10.precognition.team:50000")

parser.add_argument("--api_url_port", default="m10.precognition.team:8888")
# 7B模型很差
parser.add_argument("--vlm_model_name", default="Qwen2.5-VL-32B-Instruct/")

parser.add_argument("--max-turn", default=30, type=int, help="the message will be empty after 30 turn")

parser.add_argument("--asr_thres", type=float, default=0.5, help="for noisy env, you want to set higher")

# 跳过开头欢迎语音
parser.add_argument("--skip_open_msg", action="store_true")

# ----- for g1 stuff
parser.add_argument("--enable_g1", action="store_true")
parser.add_argument("--g1_network", default="enp58s0")
parser.add_argument("--gesture_data_path", default="gesture_data/")


# v8 添加方位角识别， 读取到新的方位角，是否要转g1过去
parser.add_argument("--enable_g1_360_wakeup", action="store_true")

parser.add_argument("--asr_cpu", action="store_true", help="running this on a non-gpu device")

# --- some global variable
# 打断词，语音生成过程中识别到这个会打断生成，重启ASR
# 语音生成中，说别的话也会打断，但是这个别的话也会发送给VLA，G1会有所回应，
# 比如问“我应该喝牛奶还是咖啡”，回答中，你说“我就是要喝咖啡”， G1会继续回应
# 打断词的话就会直接终止
stop_word = ["小科小科", "小柯小柯"]

# v8 加入方位识别，方位识别词在我们的ASR也要过滤掉。但是其实静音了麦克风就行;
asr_angle_word = ["小科小科", "小柯小柯"]

assert stop_word == asr_angle_word, "打断词和唤醒词需要一致"

# ------------------ some helper function for debugging and logging

from datetime import datetime


def print_with_time(*args, **kwargs):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(timestamp, *args, **kwargs)


"""

### ZMQ Video Stream Subscriber Class

This new class replaces `WebcamStream` and handles connecting to the ZeroMQ publisher, receiving compressed image frames, and decompressing them. It uses a `deque` with `maxlen=1` to ensure you always get the latest frame.

"""


class ZMQVideoSubscriber:
    def __init__(self, publisher_ip, publisher_port, default_h=480, default_w=640, save_video=False, output="output.avi"):
        self.publisher_ip = publisher_ip
        self.publisher_port = publisher_port
        print(f"ZMQVideoSubscriber initializing, connecting to tcp://{self.publisher_ip}:{self.publisher_port}...")

        self.frame_queue = queue.Queue(maxsize=1)  # Only store the very latest frame
        self.current_frame = None
        self.current_frame_count = 0
        self.current_frame_timestamp = time.time()  # Timestamp of the last received frame
        self.stopped = False

        self.display_width = default_w
        self.display_height = default_h

        self.current_latency_ms = 0.0  # Store the latest calculated latency

        # ZeroMQ Socket Setup
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        try:
            self.zmq_socket.connect(f"tcp://{self.publisher_ip}:{self.publisher_port}")
            self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages (empty topic)
            print(f"ZeroMQ Subscriber connected to tcp://{self.publisher_ip}:{self.publisher_port}.")
        except zmq.error.ZMQError as e:
            print(f"Error connecting ZeroMQ socket: {e}")
            print(f"Please ensure the publisher is running and accessible at tcp://{self.publisher_ip}:{self.publisher_port}.")
            raise

        self.receive_thread = threading.Thread(target=self._receive_frames, daemon=True)
        self.receive_thread.start()
        print("ZMQ frame receiving thread started.")

        self.save_video = save_video
        self.writer = None
        if save_video:
            self.write_queue = queue.Queue(maxsize=20000)  # Can tune based on memory & burst duration
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.writer = cv2.VideoWriter(output, fourcc, 30, (default_w, default_h))  # Use default, will update if first frame differs
            self.writer_thread = threading.Thread(target=self._write_loop, daemon=True)
            self.writer_thread.start()
            self.monitor_thread = threading.Thread(target=self._monitor_queue, daemon=True)
            self.monitor_thread.start()

    def _receive_frames(self):
        """Thread function to continuously receive frames from ZeroMQ."""
        while not self.stopped:
            try:
                # Receive the JSON string
                message_str = self.zmq_socket.recv_string()
                received_at_timestamp = time.time()  # Timestamp on reception

                # Parse the JSON string
                message = json.loads(message_str)
                publisher_timestamp = message["timestamp"]
                jpg_as_text = message["image"]

                # Calculate latency (optional, but good for debugging)
                latency_ms = (received_at_timestamp - publisher_timestamp) * 1000
                # print(f"Received frame, Latency: {latency_ms:.2f} ms") # Uncomment for per-frame latency

                img_bytes = base64.b64decode(jpg_as_text)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is not None:
                    self.current_frame_count += 1
                    self.current_frame_timestamp = time.time()  # Update timestamp for FPS calculation

                    # Update display dimensions based on first received frame
                    if self.current_frame is None or (self.display_width, self.display_height) != (frame.shape[1], frame.shape[0]):
                        self.display_height, self.display_width, _ = frame.shape
                        print(f"Detected ZMQ stream resolution: {self.display_width}x{self.display_height}")
                        # If saving video, update writer dimensions if changed
                        if self.save_video and self.writer:
                            self.writer.release()  # Release old writer
                            fourcc = cv2.VideoWriter_fourcc(*"XVID")
                            self.writer = cv2.VideoWriter(self.writer.baseFilename, fourcc, 30, (self.display_width, self.display_height))
                            print(f"Updated video writer resolution to {self.display_width}x{self.display_height}")

                    self.current_frame = frame.copy()  # Store the latest frame
                    self.current_latency_ms = latency_ms  # Store latest latency for GUI

                    if self.save_video:
                        try:
                            # put a timestamp for the frame for possible synchronization
                            # and a frame index to look up depth data
                            date_time = str(datetime.now())
                            self.write_queue.put_nowait((frame.copy(), date_time, self.current_frame_count))  # Don't block the capture loop
                        except queue.Full:
                            # Drop frames if queue is full
                            pass
                else:
                    print("Warning: Failed to decode received ZMQ image, skipping.")

            except zmq.Again:
                continue
            except zmq.error.ZMQError as e:
                if self.stopped:
                    break
                print(f"ZMQ error in receive thread: {e}. Retrying connection...")
                time.sleep(1)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}. Received invalid message format.")
            except Exception as e:
                print(f"Error in ZMQ receive loop: {e}")
                self.stopped = True
                break

    def _write_loop(self):
        while not self.stopped or not self.write_queue.empty():
            try:
                frame, date_time, frame_index = self.write_queue.get(timeout=0.1)
                frame = cv2.putText(frame, f"#{frame_index}: {date_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
                self.writer.write(frame)
            except queue.Empty:
                continue

    def _monitor_queue(self):
        while not self.stopped:
            if self.save_video:
                usage = self.write_queue.qsize()
                percent = 100.0 * usage / self.write_queue.maxsize
                if percent > 60.0:
                    print(f"\r[Frame write queue usage] {usage:5d}/{self.write_queue.maxsize} ({percent:5.1f}%)", end="")
            time.sleep(0.5)

    def read(self):
        """Returns the latest frame received from ZMQ."""
        return self.current_frame

    def get_dimensions(self):
        """Returns the current dimensions of the video stream."""
        return self.display_width, self.display_height

    def stop(self):
        self.stopped = True
        if self.receive_thread.is_alive():
            self.receive_thread.join()
        if self.save_video:
            print("\n[INFO] Waiting for video writer to flush remaining frames...")
            if self.writer_thread.is_alive():
                self.writer_thread.join()
            if self.writer:
                self.writer.release()
        self.zmq_socket.close()
        self.zmq_context.term()
        print("ZMQVideoSubscriber stopped cleanly.")


class ASRAgent:
    # ... (rest of your ASRAgent code remains the same) ...
    def __init__(self, mic_id=0, asr_thres=0.5, device="cuda"): # device: cuda/cpu
        # Loading remote code failed: model, No module named 'model' ? # seems harmless
        #self.model = StreamingSenseVoice(device="cuda", model="iic/SenseVoiceSmall", textnorm=True)
        self.model = StreamingSenseVoice(device=device, model="iic/SenseVoiceSmall", textnorm=True)

        # [https://github.com/pengzhendong/pysilero/blob/master/pysilero/pysilero.py#L323](https://github.com/pengzhendong/pysilero/blob/master/pysilero/pysilero.py#L323)
        self.vad_iterator = VADIterator(
            speech_pad_ms=300,  # speech chunk are paded this length each side
            # denoise=True, # 加降噪处理 # 对于某些麦克风设备，这个可能会报错
            denoise=False,  # 好像不加降噪反而语音识别更准?
            threshold=asr_thres,  # lower, than more speech segment
            # min_silence_duration_ms=550, # 每次讲话，等550ms静音才判定非speech # 降低这个可以提高反应速度
            min_silence_duration_ms=400,  # 这个太小比如200，会影响识别效果
        )

        # get one mic
        # sounddevice documentation: [https://python-sounddevice.readthedocs.io/en/0.3.12/usage.html#device-selection](https://python-sounddevice.readthedocs.io/en/0.3.12/usage.html#device-selection)
        devices = sd.query_devices()
        if len(devices) == 0:
            print("No microphone devices found")
            sys.exit(0)
        print(devices)

        input_device_idx = sd.default.device[mic_id]
        print("Use this mic: %s" % devices[input_device_idx]["name"])
        self.mic = devices[input_device_idx]
        self.samples_per_read = int(0.1 * 16000)

        # all recognized speech segments
        self.all_speech_segs = []

        self.tts_agent = None
        self.vla_agent = None
        self.asr_thread = None

    def _check_is_simple_chinese_text(self, new_text):
        # 匹配规则：
        # ^      ：字符串开头
        # [\u4e00-\u9fff]{0,n} ：0到n个中文字符（汉字）
        # [，。！？,.!?]$ ：结尾是常见中英文标点符号之一
        pattern = r"^[\u4e00-\u9fff0-9a-zA-Z]{0,2}[，。！？,.!?]$"
        return re.match(pattern, new_text) is not None

    def run_speech_rec(self):
        self.asr_thread = threading.Thread(target=self._run_speech_rec)
        self.asr_thread.daemon = True
        self.asr_thread.start()

    def _run_speech_rec(self):
        # start running the speech recognition, and do something for each seg
        with sd.InputStream(channels=1, dtype="float32", samplerate=16000) as s:
            while True:
                samples, _ = s.read(self.samples_per_read)

                for speech_dict, speech_samples in self.vad_iterator(samples[:, 0]):
                    if "start" in speech_dict:
                        self.model.reset()
                        # If a new speech segment starts, reset ASR start time if it was already set
                        # This ensures latency is measured per distinct user utterance.

                    is_last = "end" in speech_dict
                    stop_now = False
                    for res in self.model.streaming_inference(speech_samples * 32768, is_last):
                        has_stop_word = False
                        for stop_word_one in stop_word:
                            if stop_word_one in res["text"]:
                                has_stop_word = True
                        if has_stop_word:
                            # 打断词识别到了
                            print_with_time("[ASRAgent-识别到打断词]")
                            if self.tts_agent:
                                self.tts_agent.stop_and_empty()
                                print_with_time("[ASRAgent-已发送TTS agent停止指令]")
                            self.model.reset()  # reset the model so no lingering voices
                            stop_now = True

                        if stop_now:  # 退出当前ASR loop然后开启下一轮ASR等待
                            break

                        # 我们等完整的一句话说完识别了再发vlm，且当前没有在说话
                        if is_last:
                            new_text = res["text"].strip()
                            print_with_time("[ASRAgent-处理文本]: %s" % new_text)

                            # 消除噪音误识别，短的文本
                            if self._check_is_simple_chinese_text(new_text):
                                print_with_time("[ASRAgent-跳过短文本]: %s" % new_text)
                                continue

                            # 特殊指令
                            if "上下文" in new_text:
                                self.all_speech_segs = []
                                self.vla_agent.messages = [self.vla_agent.system_prompt]
                                tts_agent.send_non_block("上下文已清空")
                                print_with_time("[ASRAgent-清空上下文完成]")
                                continue

                            self.all_speech_segs.append(res)

                            print_with_time("[ASRAgent-发送vla_agent]: %s" % res["text"])

                            if self.tts_agent:
                                # 这时候也应该停掉还在讲的话
                                self.tts_agent.stop_and_empty()
                                print_with_time("[ASRAgent-已发送TTS agent停止指令]")
                            self.model.reset()  # reset the model so no lingering voices

                            # Capture time before sending to VLM
                            vlm_request_time = time.perf_counter()
                            self.vla_agent.send_non_block(new_text, vlm_request_time)  # Pass ASR start time to VLM agent


class TTSAgent:
    def __init__(self, asr_agent, speaker_id=0, api_url_port=None):

        self.tts_url = "http://{}/inference_zero_shot_fast".format(api_url_port)

        self.control_dt = 1.0 / 100  # 最高控制频率，至少隔这个时间才去看有没有东西要生成
        self.cosyvoice_sample_rate = 24000

        # First-in-first-out queue
        self.text_queue = []

        self.asr_agent = asr_agent

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._run_tts)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

        # 需要这些才能打断
        self.audio_stream = None  # The sounddevice OutputStream instance
        self.audio_data_buffer = None  # Stores the current audio segment to play
        self.current_frame = 0  # Tracks playback position within audio_data_buffer
        self.stop_playback_event = threading.Event()  # Signals the callback to stop playback

        # --- Latency Tracking ---
        self.tts_start_time = None

    def split_message(self, text):
        # Use regex to split while keeping the delimiter
        # 按标点拆分，并保留标点到原句子

        sentences = re.split(r"([。？！?!])", text)  # 按句号拆分
        result = []

        for i in range(0, len(sentences) - 1, 2):  # 两两合并（句子 + 标点）
            result.append(sentences[i] + sentences[i + 1])

        if len(sentences) % 2 == 1 and sentences[-1].strip():  # 处理没有结尾标点的情况
            result.append(sentences[-1])

        return result

    # 所以这个函数，可以一直接收要生成的文本，存起来
    def send_non_block(self, text):
        text = text.strip()
        if text:
            # 给定的text可能有多句话，要断句
            text_list = self.split_message(text)
            with self.ctrl_lock:
                self.text_queue += text_list

            # print_with_time("\t[TTSAgent-收到send_non_block , 当前text_queue %s]"% self.text_queue)

            # Capture the time when TTS is requested (if it's the first text in queue)
            if self.tts_start_time is None:
                self.tts_start_time = time.perf_counter()

    def stop_and_empty(self):
        """
        Stops current audio playback immediately and clears the text queue.
        This can be called from any thread.
        """
        with self.ctrl_lock:
            # Clear pending TTS texts
            self.text_queue = []
            print_with_time("\t[TTSAgent-Received stop_and_empty command. Text queue cleared.]")

            # Stop the currently playing audio stream if active
            if self.audio_stream and self.audio_stream.active:
                print_with_time("\t[TTSAgent-Signaling audio callback to stop...]")
                self.stop_playback_event.set()  # Signal the callback to stop providing data

                # Explicitly stop the stream object to ensure its active state updates
                print_with_time("\t[TTSAgent-Calling self.audio_stream.stop()...]")
                self.audio_stream.stop()  # This is the definitive stop command
                print_with_time("\t[TTSAgent-self.audio_stream.stop() called.]")
            else:

                print_with_time("\t[TTSAgent-No active audio stream to stop.]")
                self.stop_playback_event.clear()

            self.tts_start_time = None  # Reset TTS start time on stop

    def _audio_playback_callback(self, outdata, frames, time_info, status):
        """
        Sounddevice callback for feeding audio data.
        This runs in an internal sounddevice thread.
        """
        # 1. Handle potential errors reported by sounddevice
        if status:
            if status.output_underflow:
                print("TTSAgent Callback: Output underflow! Consider increasing blocksize or processing speed.", file=sys.stderr)
            else:
                print(f"TTSAgent Callback Status: {status}", file=sys.stderr)
            outdata.fill(0)
            return sd.CallbackStop

        # 2. Check for external stop signal (highest priority)
        if self.stop_playback_event.is_set():
            outdata.fill(0)
            return sd.CallbackStop

        # 3. Determine how many frames we can actually provide from our buffer
        num_frames_to_copy = min(frames, len(self.audio_data_buffer) - self.current_frame)

        # 4. Get the chunk of data from our buffer
        chunk = self.audio_data_buffer[self.current_frame : self.current_frame + num_frames_to_copy]

        # 5. Copy the chunk to outdata
        outdata[:num_frames_to_copy] = chunk

        # 6. Pad the rest of the outdata buffer with zeros if we didn't fill it
        if num_frames_to_copy < frames:
            outdata[num_frames_to_copy:].fill(0)

        # 7. Advance frame pointer by the amount copied
        self.current_frame += num_frames_to_copy

        # 8. Explicitly check if all audio data has been sent
        if self.current_frame >= len(self.audio_data_buffer):
            # print("returned sd.CallbackStop") # Debug print to confirm it's reached
            self.stop_playback_event.set()  # 必须用这个才能停止
            return sd.CallbackStop  # Signal that all data has been played

        # 9. If we reached here, it means more data is still available
        return sd.CallbackFlags(0)

    def _call_tts_api(self, text):
        payload = {
            "tts_text": text,
        }
        response = requests.request("GET", self.tts_url, data=payload, stream=True)

        tts_audio = b""
        for r in response.iter_content(chunk_size=16000):
            tts_audio += r
        # tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
        # print(tts_speech.shape) #torch.Size([1, 287040]) # this is for torchaudio.save()
        tts_speech = np.array(np.frombuffer(tts_audio, dtype=np.int16))

        return tts_speech

    # 一开始就准备着，当有文本生成的时候就生成
    def _run_tts(self):
        while True:
            start_time = time.time()

            text_to_read = None
            with self.ctrl_lock:
                # 取第一个text读
                if self.text_queue:
                    text_to_read = self.text_queue.pop(0)
            # for debug
            # print_with_time("\t[TTSAgent-run_tts , 当前text_queue %s, text_to_read: %s]"% (self.text_queue, text_to_read))

            if text_to_read:
                print_with_time("\t[TTSAgent-开始生成]: %s" % text_to_read)

                tts_speech = self._call_tts_api(text_to_read)

                self.audio_data_buffer = tts_speech.reshape(-1, 1)  # (N,) -> (N, 1)

                self.current_frame = 0  # Reset frame counter for new audio segment

                # 只有一个segment
                if self.tts_start_time is not None:
                    # 这个只是计算文本生成语音的时间
                    time_to_first_voice = time.perf_counter() - self.tts_start_time
                    print_with_time(f"\t[---Latency][TTSAgent-TTS time: {time_to_first_voice:.3f} seconds]")
                    self.tts_start_time = None  # Reset after first voice output

                # Initialize stop event for THIS playback segment
                self.stop_playback_event.clear()

                # --- Audio Playback ---
                try:
                    # Create the stream instance (NOT with a 'with' statement here)
                    # need to initialize every time
                    # This allows it to be controlled externally.
                    self.audio_stream = sd.OutputStream(
                        samplerate=self.cosyvoice_sample_rate,
                        channels=1,
                        dtype=np.int16,
                        callback=self._audio_playback_callback,  # Our custom callback
                        # finished_callback=self.stop_playback_event.set,
                    )
                    # with self.audio_stream:
                    #    self.stop_playback_event.wait()
                    self.audio_stream.start()
                    while True:
                        # print_with_time("playback event waiting...")
                        if not self.audio_stream.active or self.stop_playback_event.is_set():
                            break
                        time.sleep(0.05)
                    # print_with_time("playback event done.")

                except Exception as e:
                    print_with_time(f"\t[TTSAgent-Error during audio playback: {e}]")
                finally:
                    # Ensure the stream is closed after each segment or on error
                    if self.audio_stream:
                        if self.audio_stream.active:
                            # print_with_time("\t[TTSAgent-Ensuring stream is stopped before closing...]")
                            self.audio_stream.stop()  # Ensure it's stopped
                        # print_with_time("\t[TTSAgent-Closing audio stream.]")
                        self.audio_stream.close()
                        self.audio_stream = None  # Clear instance after closing

                self.stop_playback_event.clear()
                print_with_time("\t[TTSAgent-生成完成]")

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)


def frame_to_base64(frame, img_format=".png"):
    """Convert an OpenCV frame to a Base64-encoded image.

    Args:
        frame (numpy.ndarray): The OpenCV image frame.
        img_format (str): Image format (".jpg" or ".png", default is ".jpg").

    Returns:
        str: Base64-encoded image string.
    """
    # 编码为xx格式

    _, buffer = cv2.imencode(img_format, frame)  # Encode frame to image format
    base64_str = base64.b64encode(buffer).decode("utf-8")  # Convert to Base64 string
    return base64_str


class VLAAgent:
    def __init__(self, api_url_port, cam_stream, tts_agent, system_prompt, g1_ctr=None, max_turn=30, vlm_model_name="models/Qwen2.5-VL-7B-Instruct/"):

        openai_api_key = "None"
        openai_api_base = "http://%s/v1" % api_url_port
        self.max_turn = max_turn

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model_name = vlm_model_name  # vllm需要这个模型名字完全一致

        if not "BRAVE_SEARCH_API_KEY" in os.environ:
            os.environ["BRAVE_SEARCH_API_KEY"] = ""

        BRAVE_API_KEY = os.environ["BRAVE_SEARCH_API_KEY"]

        # ====== 初始化模块 ======
        self.search = BraveSearchWrapper(
            # brave_api_key=BRAVE_API_KEY,
            api_key=BRAVE_API_KEY,
            search_kwargs={
                "count": 5,  # 返回 x 条搜索结果。最大值为 20。
                "freshness": "py",  # 时间范围 "pd"（1天）|"pw"（1周）|"pm"（1月）
                # "search_lang": "zh",  # 返回中文搜索结果 # raise Brave Search API 的参数校验失败（HTTP 422）
                "ui_lang": "zh-CN",  # 返回中文界面
                # "region": "cn",  # 返回中国地区的搜索结果
            },
        )
    
        # default used in Open WebUI
        self.temperature = 0.7  # higher for creativity
        self.top_p = 1.0  # higher for creativity
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        self.cam_stream = cam_stream  # 一直在提图像帧

        self.control_dt = 1.0 / 100  # 最高控制频率，至少隔这个时间才去问VLM

        # First-in-last-out queue
        self.text_queue = []

        # the back and forth history
        self.messages = []

        # 先把system prompt写进去
        self.system_prompt = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
        self.messages.append(self.system_prompt)

        self.tts_agent = tts_agent  # text-to-speech agent
        self.g1_ctr = g1_ctr

        # 搭配讲话的可用手势
        self.speaking_hand_gestures = [
            "抬起右手",
            "左引导",
            "右引导",
        ]

        # initialize publish thread
        self.publish_thread = threading.Thread(target=self._run_vlm)
        self.ctrl_lock = threading.Lock()
        self.publish_thread.daemon = True
        self.publish_thread.start()

    # given a prompt, put the text into queue; 每次等上一次应用处理完，我就全部text一起发送
    # 所以这个函数，可以一直接收ASR的结果，存起来
    def send_non_block(self, text, vlm_request_time=None):
        with self.ctrl_lock:
            self.text_queue.append((text, vlm_request_time))  # Store text and vlm request start time

    def _make_message(self, text, image):
        # suppose the image is open cv numpy
        encoded_img = frame_to_base64(image, ".png")
        message = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,%s" % encoded_img},
                },
                {"type": "text", "text": text},
            ],
        }
        return message

    def generate_web_prompt(self, user_question, search_result):
        return f"""你是一位知识渊博的智能助手。以下是用户提出的问题和网页搜索结果：

        [问题]
        {user_question}

        [搜索结果]
        {search_result}

        请基于搜索内容，用简洁、准确的中文回答用户问题。"""

    # 一开始就准备着，当有文本要发送的时候，就发送
    def _run_vlm(self):
        while True:
            start_time = time.time()

            new_text_and_timestamps = []
            with self.ctrl_lock:
                # lock it so the ASR cannot modify it for now
                new_text_and_timestamps = self.text_queue.copy()
                if new_text_and_timestamps:
                    # empty the queue if there are new text in
                    self.text_queue = []

            if new_text_and_timestamps:
                _, vlm_request_time = new_text_and_timestamps[0]  # Take the first one for latency calculation
                new_text_list = [x[0] for x in new_text_and_timestamps]
                new_text = " ".join(new_text_list)  # 是用户的提问文本

                # Ensure cam_stream.read() doesn't return None before passing to _make_message
                frame_for_vlm = self.cam_stream.read()  # 是当前视频帧（图像数据）
                if frame_for_vlm is None:
                    print_with_time("[VLAAgent-Warning]: No frame available from ZMQ stream for VLM, skipping turn.")
                    time.sleep(0.1)  # Wait briefly before next check
                    continue

                ###########################################################################
                ############################## 联网查询功能开始 ##############################
                ###########################################################################

                # 如果用户的提问中有我要链接网络查询的关键词
                if "网络查询" in new_text:
                    print_with_time("[VLAAgent-收到网络查询指令]: %s" % new_text)

                    search_result = self.search.run(new_text)  # 调用联网搜索模块

                    # 根据搜索结果抓取网页正文
                    print_with_time("[VLAAgent-联网查询结果已返回]")
                    # 从搜索页面得到的网页，提取这些网页的信息
                    #search_result = format_search_result(str(search_result))
                    search_result = format_search_result_multithreaded(str(search_result))
                    print_with_time("[VLAAgent-联网查询结果已遍历并格式化]")

                    new_text = self.generate_web_prompt(new_text, search_result)

                ###########################################################################
                ############################## 联网查询功能结束 ##############################
                ###########################################################################

                # 将文本和图像封装为一个消息对象，并添加到 self.messages 中。
                new_message = self._make_message(new_text, frame_for_vlm)

                self.messages.append(new_message)
                print_with_time("[VLAAgent-发送message]: %s/2 轮对话" % len(self.messages))
                # this will take a while

                # VLM 模型处理用户提问
                chat_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    # default sampling_param same as OpenWebUI
                    top_p=self.top_p,  # higher for creativity
                    frequency_penalty=self.frequency_penalty,  # wdwd
                    presence_penalty=self.presence_penalty,
                    temperature=self.temperature,  # higher for creativity
                    # stream=True, # openweb UI uses this
                )
                # --- 这里计算从asr结束，发出vlm request，到vlm request回到，的时间
                vlm_inference_end_time = time.perf_counter()  # Timestamp after VLM inference
                vlm_inference_latency = vlm_inference_end_time - vlm_request_time

                print_with_time(f"[---Latency] ASR done to VLM Inference done: {vlm_inference_latency:.3f} seconds")

                # VLM 返回结果，提取模型的回复内容。
                assistant_message = chat_response.choices[0].message
                # response 是模型返回的文本结果。
                response_content = assistant_message.content.strip("., ")
                # 回复是空怎么办？
                if response_content == "":
                    print_with_time("[VLAAgent-收到空content-完整message]: %s" % assistant_message)

                # speech synthesis this without blocking
                # 不直接讲，先查看有没有中括号 动作
                # [12/2025] 发现，7B模型不会输出中括号，直接换成
                # 以“好的没问题，动作1，动作2。”模版进行回答
                #self.tts_agent.send_non_block(response_content)
                print_with_time("[VLAAgent-完整response_content]: %s" % response_content)
                #actions, text_response = self._extract_chinese_in_brackets_and_the_rest(response_content)
                actions, text_response = self._parse_command(response_content)

                self.tts_agent.send_non_block(text_response)
                print_with_time(
                    "[VLAAgent-发送TTSAgent (%d字符)]: %s" % (
                        len(text_response), text_response))
                if actions:
                    print_with_time(
                        "[VLAAgent-识别到actions]: %s" % (actions))


                # run G1 action if any
                if self.g1_ctr:
                    #result = self._extract_chinese_in_brackets(response_content)
                    if actions:
                        print_with_time("[VLAAgent-收到g1动作]:%s" % actions)
                        # 所以理论上VLM可以一次告诉g1执行多个动作
                        for item in actions:
                            self.g1_func_mapping(item)
                            time.sleep(2.0) # 保证连续执行

                    else:
                        # v8: 如果讲的话很多，添加一些手势
                        if len(text_response) > 20: # 一个中文大概两个字符, 20字符大概一句话
                            hand_gesture_choice = random.choice(self.speaking_hand_gestures)
                            print_with_time("[VLAAgent-说话手势]:%s" % hand_gesture_choice)
                            time.sleep(3.0) # 等声音开始
                            self.g1_func_mapping(hand_gesture_choice)


                self.messages.append(assistant_message.model_dump())

            if len(self.messages) >= self.max_turn * 2:  # 30 turns means 30 user-assistant
                print_with_time("[VLAAgent - 超过轮数上限自动清除上下文]")
                self.messages = []
                self.messages.append(self.system_prompt)

            current_time = time.time()
            all_t_elapsed = current_time - start_time
            sleep_time = max(0, (self.control_dt - all_t_elapsed))
            time.sleep(sleep_time)

    def _extract_chinese_in_brackets(self, text):
        # 匹配中括号中的中文字符，只保留中文
        pattern = r"\[([\u4e00-\u9fff]+)\]"
        matches = re.findall(pattern, text)
        return matches

    def _parse_command(self, text):
        """
        全能版：处理 换行符、括号、中英文逗号。

        能处理这种极端情况：

        "
        好的没问题
        [动作1]
        动作2
        "
        """
        test_cases = [
            # 情况1：前后有换行，中间也有换行
            """
            好的没问题
            [动作1]
            [动作2]
            """,

            # 情况2：混着写，有的换行，有的逗号
            "好的没问题，[打开夹爪]\n[向前移动]，动作3",

            # 情况3：纯文字换行
            "好的没问题\nMove\nStop",

            # 情况4：开头有一堆空行
            "\n\n好的没问题，[动作A]"
        ]
        if not text:
            return [], ""

        # 1. 全局去噪：先去掉字符串最开头和最结尾的 空格、换行
        text = text.strip()

        # 2. 匹配前缀
        # \s* 正则会自动匹配 空格、换行符(\n)、制表符(\t)
        prefix_match = re.match(r"^(好的没问题)[，,。.]?\s*", text)

        if prefix_match:
            text_response = "好的没问题"

            # 截取动作部分
            content_str = text[prefix_match.end():]

            # --- 核心逻辑 ---
            # 把所有可能的分隔符，统统变成英文逗号 ','

            # 1. 括号变逗号
            content_str = content_str.replace("[", ",").replace("]", ",")
            # 2. 中文逗号变英文逗号
            content_str = content_str.replace("，", ",")
            # 3. 【关键】把换行符变逗号
            content_str = content_str.replace("\n", ",")

            # 现在字符串变成了类似 ",,动作1,,动作2,,," 的样子
            raw_parts = content_str.split(",")

            actions = []
            for part in raw_parts:
                # 二次清洗：去掉每个动作两边的空格、句号
                clean_part = part.strip(" .。")
                if clean_part:
                    actions.append(clean_part)

            return actions, text_response

        else:
            return [], text

    def _extract_chinese_in_brackets_and_the_rest(self, text):
        pattern = r"\[([\u4e00-\u9fff]+)\]"

        # 1. Get the list of matches (only the content inside the brackets)
        matches = re.findall(pattern, text)

        # 2. Get the "rest of the text" by removing the matches
        # This replaces the whole pattern (including brackets) with an empty string
        rest_of_text = re.sub(pattern, "", text)

        # Return both
        return matches, rest_of_text

    def g1_func_mapping(self, skill_name):
        if self.g1_ctr:
            # 这个mapping需要与system prompt 对应
            func_mapping = {
                "往前走一步": self.g1_ctr.move_forward,
                "向前走一步": self.g1_ctr.move_forward, # 7B模型有时候会记错，加个保险
                "往左走一步": self.g1_ctr.move_left_lateral,
                "向左走一步": self.g1_ctr.move_left_lateral,
                "往右走一步": self.g1_ctr.move_right_lateral,
                "向右走一步": self.g1_ctr.move_right_lateral,
                "向左转": self.g1_ctr.move_turn_left,
                "向右转": self.g1_ctr.move_turn_right,
                "挥手": self.g1_ctr.wave_hand,
                #"握手": self.g1_ctr.shake_hand_up,
                "握手": self.g1_ctr.shake_hand_up_and_down, # junwei: 6秒自动放下手
                "放下手": self.g1_ctr.release_arm,
                "鼓掌": self.g1_ctr.clap,
                "比心": self.g1_ctr.heart,
                "抬起右手": self.g1_ctr.hand_up_and_down,
                "左引导": self.g1_ctr.right_welcome,
                "右引导": self.g1_ctr.left_welcome, # 这里左右引导，反过来，我假设说G1是对着访客，访客的左边就是G1的右边
            }
            if skill_name not in func_mapping:
                print_with_time("[VLAAgent-收到未定义的g1动作]: %s" % skill_name)
            else:
                g1_func_to_run = func_mapping[skill_name]
                g1_func_to_run()


if __name__ == "__main__":
    args = parser.parse_args()
    zmq_subscriber = None  # Renamed from cam_stream
    try:
        # v5
        system_prompt = """
        你是一个机器人。图像是你前方看到的视觉信息，并且从Picture 1开始按时间顺序往后排序。
        回答尽量简短，数字用中文表示，用句号断开。
        你有这些动作可以使用：[往前走一步]，[往左走一步]，[往右走一步]，[往左转一下]，[挥手]，[握手]，[放下手]。
        如果你遇到了需要用行动的指令，选择三个以内上述动作，以“好的，那么现在我[动作1][动作2]”模版进行回答，不要说多余的话。
        """
        # v5 -> v6，新增了很多手势
        system_prompt = """
        你是一个机器人。图像是你前方看到的视觉信息，并且从Picture 1开始按时间顺序往后排序。
        回答尽量简短，数字用中文表示，用句号断开。
        你有这些动作可以使用：[往前走一步]，[往左走一步]，[往右走一步]，[向左转]，[向右转]，[挥手]，[握手]，[放下手]，[比心]，[鼓掌]，[抬起右手]。
        如果你遇到了需要用行动的指令，选择三个以内上述动作，以“好的，那么现在我[动作1][动作2]”模版进行回答，不要说多余的话。
        """

        # v8，新增自定义手势, see robot_arm_high_level_v3.py
        system_prompt = """
        你是一个机器人。图像是你前方看到的视觉信息，并且从Picture 1开始按时间顺序往后排序。
        回答尽量简短，数字用中文表示，用句号断开。
        你有这些动作可以使用：[往前走一步]，[往左走一步]，[往右走一步]，[向左转]，[向右转]，[挥手]，[握手]，[放下手]，[比心]，[鼓掌]，[抬起右手]，[左引导]，[右引导]。
        如果你遇到了需要用行动的指令，选择三个以内上述动作，以“好的，我现在[动作1][动作2]”模版进行回答，不要说多余的话。
        """
        # [12/2025] 发现，7B模型不会输出中括号，直接换成
        # 以“好的没问题，动作1，动作2。”模版进行回答
        system_prompt = """
        你是一个导览机器人。图像是你前方看到的视觉信息，并且从Picture 1开始按时间顺序往后排序。
        回答尽量简短，数字用中文表示，用句号断开。
        导览结束语是：“感谢领导们的观看，我是机器人小科，预祝论坛圆满成功、。”，
        当我叫你“请说结束语”，你就把结束语说出来，不要说别的话。
        你有这些动作可以使用：
            [往前走一步]，
            [往左走一步]，
            [往右走一步]，
            [向左转]，
            [向右转]，
            [挥手]，[握手]，
            [放下手]，[比心]，[鼓掌]，[抬起右手]，
            [左引导]，[右引导]。
        如果你遇到了需要用行动的指令，选择三个以内上述动作，
        以“好的没问题，动作1，动作2。”模版进行回答，不要说多余的话。
        """

        # 开头信息

        intro_message = """
        领导您好，欢迎来到港科大广州具身智能实验室，
        我是打工机器人“小科”，我已经接入了视觉大模型，可以理解人类的指令、。
        """

        highlevel_ctr = None
        if args.enable_g1:
            from robot_arm_high_level_v3 import G1_Highlevel_Controller

            highlevel_ctr = G1_Highlevel_Controller(args.g1_network, args.gesture_data_path)
            print("starting g1 wave hand..")
            highlevel_ctr.wave_hand()
            time.sleep(2)
            print("wave hand returned.")
            print("setting Run Walk...")
            highlevel_ctr.set_run_walk()  # 走跑运控
            # highlevel_ctr.set_normal_walk() # 主运控，更稳一点，但是走路不拟人
            time.sleep(1)
            print("done.")



        # 1. Initialize ZMQ Video Stream Subscriber
        # Use default H/W first, it will be updated by the first frame received from ZMQ
        start_time_zmq = time.time()
        zmq_subscriber = ZMQVideoSubscriber(
            publisher_ip=args.publisher_ip,
            publisher_port=args.publisher_port,
            default_h=args.h,  # Pass default height
            default_w=args.w,  # Pass default width
            save_video=args.save_video,
            output=args.write_video_path,
        )
        print("ZMQVideoSubscriber initialized.")

        # 1. 载入asr agent
        device = "cpu" if args.asr_cpu else "cuda"
        asr_agent = ASRAgent(mic_id=args.mic_id, asr_thres=args.asr_thres, device=device)
        # 开启持续语音识别
        print("running ASR forever...")
        asr_agent.run_speech_rec()  # start the asr thread, non-blocking

        # 2. 开启语音生成agent, 有信息就会说话
        tts_agent = TTSAgent(asr_agent, args.speaker_id, api_url_port=args.tts_api_url_port)
        print("tts_agent initialized.")

        # 3. 开启VLM agent，有信息就会去发送
        # Pass the zmq_subscriber instance to VLAAgent
        vla_agent = VLAAgent(
            args.api_url_port,
            zmq_subscriber,
            tts_agent,
            system_prompt,  # Changed cam_stream to zmq_subscriber
            g1_ctr=highlevel_ctr,
            max_turn=args.max_turn,
            vlm_model_name=args.vlm_model_name,
        )
        print("vla_agent initialized.")

        # 4. 开启360度语音方位角唤醒、打断
        if args.enable_g1_360_wakeup:
            assert args.enable_g1
            from robot_arm_high_level_v3 import G1_29_ASR_360_Wakeup
            # 这里开始持续监听方位角信息，后面设置开始flag才会动
            g1_wakeup = G1_29_ASR_360_Wakeup(
                args.g1_network,
                no_dds_init=True, # g1_ctr已经init 了
                # 需要语音生成agent+ g1_highlevel配合，如果要响应唤醒
                tts_agent=tts_agent,
                g1_ctr=highlevel_ctr)

        # asr可能需要和tts, vla_agent通讯
        asr_agent.tts_agent = tts_agent
        asr_agent.vla_agent = vla_agent

        # wait till user says start.
        input("(注意此时ASR已开启)按回车键开始演示...")

        # 开启唤醒
        if args.enable_g1_360_wakeup:
            g1_wakeup.set_ctr_flag(True)

        if not args.skip_open_msg:
            tts_agent.send_non_block(intro_message)
            # run G1 action if any
            # v8: 如果讲的话很多，添加一些手势
            if len(intro_message) > 20: # 一个中文大概两个字符, 20字符大概一句话
                hand_gesture_choice = random.choice(vla_agent.speaking_hand_gestures)
                print_with_time("[VLAAgent-说话手势]:%s" % hand_gesture_choice)
                time.sleep(3.0) # 等声音开始
                if vla_agent.g1_ctr:
                    vla_agent.g1_func_mapping(hand_gesture_choice)

        # Pygame initialization
        pygame.init()
        # Get actual dimensions from the subscriber once the first frame is received
        current_w, current_h = zmq_subscriber.get_dimensions()
        screen = pygame.display.set_mode((current_w, current_h))
        # Create clock for FPS control
        clock = pygame.time.Clock()
        pygame_font = pygame.font.Font(None, 30)

        gui_frame_count = 0
        start_time_gui = time.time()
        while True:
            current_latency_for_display = 0.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

            if args.show_video:
                frame = zmq_subscriber.read()  # Read from the ZMQ subscriber
                if frame is not None:
                    # Update screen dimensions if they changed (e.g., from default to actual frame size)
                    current_w, current_h = zmq_subscriber.get_dimensions()
                    if (current_w, current_h) != screen.get_size():
                        screen = pygame.display.set_mode((current_w, current_h))
                        print(f"Resizing Pygame window to {current_w}x{current_h}")

                    current_latency_for_display = zmq_subscriber.current_latency_ms

                    # Convert to RGB and transpose for pygame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    surf = pygame.surfarray.make_surface(np.transpose(rgb_frame, (1, 0, 2)))
                    screen.blit(surf, (0, 0))

                    # --- Display FPS and Latency Info ---
                    gui_frame_count += 1
                    current_time_gui = time.time()
                    if (current_time_gui - start_time_gui) > 0:
                        received_fps = zmq_subscriber.current_frame_count / (current_time_gui - start_time_zmq)
                        gui_fps = int(gui_frame_count / (current_time_gui - start_time_gui))
                    else:
                        received_fps = 0
                        gui_fps = 0

                    # Render FPS text
                    fps_text_line1 = pygame_font.render(f"Received FPS: {received_fps:.1f}", True, (0, 255, 0))  # Green
                    fps_text_line2 = pygame_font.render(f"GUI FPS: {gui_fps} (Limit: {args.display_fps_limit})", True, (255, 255, 0))  # Yellow
                    # Render Latency text
                    latency_text = pygame_font.render(f"Latency: {current_latency_for_display:.2f} ms", True, (255, 0, 0))  # Red

                    screen.blit(fps_text_line1, (10, 10))
                    screen.blit(fps_text_line2, (10, 40))
                    screen.blit(latency_text, (10, 70))  # Position below FPS info

                    pygame.display.flip()

                    pygame.display.flip()
                else:

                    time.sleep(0.05)  # Prevent busy-waiting if no frames

            # Control the GUI display FPS
            clock.tick(args.display_fps_limit)

    except KeyboardInterrupt:
        print("Shutting down due to Ctrl+C")
    finally:
        if zmq_subscriber:  # Ensure to stop the ZMQ subscriber
            zmq_subscriber.stop()
        cv2.destroyAllWindows()  # No longer strictly needed for display, but good for cleanup
        pygame.quit()
