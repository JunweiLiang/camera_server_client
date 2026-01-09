# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import requests
import torch
import torchaudio
import numpy as np
import time
import sounddevice as sd
import threading

devices = sd.query_devices()
if len(devices) == 0:
    print("No microphone devices found")
    sys.exit(0)
print(devices)
input_device_idx = sd.default.device[0]
print("Use this mic: %s" % devices[input_device_idx]["name"])


audio_data_buffer = None # Stores the current audio segment to play
current_frame = 0 # Tracks playback position within audio_data_buffer
stop_playback_event = None

def _audio_playback_callback(outdata, frames, time_info, status):
    """
    Sounddevice callback for feeding audio data.
    This runs in an internal sounddevice thread.
    """
    global audio_data_buffer, stop_playback_event, current_frame

    # 1. Handle potential errors reported by sounddevice
    if status:
        if status.output_underflow:
            print('TTSAgent Callback: Output underflow! Consider increasing blocksize or processing speed.', file=sys.stderr)
        else:
            print(f"TTSAgent Callback Status: {status}", file=sys.stderr)
        outdata.fill(0)
        return sd.CallbackStop

    # 2. Check for external stop signal (highest priority)
    if stop_playback_event.is_set():
        outdata.fill(0)
        return sd.CallbackStop

    # 3. Determine how many frames we can actually provide from our buffer
    num_frames_to_copy = min(frames, len(audio_data_buffer) - current_frame)

    # 4. Get the chunk of data from our buffer
    chunk = audio_data_buffer[current_frame : current_frame + num_frames_to_copy]

    # 5. Copy the chunk to outdata
    outdata[:num_frames_to_copy] = chunk

    # 6. Pad the rest of the outdata buffer with zeros if we didn't fill it
    if num_frames_to_copy < frames:
        outdata[num_frames_to_copy:].fill(0)

    # 7. Advance frame pointer by the amount copied
    current_frame += num_frames_to_copy

    # 8. Explicitly check if all audio data has been sent
    if current_frame >= len(audio_data_buffer):
        #print("returned sd.CallbackStop") # Debug print to confirm it's reached
        stop_playback_event.set() # 必须用这个才能停止
        return sd.CallbackStop # Signal that all data has been played

    # 9. If we reached here, it means more data is still available
    return sd.CallbackFlags(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default='127.0.0.1')
    parser.add_argument('--port',
                        type=int,
                        default='50000')

    parser.add_argument('--tts_text',
                        type=str,
                        default='收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。')


    args = parser.parse_args()

    url = "http://{}:{}/inference_zero_shot_fast".format(args.host, args.port)

    target_sr = 24000  # cosyvoice.sample_rate

    payload = {
        'tts_text': args.tts_text,
    }

    # init
    audio_stream = sd.OutputStream(
        samplerate=target_sr,
        channels=1,
        dtype=np.int16,
        callback=_audio_playback_callback, # Our custom callback
        #finished_callback=self.stop_playback_event.set,
    )
    stop_playback_event = threading.Event()

    start_time = time.perf_counter()
    response = requests.request("GET", url, data=payload, stream=True)

    tts_audio = b''
    for r in response.iter_content(chunk_size=16000):
        tts_audio += r
    #tts_speech = torch.from_numpy(np.array(np.frombuffer(tts_audio, dtype=np.int16))).unsqueeze(dim=0)
    #print(tts_speech.shape) #torch.Size([1, 287040]) # this is for torchaudio.save()
    tts_speech = np.array(np.frombuffer(tts_audio, dtype=np.int16))

    audio_data_buffer = tts_speech.reshape(-1, 1) # (N,) -> (N, 1)


    current_frame = 0 # Reset frame counter for new audio segment
    stop_playback_event.clear()

    #06/2025 换成用sounddevice.OutputStream才能发音，请查看start_vlm_speech_g1_agent_v6_zmq_tts_api.py
    #sd.play(tts_speech, target_sr) # 24000 Hz
    print("took %.3f seconds to first voice." % (time.perf_counter() - start_time))
    audio_stream.start()
    while True:
        #print_with_time("playback event waiting...")
        if not audio_stream.active or stop_playback_event.is_set():
            break
        time.sleep(0.05)

    audio_stream.close()



