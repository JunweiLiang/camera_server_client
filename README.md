### Note for simple zmq-web-socket-based camera streaming server-client

+ Start camera server:
```
    $ python3 rgb_zmq_publisher.py --fps 10 --h 720 --w 1280 --cam_num 0 --port 5555
```

+ Start camera client:
```
    $ python rgb_zmq_sub_pygame_allthreads.py --publisher_ip 127.0.0.1 --publisher_port 5555 --show_video
```

### Note for Text-to-speech (TTS) server client

1. TTS server: the example server code is in `cosyvoice_server_vllm_junwei.py`. It uses FastAPI to setup a TTS http server.
2. TTS client: then we can use this code `cosyvoice_client_vllm_junwei_speak.py` to query the above http server and get audio response.

### Note for VLM server client
1. The VLM (Qwen2.5-VL, etc.) is setup using standard [vLLM](https://docs.vllm.ai/en/latest/) tool.
2. The VLM client: In this code `start_vlm_speech_g1_agent_v8.py`, starting from line 775, it sends message to the vLLM server to get responses from VLM. From line 511, it sends request to TTS server
