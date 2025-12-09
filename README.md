### Note for simple zmq-web-socket-based camera streaming server-client

+ Start camera server:
```
    $ python3 rgb_zmq_publisher.py --fps 10 --h 720 --w 1280 --cam_num 0 --port 5555
```

+ Start camera client:
```
    $ python rgb_zmq_sub_pygame_allthreads.py --publisher_ip 127.0.0.1 --publisher_port 5555 --show_video
```
