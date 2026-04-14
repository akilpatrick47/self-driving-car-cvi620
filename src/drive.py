import os
import base64
from io import BytesIO
from typing import Any

import eventlet
import numpy as np
import socketio
from flask import Flask
from PIL import Image
from tensorflow.keras.models import load_model

print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from src.preprocess import preprocess_image

sio = socketio.Server()
app = Flask(__name__)  # __main__
MAX_SPEED = 10


@sio.on('telemetry')
def telemetry(sid: str, data: dict[str, Any]) -> None:
    """Receive simulator telemetry, run inference, and emit controls."""
    speed = float(data['speed'])
    frame_rgb = Image.open(BytesIO(base64.b64decode(data['image'])))
    frame_array = np.asarray(frame_rgb)
    frame_processed = preprocess_image(frame_array)
    model_input = np.array([frame_processed])
    steering = float(model.predict(model_input))
    throttle = 1.0 - speed / MAX_SPEED
    print(f'{throttle}, {steering}, {speed}')
    send_control(steering, throttle)


@sio.on('connect')
def connect(sid: str, environ: dict[str, Any]) -> None:
    """Initialize steering/throttle state when a client connects."""
    print('Connected')
    send_control(0, 0)


def send_control(steering: float, throttle: float) -> None:
    """Emit steering and throttle commands to the simulator."""
    sio.emit(
        'steer',
        data={
            'steering_angle': steering.__str__(),
            'throttle': throttle.__str__(),
        },
    )


if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
