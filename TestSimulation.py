"""
TestSimulation.py
-----------------
Connects to the Udacity self-driving car simulator in Autonomous Mode via
Flask-SocketIO and drives the car using the trained Nvidia CNN model.

Usage
-----
1. Activate your virtual environment
2. Run:  python TestSimulation.py
3. Launch beta_simulator.exe → select a track → click "Autonomous Mode"

The server listens on port 4567.  The simulator connects automatically.

Package versions (must match virtual environment):
    flask             == 1.1.2
    flask-socketio    == 3.3.1
    python-socketio   == 4.2.1
    python-engineio   == 3.8.2
    eventlet          == 0.25.1
"""

import base64
import os

import cv2
import eventlet
import numpy as np
import socketio
from flask import Flask
from io import BytesIO
from PIL import Image
from keras.models import load_model

from preprocess import preprocess_image


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'models/model.h5'
MAX_SPEED    = 25    # km/h cruise target
MIN_SPEED    = 10    # km/h minimum before applying extra throttle
SPEED_LIMIT  = 30    # km/h hard cap used in throttle formula


# ─────────────────────────────────────────────────────────────────────────────
# Flask + SocketIO setup
# Compatible with python-socketio 4.x  /  flask-socketio 3.x
# ─────────────────────────────────────────────────────────────────────────────
sio  = socketio.Server()
app  = Flask(__name__)
app.wsgi_app = socketio.Middleware(sio, app.wsgi_app)


# ─────────────────────────────────────────────────────────────────────────────
# Load the trained model once at startup
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f'Trained model not found at "{MODEL_PATH}". '
        'Run train.py first to generate the model.'
    )

model = load_model(MODEL_PATH)
print(f'[INFO] Model loaded from {MODEL_PATH}')


# ─────────────────────────────────────────────────────────────────────────────
# Helper — decode base64 image sent by the simulator
# ─────────────────────────────────────────────────────────────────────────────
def decode_image(image_b64: str) -> np.ndarray:
    """Convert simulator base-64 JPEG string → BGR numpy array."""
    img_bytes = base64.b64decode(image_b64)
    img_pil   = Image.open(BytesIO(img_bytes))
    img_rgb   = np.array(img_pil)
    img_bgr   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr


# ─────────────────────────────────────────────────────────────────────────────
# SocketIO event handlers
# ─────────────────────────────────────────────────────────────────────────────
@sio.on('connect')
def on_connect(sid, environ):
    print(f'[INFO] Simulator connected  (sid={sid})')
    send_control(0.0, 0.0)   # send neutral command on connect


@sio.on('disconnect')
def on_disconnect(sid):
    print(f'[INFO] Simulator disconnected (sid={sid})')


@sio.on('telemetry')
def on_telemetry(sid, data):
    """
    Received on every simulator frame.
    data keys: 'image'   – base64 JPEG of front camera
               'speed'   – current vehicle speed (km/h)
               'throttle', 'brake', 'steering_angle' – current actuator values
    """
    if data is None:
        send_control(0.0, 0.0)
        return

    # 1. Decode + preprocess the front-camera image
    img = decode_image(data['image'])
    img = preprocess_image(img)                        # → (66, 200, 3) float32
    img_input = np.expand_dims(img, axis=0)            # → (1, 66, 200, 3)

    # 2. Predict steering angle
    steering_angle = float(model.predict(img_input, verbose=0)[0][0])

    # 3. Compute throttle (simple speed regulator)
    speed = float(data.get('speed', 0))
    throttle = 1.0 - (speed / SPEED_LIMIT) ** 2

    # Clamp values to safe ranges
    steering_angle = float(np.clip(steering_angle, -1.0, 1.0))
    throttle       = float(np.clip(throttle, 0.0, 1.0))

    print(f'  Steering: {steering_angle:+.4f}  |  '
          f'Throttle: {throttle:.4f}  |  Speed: {speed:.1f} km/h')

    send_control(steering_angle, throttle)


# ─────────────────────────────────────────────────────────────────────────────
# Send control command back to the simulator
# ─────────────────────────────────────────────────────────────────────────────
def send_control(steering_angle: float, throttle: float):
    sio.emit('steer', {
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('[INFO] Server starting on port 4567 …')
    print('[INFO] Launch the Udacity simulator and select Autonomous Mode.')
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
