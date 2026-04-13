"""
TestSimulation.py
-----------------
Connects to the Udacity self-driving car simulator in Autonomous Mode via
Flask-SocketIO and drives the car using the trained Nvidia CNN model.

Usage
-----
1. Activate your virtual environment:
       venv\Scripts\activate
2. Run:
       python TestSimulation.py
3. Launch beta_simulator.exe → choose a track → click "Autonomous Mode"

The server listens on port 4567. The simulator connects automatically.

Compatible package versions (must match virtual environment):
    flask             == 1.1.2
    flask-socketio    == 3.3.1
    python-socketio   == 4.2.1
    python-engineio   == 3.8.2.post1
    eventlet          == 0.25.1
    tensorflow        == 2.3.0
"""

import os
print('Setting Up ...')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TF info/warning logs

import base64
import cv2
import eventlet
import numpy as np
import socketio
from flask import Flask
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = 'model.h5'
MAX_SPEED  = 10     # speed cap used in throttle formula (km/h)


# ─────────────────────────────────────────────────────────────────────────────
# Flask + SocketIO setup  (python-socketio 4.x / flask-socketio 3.x style)
# ─────────────────────────────────────────────────────────────────────────────
sio = socketio.Server()
app = Flask(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing  (must exactly match the preprocessing used in training)
# Pipeline: crop → RGB→YUV → Gaussian blur → resize 200×66 → normalize
# NOTE: PIL.Image gives RGB, so we convert RGB→YUV (not BGR→YUV)
# ─────────────────────────────────────────────────────────────────────────────
def preProcessing(img: np.ndarray) -> np.ndarray:
    """
    Preprocess a raw simulator image before feeding it to the model.

    Parameters
    ----------
    img : np.ndarray  RGB image decoded from PIL (simulator sends RGB JPEG)

    Returns
    -------
    np.ndarray  shape (66, 200, 3)  float32  normalized to [0, 1]
    """
    img = img[60:135, :, :]                          # crop sky + hood
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)       # RGB → YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)           # noise reduction
    img = cv2.resize(img, (200, 66))                 # Nvidia input size
    img = img / 255                                  # normalize to [0, 1]
    return img


# ─────────────────────────────────────────────────────────────────────────────
# SocketIO event handlers
# ─────────────────────────────────────────────────────────────────────────────
@sio.on('telemetry')
def telemetry(sid, data):
    """
    Called on every simulator frame.
    Receives camera image + speed → predicts steering → sends control back.
    """
    speed = float(data['speed'])

    # Decode base64 JPEG → PIL → numpy (RGB)
    image = Image.open(BytesIO(base64.b64decode(data['image'])))
    image = np.asarray(image)

    # Preprocess + add batch dimension
    image = preProcessing(image)
    image = np.array([image])

    # Predict steering angle
    steering = float(model.predict(image))

    # Simple speed regulator: apply more throttle when slow
    throttle = 1.0 - speed / MAX_SPEED

    print(f'Throttle: {throttle:.4f} | Steering: {steering:+.4f} | Speed: {speed:.1f}')
    sendControl(steering, throttle)


@sio.on('connect')
def connect(sid, environ):
    print('Connected to simulator')
    sendControl(0, 0)


# ─────────────────────────────────────────────────────────────────────────────
# Send steering + throttle command to simulator
# ─────────────────────────────────────────────────────────────────────────────
def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle':       throttle.__str__()
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    print(f'[INFO] Model loaded from {MODEL_PATH}')
    print('[INFO] Server starting on port 4567 …')
    print('[INFO] Launch the Udacity simulator and select Autonomous Mode.')

    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
