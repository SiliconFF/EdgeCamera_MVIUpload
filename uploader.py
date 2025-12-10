import requests
import time
import cv2
import threading
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion
import ssl
from io import BytesIO
import numpy as np
import yaml
import os
import sys
import urllib3
import logging
import signal
import tenacity
from picamera2 import Picamera2  # For Raspberry Pi Camera support
from watchdog.observers import Observer  # For config reload
from watchdog.events import FileSystemEventHandler

# --------------------- Setup Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('/var/log/camera_edge.log', mode='a'),
        logging.StreamHandler()  # Also log to console for debugging
    ]
)
logger = logging.getLogger(__name__)

# Suppress insecure HTTPS warnings temporarily; will be removed with proper verify
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --------------------- Load Configuration ---------------------
CONFIG_FILE = "camera_edge_config.yaml"
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

def load_config():
    config_path = os.path.join(CONFIG_DIR, CONFIG_FILE)
    if not os.path.exists(config_path):
        logger.error(f"Config file '{config_path}' not found!")
        raise FileNotFoundError

    with open(config_path, 'r') as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
            raise

config = load_config()

# Secure credentials: Prefer env vars if set
mvi_username = os.environ.get('MVI_USERNAME', config.get('mvi-username', '').strip())
mvi_password = os.environ.get('MVI_PASSWORD', config.get('mvi-password', '').strip())

# MVI Config
mvi_endpoint_base = config.get('mvi-edge-endpoint', '').strip()
if not mvi_endpoint_base:
    logger.error("Missing 'mvi-edge-endpoint' in config")
    sys.exit(1)  # Still exit on critical config miss, but can be changed to retry

if not mvi_username or not mvi_password:
    logger.error("Missing MVI username or password")
    sys.exit(1)

# MQTT Config
mqtt_broker = config.get('mqtt-broker', '').strip()
mqtt_port = config.get('mqtt-port', 8883)
mqtt_username = os.environ.get('MQTT_USERNAME', config.get('mqtt-username', ''))
mqtt_password = os.environ.get('MQTT_PASSWORD', config.get('mqtt-password', ''))
mqtt_tls_required = config.get('mqtt-tls-required', True)
mqtt_tls_file = config.get('mqtt-tls-file-name', '').strip()
mqtt_topic = config.get('mqtt-trigger-topic', '').strip()

if not mqtt_broker or not mqtt_topic:
    logger.error("Missing MQTT broker or trigger topic in config")
    sys.exit(1)

if mqtt_tls_required and not mqtt_tls_file:
    logger.error("TLS required but no CA cert file specified")
    sys.exit(1)

# Camera Config
camera_type = config.get('camera-type', 'USB').upper()
camera_ip = config.get('camera-ip', '').strip() if camera_type == 'RTSP' else None
camera_device = config.get('camera-device', None)  # Optional: e.g., '/dev/video0' or index
camera_width = int(config.get('camera-width', 1920))
camera_height = int(config.get('camera-height', 1080))
gamma = float(config.get('gamma', 1.5))  # Configurable gamma
warm_up_frames = int(config.get('warm-up-frames', 15))
keep_alive_interval = int(config.get('keep-alive-interval', 300))  # Seconds

# SSL Verification: Path to CA cert for MVI (assume provided in config)
mvi_ca_cert = config.get('mvi-ca-cert', None)  # e.g., '/path/to/mvi_ca.pem'
verify_ssl = mvi_ca_cert if mvi_ca_cert else False  # Use cert if provided, else warn

if camera_type not in ['USB', 'RTSP', 'PICAM']:
    logger.error(f"Invalid camera-type: {camera_type} (must be USB, RTSP, or PICAM)")
    sys.exit(1)

if camera_type == 'RTSP' and not camera_ip:
    logger.error("camera-type is RTSP but no camera-ip provided")
    sys.exit(1)
elif camera_type == 'RTSP':
    # Ensure full RTSP URL
    if not camera_ip.startswith('rtsp://'):
        camera_ip = f'rtsp://{camera_ip}'  # Basic fix; add auth if needed via config

# --------------------- Authentication with Retry ---------------------
session_url = f"https://{mvi_endpoint_base}/users/sessions"
device_endpoint = f"https://{mvi_endpoint_base}/devices/images?uuid=16c02770-ca42-4404-b9b4-f07419414f4d"
keep_alive_url = f"https://{mvi_endpoint_base}/users/sessions/keepalive"

@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
def authenticate():
    data = {
        "grant_type": "password",
        "password": mvi_password,
        "user": mvi_username
    }
    response = requests.post(session_url, json=data, verify=verify_ssl)
    response.raise_for_status()
    return response.json()['token']

try:
    token = authenticate()
    logger.info("Successfully authenticated")
except Exception as e:
    logger.error(f"Authentication failed after retries: {e}")
    sys.exit(1)

# --------------------- Find Working USB Camera ---------------------
def find_working_camera(max_index=10, timeout_sec=2.0):
    logger.info("Searching for a working USB camera...")
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)  # Use V4L2 for Raspberry Pi/Linux
        if not cap.isOpened():
            logger.warning(f"Index {index}: Not accessible (not opened)")
            cap.release()
            continue
        
        start_time = time.time()
        success = False
        while time.time() - start_time < timeout_sec:
            ret, frame = cap.read()
            if ret and frame is not None:
                success = True
                break
            time.sleep(0.05)
        
        if success:
            logger.info(f"Found working USB camera at index {index}")
            cap.release()
            return index
        else:
            logger.warning(f"Index {index}: Opened but failed to grab frame")
        
        cap.release()
    
    logger.error(f"No working USB camera found after checking indices 0-{max_index-1}")
    return None

# Determine video source
if camera_type == 'USB':
    video_src = camera_device if camera_device is not None else find_working_camera(max_index=10)
    if video_src is None:
        logger.error("No working USB camera detected.")
        sys.exit(1)
elif camera_type == 'RTSP':
    video_src = camera_ip
elif camera_type == 'PICAM':
    video_src = None  # Handled separately in FrameGrabber

# --------------------- Upload Function (In-Memory) with Retry ---------------------
@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
def upload_frame_in_memory(frame, destination):
    success, buffer = cv2.imencode('.png', frame)  # Changed to PNG for lossless
    if not success:
        logger.error("Failed to encode frame to PNG")
        raise ValueError("Encoding failed")

    img_io = BytesIO(buffer)
    img_io.seek(0)

    headers = {
        "mvie-controller": token,
        "accept": "application/json"
    }
    files = {
        "file": ("captured_frame.png", img_io, "image/png")
    }
    response = requests.post(destination, headers=headers, files=files, verify=verify_ssl)
    response.raise_for_status()
    logger.info("Frame uploaded successfully (in-memory)")

# --------------------- Frame Grabber Class ---------------------
class FrameGrabber:
    def __init__(self, src, width, height, camera_type):
        self.camera_type = camera_type
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True

        if self.camera_type == 'PICAM':
            self.picam = Picamera2()
            config = self.picam.create_video_configuration(main={"size": (width, height)})
            self.picam.configure(config)
            self.picam.start()
            logger.info(f"PiCamera initialized at {width}x{height}")
        else:
            self.cap = cv2.VideoCapture(src, cv2.CAP_V4L2 if camera_type == 'USB' else cv2.CAP_ANY)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video source: {src}")
                raise IOError("Camera open failed")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Requested resolution: {width}x{height} | Actual: {actual_w}x{actual_h}")

        # Warm up
        logger.info("Warming up camera...")
        for _ in range(warm_up_frames):
            self._read_frame()  # Discard initial frames
            time.sleep(0.1)

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("FrameGrabber thread started")

    def _read_frame(self):
        if self.camera_type == 'PICAM':
            return self.picam.capture_array()
        else:
            ret, frame = self.cap.read()
            if ret:
                return frame
            return None

    def _update(self):
        while self.running:
            frame = self._read_frame()
            if frame is not None:
                with self.lock:
                    self.latest_frame = frame.copy()
            else:
                logger.warning("Failed to read frame - retrying...")
                time.sleep(0.1)

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        self.running = False
        if self.camera_type == 'PICAM':
            self.picam.stop()
        else:
            self.cap.release()

# Initialize FrameGrabber
grabber = FrameGrabber(video_src, camera_width, camera_height, camera_type)
logger.info(f"Using {camera_type} camera at {camera_width}x{camera_height}")

# --------------------- Image Processing ---------------------
def brighten_frame(frame, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)

# --------------------- Capture and Upload Function ---------------------
def capture_and_upload():
    frame = grabber.get_latest_frame()
    if frame is None:
        logger.warning("No fresh frame available - attempting camera recovery...")
        grabber.stop()
        time.sleep(1)
        global grabber
        grabber = FrameGrabber(video_src, camera_width, camera_height, camera_type)
        return

    frame = brighten_frame(frame, gamma)
    logger.info(f"Frame captured successfully - shape: {frame.shape}")
    try:
        upload_frame_in_memory(frame, device_endpoint)
    except Exception as e:
        logger.error(f"Upload failed: {e}")

# --------------------- MQTT Listener with Reconnect ---------------------
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        logger.info("MQTT connected successfully")
        client.subscribe(mqtt_topic)
    else:
        logger.warning(f"MQTT connect failed with code {reason_code}")

def on_disconnect(client, userdata, flags, reason_code, properties):
    logger.warning(f"MQTT disconnected (code {reason_code}). Reconnecting...")
    client.reconnect()

def on_message(client, userdata, message):
    logger.info(f"Message received on topic {message.topic}: {message.payload.decode()}")
    capture_and_upload()

def mqtt_listener():
    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    if mqtt_username and mqtt_password:
        client.username_pw_set(mqtt_username, mqtt_password)

    if mqtt_tls_required:
        tls_path = os.path.join(CONFIG_DIR, mqtt_tls_file)
        if not os.path.exists(tls_path):
            logger.error(f"TLS CA cert not found: {tls_path}")
            sys.exit(1)
        client.tls_set(ca_certs=tls_path, cert_reqs=ssl.CERT_REQUIRED, tls_version=ssl.PROTOCOL_TLS)

    client.connect_async(mqtt_broker, port=mqtt_port, keepalive=60)
    client.loop_start()
    logger.info(f"Started MQTT client for broker {mqtt_broker}:{mqtt_port}")

# Start MQTT
mqtt_thread = threading.Thread(target=mqtt_listener, daemon=True)
mqtt_thread.start()

# --------------------- Keep-Alive Function with Token Refresh ---------------------
def keep_alive():
    headers = {
        "mvie-controller": token,
        "accept": "application/json"
    }
    while True:
        try:
            response = requests.get(keep_alive_url, headers=headers, verify=verify_ssl, timeout=10)
            if response.status_code == 401:  # Unauthorized - refresh token
                logger.warning("Session expired - re-authenticating...")
                global token
                token = authenticate()
                headers["mvie-controller"] = token
            else:
                response.raise_for_status()
            logger.info(f"Keep-alive ping sent (status: {response.status_code})")
        except Exception as e:
            logger.error(f"Keep-alive ping failed: {e}")
        time.sleep(keep_alive_interval)

keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()
logger.info(f"Keep-alive started - pinging every {keep_alive_interval} seconds")

# --------------------- Config Reload Handler ---------------------
class ConfigHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith(CONFIG_FILE):
            logger.info("Config file modified - reloading...")
            global config, gamma, warm_up_frames, keep_alive_interval
            try:
                config = load_config()
                gamma = float(config.get('gamma', 1.5))
                warm_up_frames = int(config.get('warm-up-frames', 15))
                keep_alive_interval = int(config.get('keep-alive-interval', 300))
                # Note: Some changes (e.g., resolution, camera_type) may require restart
                logger.info("Config reloaded successfully")
            except Exception as e:
                logger.error(f"Config reload failed: {e}")

observer = Observer()
observer.schedule(ConfigHandler(), path=CONFIG_DIR, recursive=False)
observer.start()
logger.info("Config file watcher started")

# --------------------- Graceful Shutdown ---------------------
def shutdown_handler(signum, frame):
    logger.info("Received shutdown signal - stopping gracefully...")
    observer.stop()
    grabber.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# --------------------- Main Loop ---------------------
logger.info("All components started. Waiting for MQTT triggers...")
while True:
    time.sleep(1)  # Idle loop; can be replaced with Event.wait() if needed
