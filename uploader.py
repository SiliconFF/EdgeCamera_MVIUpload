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
from logging.handlers import RotatingFileHandler
import signal
import tenacity
import platform

# Optional imports with graceful fallbacks
try:
    from picamera2 import Picamera2  # Raspberry Pi specific
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    Picamera2 = None

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

# --------------------- Setup Logging ---------------------
system = platform.system()
if system == "Windows":
    log_path = os.path.join(os.getenv('APPDATA', ''), 'camera_edge.log')
else:  # Linux / Raspberry Pi
    log_path = '/var/log/camera_edge.log'

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler('uploader.log', maxBytes=1024 * 1024 * 5, backupCount=1) # 5 MB max size, keep 5 backups
    ]
)
logger = logging.getLogger(__name__)

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

# Host platform from config
host_platform = config.get('host-platform', 'RPI').upper()
if host_platform not in ['WINDOWS', 'LINUX', 'RPI']:
    logger.error(f"Invalid host-platform: {host_platform} (must be WINDOWS, LINUX, or RPI)")
    sys.exit(1)

actual_platform = platform.system()
if host_platform == 'WINDOWS' and actual_platform != 'Windows':
    logger.warning("Config specifies WINDOWS but running on non-Windows OS")
elif host_platform in ['LINUX', 'RPI'] and actual_platform != 'Linux':
    logger.warning("Config specifies LINUX/RPI but running on non-Linux OS")

# Platform-specific OpenCV backend
if host_platform == 'WINDOWS':
    OPENCV_BACKEND = cv2.CAP_DSHOW
    DEFAULT_CAMERA_TYPE = 'USB'
elif host_platform == 'LINUX':
    OPENCV_BACKEND = cv2.CAP_ANY
    DEFAULT_CAMERA_TYPE = 'USB'
else:  # RPI
    OPENCV_BACKEND = cv2.CAP_V4L2
    DEFAULT_CAMERA_TYPE = 'USB'

# Credentials (prefer environment variables)
mvi_username = os.environ.get('MVI_USERNAME', config.get('mvi-username', '').strip())
mvi_password = os.environ.get('MVI_PASSWORD', config.get('mvi-password', '').strip())

# MVI Config
mvi_endpoint_base = config.get('mvi-edge-endpoint', '').strip()
mvi_device_uuid = config.get('mvi-device-uuid', '').strip()
if not mvi_endpoint_base or not mvi_username or not mvi_password:
    logger.error("Missing MVI endpoint or credentials")
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
    logger.error("Missing MQTT broker or topic")
    sys.exit(1)

if mqtt_tls_required and not mqtt_tls_file:
    logger.error("TLS required but no CA cert file specified")
    sys.exit(1)

# Camera Config
camera_type = config.get('camera-type', DEFAULT_CAMERA_TYPE).upper()
camera_ip = config.get('camera-ip', '').strip() if camera_type == 'RTSP' else None
camera_device = config.get('camera-device', None)
camera_width = int(config.get('camera-width', 1920))
camera_height = int(config.get('camera-height', 1080))
gamma = float(config.get('gamma', 1.5))
warm_up_frames = int(config.get('warm-up-frames', 15))
keep_alive_interval = int(config.get('keep-alive-interval', 300))

# SSL Verification
mvi_ca_cert = config.get('mvi-ca-cert', None)
verify_ssl = mvi_ca_cert if mvi_ca_cert else False

# Validate camera type
allowed_types = ['USB', 'RTSP']
if host_platform == 'RPI' and PICAMERA_AVAILABLE:
    allowed_types.append('PICAM')

if camera_type not in allowed_types:
    logger.error(f"Invalid camera-type '{camera_type}' for platform {host_platform}")
    sys.exit(1)

if camera_type == 'RTSP':
    if not camera_ip:
        logger.error("RTSP selected but no camera-ip provided")
        sys.exit(1)
    if not camera_ip.startswith('rtsp://'):
        camera_ip = f'rtsp://{camera_ip}'

# --------------------- Authentication ---------------------
session_url = f"https://{mvi_endpoint_base}/users/sessions"
device_endpoint = f"https://{mvi_endpoint_base}/devices/images?uuid={mvi_device_uuid}"
keep_alive_url = f"https://{mvi_endpoint_base}/users/sessions/keepalive"

@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=4, max=10))
def authenticate():
    data = {"grant_type": "password", "password": mvi_password, "user": mvi_username}
    response = requests.post(session_url, json=data, verify=verify_ssl)
    response.raise_for_status()
    return response.json()['token']

try:
    token = authenticate()
    logger.info("Authenticated successfully with MVI")
except Exception as e:
    logger.error(f"Authentication failed after retries: {e}")
    sys.exit(1)

# --------------------- USB Camera Discovery ---------------------
def find_working_camera(max_index=10, timeout_sec=2.0):
    logger.info("Searching for a working USB camera...")
    for index in range(max_index):
        cap = cv2.VideoCapture(index, OPENCV_BACKEND)
        if not cap.isOpened():
            continue
        start = time.time()
        while time.time() - start < timeout_sec:
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                logger.info(f"Found working camera at index {index}")
                return index
            time.sleep(0.05)
        cap.release()
    logger.error("No working USB camera found")
    return None

# Determine source
if camera_type == 'USB':
    video_src = camera_device if camera_device is not None else find_working_camera()
    if video_src is None:
        sys.exit(1)
elif camera_type == 'RTSP':
    video_src = camera_ip
else:  # PICAM
    video_src = None

# --------------------- Upload (Lossless PNG) ---------------------
@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(2))
def upload_frame_in_memory(frame, destination):
    success, buffer = cv2.imencode('.png', frame)
    if not success:
        raise ValueError("PNG encoding failed")
    img_io = BytesIO(buffer)
    headers = {"mvie-controller": token, "accept": "application/json"}
    files = {"file": ("captured_frame.png", img_io, "image/png")}
    response = requests.post(destination, headers=headers, files=files, verify=verify_ssl)
    response.raise_for_status()
    logger.info("Frame uploaded successfully")

# --------------------- FrameGrabber Class ---------------------
class FrameGrabber:
    def __init__(self, src, width, height, camera_type):
        self.camera_type = camera_type
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.consecutive_failures = 0
        self.failure_threshold = 10  # Trigger restart after 10 failed reads (~1-2 seconds if sleeping 0.1s)
        
        self._initialize_camera(src, width, height)
        
        # Warm-up
        logger.info("Warming up camera...")
        for _ in range(warm_up_frames):
            self._read_frame()
            time.sleep(0.1)

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        logger.info("FrameGrabber thread started")

    def _initialize_camera(self, src, width, height):
        if self.camera_type == 'PICAM':
            if not PICAMERA_AVAILABLE:
                raise ImportError("picamera2 is not available")
            self.picam = Picamera2()
            config = self.picam.create_video_configuration(main={"size": (width, height)})
            self.picam.configure(config)
            self.picam.start()
            logger.info(f"PiCamera initialized at {width}x{height}")
        else:
            backend = OPENCV_BACKEND if self.camera_type == 'USB' else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(src, backend)
            if not self.cap.isOpened():
                raise IOError(f"Failed to open video source: {src}")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Requested {width}x{height} | Actual {actual_w}x{actual_h}")

    def _read_frame(self):
        if self.camera_type == 'PICAM':
            return self.picam.capture_array()
        else:
            ret, frame = self.cap.read()
            return frame if ret else None

    def _reopen_camera(self, src, width, height):
        logger.info("Attempting to reopen camera...")
        if self.camera_type == 'PICAM':
            self.picam.stop()
            self.picam.close()  # Ensure clean close
        else:
            self.cap.release()
        time.sleep(1)  # Give hardware time to settle
        self._initialize_camera(src, width, height)
        self.consecutive_failures = 0

    def _update(self):
        while self.running:
            try:
                if self.camera_type != 'PICAM' and not self.cap.isOpened():
                    raise IOError("Camera capture is not opened")
                
                frame = self._read_frame()
                if frame is not None:
                    with self.lock:
                        self.latest_frame = frame.copy()
                    self.consecutive_failures = 0
                else:
                    raise ValueError("Frame read returned None")
            except Exception as e:
                self.consecutive_failures += 1
                logger.warning(f"Frame read failed ({self.consecutive_failures}/{self.failure_threshold}): {e}")
                if self.consecutive_failures >= self.failure_threshold:
                    try:
                        self._reopen_camera(video_src, camera_width, camera_height)
                    except Exception as reopen_e:
                        logger.error(f"Reopen failed: {reopen_e} - retrying after delay")
                        time.sleep(5)  # Longer backoff for reopen failures
                else:
                    time.sleep(0.1)
    
    def get_latest_frame(self):
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
        if frame is not None:
            # Optional: Check for black frame (mean intensity < 5)
            if np.mean(frame) < 5:
                logger.warning("Detected potential black frame - treating as invalid")
                return None
        return frame

    def stop(self):
        self.running = False
        if self.camera_type == 'PICAM':
            if hasattr(self, 'picam'):
                self.picam.stop()
                self.picam.close()
        else:
            if hasattr(self, 'cap'):
                self.cap.release()

grabber = FrameGrabber(video_src, camera_width, camera_height, camera_type)
logger.info(f"Using {camera_type} camera at {camera_width}x{camera_height}")

# --------------------- Image Processing ---------------------
def brighten_frame(frame, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(frame, table)

# --------------------- Capture and Upload ---------------------
@tenacity.retry(stop=tenacity.stop_after_attempt(5), wait=tenacity.wait_exponential(multiplier=1, min=2, max=10), reraise=True)
def restart_grabber():
    global grabber
    grabber.stop()
    time.sleep(1)
    grabber = FrameGrabber(video_src, camera_width, camera_height, camera_type)

def capture_and_upload():
    global grabber
    frame = grabber.get_latest_frame()
    if frame is None:
        logger.warning("No fresh frame available - recovering camera...")
        try:
            restart_grabber()
            # After restart, try to grab a new frame immediately
            time.sleep(0.5)  # Short wait for warm-up
            frame = grabber.get_latest_frame()
            if frame is None:
                logger.error("Recovery failed - skipping upload")
                return
        except Exception as e:
            logger.error(f"Camera recovery failed after retries: {e}")
            return  # Don't crash the script
    
    #frame = brighten_frame(frame, gamma)
    logger.info(f"Frame captured - shape: {frame.shape}")
    try:
        upload_frame_in_memory(frame, device_endpoint)
    except Exception as e:
        logger.error(f"Upload failed: {e}")

# --------------------- MQTT Listener ---------------------
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        logger.info("MQTT connected successfully")
        client.subscribe(mqtt_topic)
    else:
        logger.warning(f"MQTT connect failed: {reason_code}")

def on_disconnect(client, userdata, flags, reason_code, properties):
    logger.warning(f"MQTT disconnected ({reason_code}) - will reconnect automatically")

def on_message(client, userdata, message):
    logger.info(f"MQTT message on {message.topic}: {message.payload.decode()}")
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
    logger.info(f"MQTT client started for {mqtt_broker}:{mqtt_port}")

mqtt_thread = threading.Thread(target=mqtt_listener, daemon=True)
mqtt_thread.start()

# --------------------- Keep-Alive ---------------------
def keep_alive():
    global token
    headers = {"mvie-controller": token, "accept": "application/json"}
    while True:
        try:
            response = requests.get(keep_alive_url, headers=headers, verify=verify_ssl, timeout=10)
            if response.status_code == 401:
                logger.warning("Session expired - re-authenticating...")
                token = authenticate()
                headers["mvie-controller"] = token
            else:
                response.raise_for_status()
            logger.info(f"Keep-alive successful ({response.status_code})")
        except Exception as e:
            logger.error(f"Keep-alive failed: {e}")
        time.sleep(keep_alive_interval)

keep_alive_thread = threading.Thread(target=keep_alive, daemon=True)
keep_alive_thread.start()
logger.info(f"Keep-alive thread started (interval: {keep_alive_interval}s)")

# --------------------- Config Reload (Optional) ---------------------
observer = None
if WATCHDOG_AVAILABLE:
    class ConfigHandler(FileSystemEventHandler):
        def on_modified(self, event):
            if event.src_path.endswith(CONFIG_FILE):
                logger.info("Config file changed - reloading settings...")
                global config, gamma, warm_up_frames, keep_alive_interval
                try:
                    config = load_config()
                    gamma = float(config.get('gamma', 1.5))
                    warm_up_frames = int(config.get('warm-up-frames', 15))
                    keep_alive_interval = int(config.get('keep-alive-interval', 300))
                    logger.info("Config reloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to reload config: {e}")

    observer = Observer()
    observer.schedule(ConfigHandler(), path=CONFIG_DIR, recursive=False)
    observer.start()
    logger.info("Config file watcher started")
else:
    logger.info("watchdog not available - config reload disabled")

# --------------------- Graceful Shutdown ---------------------
def shutdown_handler(signum, frame):
    logger.info("Shutdown signal received - stopping gracefully...")
    if observer:
        observer.stop()
    grabber.stop()
    logger.info("Shutdown complete")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)

# --------------------- Main Loop ---------------------
logger.info("All components initialized. Waiting for MQTT triggers...")
last_health_check = time.time()
health_check_interval = 30  # seconds

while True:
    time.sleep(1)
    if time.time() - last_health_check > health_check_interval:
        frame = grabber.get_latest_frame()
        if frame is None:
            logger.info("Periodic health check failed - triggering recovery")
            capture_and_upload()  # Reuse the function (it won't upload, just recover)
        last_health_check = time.time()
