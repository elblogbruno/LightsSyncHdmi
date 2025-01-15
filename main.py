import io
import cv2
import numpy as np
import os
import dotenv
import time
import threading
import argparse
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from api import CustomAPIClient, CustomWebsocketClient
from color_algorithm import get_dominant_color_average, get_dominant_color_kmeans, smooth_color, get_dominant_color_median, get_dominant_color_mode, calculate_ww_values
from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles

import asyncio
import concurrent.futures
from functools import lru_cache
import logging
from datetime import datetime

dotenv.load_dotenv()

# Configurar logging
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file = os.path.join(log_directory, f"smartroom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class VideoCapture:
    def __init__(self, device_id=0):
        self.logger = logging.getLogger(f"{__name__}.VideoCapture")

        self.device_id = device_id
        self.lock = threading.Lock()
        self._frame = None
        self._ret = False
        self.running = True
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_frame_time = time.time()
        self.init_capture()
        
        # Iniciar thread de captura
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def init_capture(self):
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            if not self.cap.isOpened():
                raise Exception("No se pudo abrir la captura de video")
                
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.reconnect_attempts = 0
            self.logger.info("Captura de video inicializada correctamente")
        except Exception as e:
            self.logger.error(f"Error al inicializar captura: {e}")
            self.reconnect_attempts += 1
            if self.reconnect_attempts < self.max_reconnect_attempts:
                self.logger.info(f"Reintentando... ({self.reconnect_attempts}/{self.max_reconnect_attempts})")
                time.sleep(2)
                self.init_capture()
            else:
                raise Exception("No se pudo inicializar la captura después de múltiples intentos")

    def _capture_loop(self):
        while self.running:
            try:
                if not self.cap.isOpened():
                    self.logger.warning("Conexión perdida, reintentando...")
                    self.init_capture()
                    continue

                ret, frame = self.cap.read()
                
                # Verificar si el frame es válido
                if ret and frame is not None and frame.size > 0:
                    with self.lock:
                        self._ret = ret
                        self._frame = frame
                        self.last_frame_time = time.time()
                else:
                    # Si no hay frame válido por más de 5 segundos, reiniciar captura
                    if time.time() - self.last_frame_time > 5:
                        self.logger.warning("No hay frames válidos por 5 segundos, reiniciando captura...")
                        self.cap.release()
                        self.init_capture()
                        
            except Exception as e:
                self.logger.error(f"Error en capture_loop: {e}")
                time.sleep(1)
                continue

            time.sleep(0.03)  # ~30 fps
    
    def read(self):
        with self.lock:
            if time.time() - self.last_frame_time > 5:
                return False, None
            return self._ret, self._frame.copy() if self._frame is not None else None
    
    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

logger.info("Starting the script...")

smoothing_factor = 0.05
prev_dominant_color = np.array([255, 255, 255])

light_entity_id = os.environ.get("LIGHT_ENTITY_ID", "light.ledcomedorsamsung")
media_player_entity_id = os.environ.get("MEDIA_PLAYER_ENTITY_ID", "media_player.samsung_qn85ca_75_2")
color_algorithm = os.environ.get("COLOR_ALGORITHM", "kmeans")

last_update_time = time.time()
update_interval = 0.5  # Reducir a medio segundo

# api_client = CustomAPIClient(os.environ['HASSIO_HOST'], os.environ['HASSIO_TOKEN'])

# we pass the entities to the websocket client so we can get the state of the TV and the light and control the light
entities = [
    light_entity_id,
    media_player_entity_id
]

def parse_args():
    parser = argparse.ArgumentParser(description='Smart Room Control')
    parser.add_argument('--entity', help='Entity ID to set initial state')
    parser.add_argument('--status', help='Initial state for the entity')
    return parser.parse_args()

# Initialize websocket client with command line arguments
api_client_websocket = CustomWebsocketClient(
    os.environ['HASSIO_HOST_WEBSOCKET'], 
    os.environ['HASSIO_TOKEN'], 
    entities=entities
)

# Process command line arguments
args = parse_args()
if args.entity and args.status:
    api_client_websocket.set_initial_state(args.entity, args.status)
    logger.info(f"Initialized {args.entity} with state: {args.status}")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurar Jinja2Templates
templates = Jinja2Templates(directory="templates")

def is_tv_on():
    state = api_client_websocket.get_entity_state(media_player_entity_id)
    logger.info(f"TV state check: {state}")  # Debug
    return state == "on"

async def turn_on_light(count=0):
    if count >= 3:
        logger.error("Failed to turn on the light after 3 attempts. Exiting the script...")
        exit()
    try:
        rgb_color = [255, 0, 0]
        await api_client_websocket.turn_on(entity_id=light_entity_id, brightness_pct=100, rgb_color=rgb_color)
    except Exception as e:
        logger.error(f"Error controlling lights: {e}")
        await turn_on_light(count+1)

async def turn_off_light(count=0):
    if count >= 3:
        logger.error("Failed to turn off the light after 3 attempts. Exiting the script...")
        exit()
    try:
        await api_client_websocket.turn_off(entity_id=light_entity_id)
    except Exception as e:
        logger.error(f"Error turning off lights: {e}")
        await turn_off_light(count+1)

async def turn_on_set_light(target_color, brightness_pct, rgbww_values=[255, 255], count=0):
    if count >= 3:
        logger.error("Failed to turn on the light after 3 attempts. Exiting the script...")
        return
    
    try:
        rgb_color = target_color
        logger.info(f"Setting light color to: {rgb_color} with brightness: {brightness_pct}%")
        await api_client_websocket.turn_on(entity_id=light_entity_id, brightness_pct=brightness_pct, rgb_color=rgb_color)
    except Exception as e:
        logger.error(f"Error turning on lights: {e}")
        await turn_on_set_light(target_color, brightness_pct, rgbww_values, count+1)

pause_color_change = False

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "smoothing_factor": smoothing_factor,
        "update_interval": update_interval,
        "light_entity_id": light_entity_id,
        "media_player_entity_id": media_player_entity_id
    })

@app.post("/turn_on")
async def turn_on(request: Request):
    data = await request.json()
    color = data.get('color', [255, 255, 255])
    brightness = data.get('brightness', 100)
    await api_client_websocket.turn_on(entity_id=light_entity_id, brightness_pct=brightness, rgb_color=color)
    return JSONResponse({"status": "success"})

@app.post("/turn_off")
async def turn_off():
    await api_client_websocket.turn_off(entity_id=light_entity_id)
    return JSONResponse({"status": "success"})

@app.post("/set_smoothing_factor")
async def set_smoothing_factor(request: Request):
    global smoothing_factor
    data = await request.json()
    smoothing_factor = data.get('smoothing_factor', 0.05)
    return JSONResponse({"status": "success", "smoothing_factor": smoothing_factor})

@app.post("/set_update_interval")
async def set_update_interval(request: Request):
    global update_interval
    data = await request.json()
    update_interval = data.get('update_interval', 0.5)
    return JSONResponse({"status": "success", "update_interval": update_interval})

@app.post("/set_entity_ids")
async def set_entity_ids(request: Request):
    global light_entity_id, media_player_entity_id
    data = await request.json()
    
    # Update light entity if provided
    if 'light_entity_id' in data:
        old_light_id = light_entity_id
        light_entity_id = data['light_entity_id']
        api_client_websocket.update_entity(old_light_id, light_entity_id)
    
    # Update media player entity if provided
    if 'media_player_entity_id' in data:
        old_media_id = media_player_entity_id
        media_player_entity_id = data['media_player_entity_id']
        api_client_websocket.update_entity(old_media_id, media_player_entity_id)
    
    # Después de actualizar, intentar obtener los estados iniciales
    await asyncio.sleep(0.1)  # Ensure no other coroutines are running
    await api_client_websocket._fetch_initial_states()
    
    return JSONResponse({
        "status": "success",
        "light_entity_id": light_entity_id,
        "media_player_entity_id": media_player_entity_id
    })

@app.post("/set_color_algorithm")
async def set_color_algorithm(request: Request):
    global color_algorithm
    data = await request.json()
    color_algorithm = data.get('color_algorithm', 'kmeans')
    return JSONResponse({"status": "success", "color_algorithm": color_algorithm})

@app.post("/pause")
async def pause():
    global pause_color_change
    pause_color_change = True
    return JSONResponse({"status": "paused"})

@app.post("/resume")
async def resume():
    global pause_color_change
    pause_color_change = False
    return JSONResponse({"status": "resumed"})

@app.get("/get_settings")
async def get_settings():
    return JSONResponse({
        "smoothing_factor": smoothing_factor,
        "update_interval": update_interval,
        "light_entity_id": light_entity_id,
        "media_player_entity_id": media_player_entity_id,
        "color_algorithm": color_algorithm
    })

@app.get("/get_feedback")
async def get_feedback():
    global frame_grab_success, updating_colors, error_occurred, current_frame
    frame_grab_success = frame_grab_success and current_frame is not None
    updating_colors = updating_colors and not error_occurred
    brightness_pct = int((calculate_brightness(prev_dominant_color) / 255) * 100)
    return JSONResponse({
        "current_color": prev_dominant_color.tolist(),
        "brightness_pct": brightness_pct,  # Incluir brightness_pct
        "frame_grab_success": frame_grab_success,
        "updating_colors": updating_colors,
        "error_occurred": error_occurred,
        "flask_thread_alive": flask_thread.is_alive(),
        "video_thread_alive": video_thread.is_alive(),
        "tv_status": is_tv_on(),  # Incluir el estado de la TV
        "pause_color_change": pause_color_change  # Incluir el estado de pausa
    })

@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            if current_frame is not None:
                frame_copy = current_frame.copy()
                # Añadir timestamp al frame
                cv2.putText(frame_copy, str(time.time())[:10], (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Codificar frame a JPEG
                _, buffer = cv2.imencode('.jpg', frame_copy)
                frame_bytes = buffer.tobytes()
                
                # Enviar frame en formato multipart
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            await asyncio.sleep(0.033)  # ~30 FPS

    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.post("/restart_flask_thread")
async def restart_flask_thread():
    global flask_thread
    if flask_thread.is_alive():
        return JSONResponse({"status": "Flask thread is already running"})
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    return JSONResponse({"status": "Flask thread restarted"})

@app.post("/restart_video_thread")
async def restart_video_thread():
    global video_thread
    if video_thread.is_alive():
        return JSONResponse({"status": "Video capture thread is already running"})
    video_thread = threading.Thread(target=run_video_capture)
    video_thread.start()
    return JSONResponse({"status": "Video capture thread restarted"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await emit_feedback(websocket)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected")

async def emit_feedback(websocket: WebSocket):
    global frame_grab_success, updating_colors, error_occurred, current_frame
    frame_grab_success = frame_grab_success and current_frame is not None
    updating_colors = updating_colors and not error_occurred
    brightness_pct = int((calculate_brightness(prev_dominant_color) / 255) * 100)
    await websocket.send_json({
        "current_color": prev_dominant_color.tolist(),
        "brightness_pct": brightness_pct,  # Incluir brightness_pct
        "frame_grab_success": frame_grab_success,
        "updating_colors": updating_colors,
        "error_occurred": error_occurred,
        "flask_thread_alive": flask_thread.is_alive(),
        "video_thread_alive": video_thread.is_alive(),
        "tv_status": is_tv_on(),  # Incluir el estado de la TV
        "pause_color_change": pause_color_change  # Incluir el estado de pausa
    })

def run_flask():
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)

def run_websocket_client():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(api_client_websocket.init_socket(loop))
    finally:
        loop.close()

def run_video_capture():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_video_capture_async())
    finally:
        loop.close()

# @lru_cache(maxsize=32)
def calculate_brightness(color):
    # Convertir array numpy a lista si es necesario
    if isinstance(color, np.ndarray):
        color = color.tolist()
    
    # Asegurarse de que tenemos una secuencia de 3 valores RGB
    if len(color) != 3:
        raise ValueError("El color debe tener 3 componentes (RGB)")
        
    # Fórmula para calcular el brillo percibido
    return int(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])

async def run_video_capture_async():
    global prev_dominant_color, last_update_time, skipped_frames, frame_grab_success
    global updating_colors, error_occurred, current_frame
    
    retry_count = 0
    max_retries = 3
    tv_was_off = False
    color_processor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    try:
        # Única inicialización de la cámara aquí
        cap = VideoCapture(0)
        logger.info("Video capture object created...")
        
        if not cap.cap.isOpened():
            raise Exception("No se pudo abrir la captura de video")

        while True:
            try:
                frame_grab_success = False
                updating_colors = False
                error_occurred = False
                current_frame = None

                # Verificar estado de TV y manejar luces
                if not is_tv_on():
                    if not tv_was_off:
                        logger.info("Samsung TV is off. Turning off lights...")
                        await turn_off_light()
                        tv_was_off = True
                        skipped_frames = 0
                    await asyncio.sleep(1)
                    continue
                else:
                    if tv_was_off:
                        skipped_frames = 0
                    tv_was_off = False

                

                if skipped_frames < 5:
                    logger.info(f"Skipping frame {skipped_frames + 1}/5...")
                    ret, _ = cap.read()
                    skipped_frames += 1
                    await asyncio.sleep(0.1)
                    continue

                ret, frame = cap.read()

                if not ret or frame is None:
                    logger.warning("Frame vacío detectado, saltando...")
                    frame_grab_success = False
                    error_occurred = True
                    await asyncio.sleep(1)
                    continue
                
                frame_grab_success = True
                error_occurred = False
                current_frame = frame

                # Resto del código de procesamiento de video
                if pause_color_change:
                    logger.info("Color change is paused.")
                    await asyncio.sleep(1)
                    continue

                # Procesar color en thread separado
                dominant_color = await asyncio.get_event_loop().run_in_executor(
                    color_processor, 
                    get_dominant_color,
                    frame, 
                    prev_dominant_color
                )

                if time.time() - last_update_time > update_interval:
                    # Reducir frecuencia de actualizaciones si no hay cambios significativos
                    if np.all(np.abs(dominant_color - prev_dominant_color) < 5):
                        continue
                        
                    try:
                        dominant_color = smooth_color(prev_dominant_color, dominant_color)
                        brightness = calculate_brightness(tuple(dominant_color))  # Convertir a tupla
                        brightness_pct = int((brightness / 255) * 100)
                        logger.info("Updating LED color  to: %s with brightness: %d%%", dominant_color, brightness_pct)
                        ww_values = calculate_ww_values(dominant_color)
                        await turn_on_set_light(dominant_color.astype(int).tolist(), brightness_pct, ww_values)
                        prev_dominant_color = dominant_color
                        updating_colors = True
                        error_occurred = False
                    except Exception as e:
                        logger.info(f"Error updating LED color: {e}")
                        updating_colors = False
                        error_occurred = True
                    last_update_time = time.time()
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except Exception as e:
                logger.error(f"Unexpected error in video capture loop: {e}")
                error_occurred = True
                retry_count += 1
                if retry_count >= max_retries:
                    logger.warning("Reiniciando captura después de múltiples errores...")
                    if cap is not None:
                        cap.release()
                    cap = VideoCapture(0)
                    retry_count = 0
                await asyncio.sleep(1)

    except Exception as e:
        logger.critical(f"Critical error in video capture: {e}")
        error_occurred = True
    
    finally:
        color_processor.shutdown()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

def get_dominant_color(frame, prev_dominant_color):
    if color_algorithm == "kmeans":
        return get_dominant_color_kmeans(frame, prev_dominant_color)
    elif color_algorithm == "average":
        return get_dominant_color_average(frame)
    elif color_algorithm == "median":
        return get_dominant_color_median(frame)
    elif color_algorithm == "mode":
        return get_dominant_color_mode(frame)
    else:
        return prev_dominant_color

def run_async_thread(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

if __name__ == '__main__':
    frame_grab_success = False
    updating_colors = False
    error_occurred = False
    skipped_frames = 0

    # Iniciar primero el websocket y esperar a que se conecte
    websocket_thread = threading.Thread(target=run_websocket_client)
    websocket_thread.start()
    logger.info("Waiting for WebSocket connection...")
    time.sleep(2)  # Dar tiempo para que se establezca la conexión

    # Verificar el estado inicial de la TV
    if not is_tv_on():
        logger.info("TV is off. Starting with lights off...")
        run_async_thread(turn_off_light())
    
    # Luego iniciar los demás servicios
    flask_thread = threading.Thread(target=run_flask)
    video_thread = threading.Thread(target=run_video_capture)
    
    flask_thread.start()
    video_thread.start()

    while True:
        if not websocket_thread.is_alive():
            logger.warning("Websocket thread stopped. Restarting...")
            websocket_thread = threading.Thread(target=run_websocket_client)
            websocket_thread.start()
            time.sleep(2)  # Esperar a que se reconecte

        if not flask_thread.is_alive():
            logger.warning("Flask thread stopped. Restarting...")
            flask_thread = threading.Thread(target=run_flask)
            flask_thread.start()

        if not video_thread.is_alive():
            logger.warning("Video capture thread stopped. Restarting...")
            video_thread = threading.Thread(target=run_video_capture)
            video_thread.start()

        time.sleep(1)
