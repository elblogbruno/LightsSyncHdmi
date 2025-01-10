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
from color_algorithm import get_dominant_color_average, get_dominant_color_kmeans, smooth_color, calculate_brightness, get_dominant_color_median, get_dominant_color_mode, calculate_ww_values
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import asyncio

dotenv.load_dotenv()

print("Starting the script...")

start_time = time.time()
cap = cv2.VideoCapture(0)
if os.cpu_count() < 4:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced frame size
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print(f"Camera opened in {time.time() - start_time:.2f} seconds")

print("Video capture object created...")

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

smoothing_factor = 0.05
prev_dominant_color = np.array([255, 255, 255])

light_entity_id = os.environ.get("LIGHT_ENTITY_ID", "light.ledcomedorsamsung")
media_player_entity_id = os.environ.get("MEDIA_PLAYER_ENTITY_ID", "media_player.samsung_qn85ca_75_2")
color_algorithm = os.environ.get("COLOR_ALGORITHM", "kmeans")

last_update_time = time.time()
update_interval = 1.0  # Increased update interval for better performance

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
    print(f"Initialized {args.entity} with state: {args.status}")

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
    print(f"TV state check: {state}")  # Debug
    return state == "on"

async def turn_on_light(count=0):
    if count >= 3:
        print("Failed to turn on the light after 3 attempts. Exiting the script...")
        exit()
    try:
        rgb_color = [255, 0, 0]
        await api_client_websocket.turn_on(entity_id=light_entity_id, brightness_pct=100, rgb_color=rgb_color)
    except Exception as e:
        print(f"Error controlling lights: {e}")
        await turn_on_light(count+1)

async def turn_off_light(count=0):
    if count >= 3:
        print("Failed to turn off the light after 3 attempts. Exiting the script...")
        exit()
    try:
        await api_client_websocket.turn_off(entity_id=light_entity_id)
    except Exception as e:
        print(f"Error turning off lights: {e}")
        await turn_off_light(count+1)

async def turn_on_set_light(target_color, brightness_pct, rgbww_values=[255, 255], count=0):
    if count >= 3:
        print("Failed to turn on the light after 3 attempts. Exiting the script...")
        return
    
    try:
        rgb_color = target_color
        print(f"Setting light color to: {rgb_color} with brightness: {brightness_pct}%")
        await api_client_websocket.turn_on(entity_id=light_entity_id, brightness_pct=brightness_pct, rgb_color=rgb_color)
    except Exception as e:
        print(f"Error turning on lights: {e}")
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
    light_entity_id = data.get('light_entity_id', light_entity_id)
    media_player_entity_id = data.get('media_player_entity_id', media_player_entity_id)
    return JSONResponse({"status": "success", "light_entity_id": light_entity_id, "media_player_entity_id": media_player_entity_id})

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

@app.get("/random_frame")
async def random_frame():
    global current_frame
    if current_frame is None:
        return JSONResponse(content={"message": "Frame not found"}, status_code=404)
    
    random_frame_encoded = cv2.imencode('.jpg', current_frame)[1].tobytes()
    return StreamingResponse(io.BytesIO(random_frame_encoded), media_type="image/jpeg")

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
        print("WebSocket disconnected")

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

async def run_video_capture_async():
    global prev_dominant_color, last_update_time, skipped_frames, frame_grab_success, updating_colors, error_occurred, current_frame
    while True:
        try:
            frame_grab_success = False
            updating_colors = False
            error_occurred = False
            current_frame = None
            while True:
                try:
                    if not is_tv_on():
                        print("Samsung TV is off. Pausing the script...")
                        await turn_off_light()
                        await asyncio.sleep(1)
                        break

                    if skipped_frames < 5:
                        print("Skipping frames...")
                        cap.read()
                        skipped_frames += 1
                        continue

                    ret, frame = cap.read()

                    if not ret:
                        print("Failed to grab frame")
                        frame_grab_success = False
                        error_occurred = True
                        continue
                    else:
                        frame_grab_success = True
                        error_occurred = False

                    current_frame = frame

                    if pause_color_change:
                        print("Color change is paused.")
                        await asyncio.sleep(1)
                        continue

                    dominant_color = get_dominant_color(frame, prev_dominant_color)

                    if time.time() - last_update_time > update_interval:
                        try:
                            dominant_color = smooth_color(prev_dominant_color, dominant_color)
                            brightness = calculate_brightness(dominant_color)
                            brightness_pct = int((brightness / 255) * 100)
                            print("Updating LED color to:", dominant_color, "with brightness:", brightness_pct)
                            ww_values = calculate_ww_values(dominant_color)
                            await turn_on_set_light(dominant_color.astype(int).tolist(), brightness_pct, ww_values)
                            prev_dominant_color = dominant_color
                            updating_colors = True
                            error_occurred = False
                        except Exception as e:
                            print(f"Error updating LED color: {e}")
                            updating_colors = False
                            error_occurred = True
                        last_update_time = time.time()

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                except Exception as e:
                    print(f"Unexpected error in video capture loop: {e}")
                    error_occurred = True
                    await asyncio.sleep(1)

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Video capture thread encountered an error: {e}")
            await asyncio.sleep(1)

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

    flask_thread = threading.Thread(target=run_flask)
    video_thread = threading.Thread(target=run_video_capture)
    websocket_thread = threading.Thread(target=run_websocket_client)
    
    flask_thread.start()
    video_thread.start()
    websocket_thread.start()

    while True:
        if not flask_thread.is_alive():
            print("Flask thread stopped. Restarting...")
            flask_thread = threading.Thread(target=run_flask)
            flask_thread.start()

        if not video_thread.is_alive():
            print("Video capture thread stopped. Restarting...")
            video_thread = threading.Thread(target=run_video_capture)
            video_thread.start()

        if not websocket_thread.is_alive():
            print("Websocket thread stopped. Restarting...")
            websocket_thread = threading.Thread(target=run_websocket_client)
            websocket_thread.start()

        time.sleep(1)  # Check thread status every second
