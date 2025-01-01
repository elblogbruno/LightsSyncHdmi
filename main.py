import cv2
import numpy as np
import os
import dotenv
import time
import threading
from flask import Flask, render_template, request, jsonify, Response
# from sklearn.cluster import MiniBatchKMeans
from api import CustomAPIClient
# import base64
# import random
from color_algorithm import smooth_color, calculate_brightness, get_dominant_color

dotenv.load_dotenv()

print("Starting the script...")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced frame size
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("Video capture object created...")

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

smoothing_factor = 0.05
prev_dominant_color = np.array([255, 255, 255])

light_entity_id = os.environ.get("LIGHT_ENTITY_ID", "light.ledcomedorsamsung")
media_player_entity_id = os.environ.get("MEDIA_PLAYER_ENTITY_ID", "media_player.samsung_qn85ca_75_2")

last_update_time = time.time()
update_interval = 1.0  # Increased update interval for better performance

api_client = CustomAPIClient(os.environ['HASSIO_HOST'], os.environ['HASSIO_TOKEN'])

def is_tv_on(count=0):
    if count >= 3:
        print("Failed to check TV state after 3 attempts. Exiting the script...")
        return False
    
    try:
        tv = api_client.get_entity(entity_id=media_player_entity_id)
        if not tv:
            return False
        
        print(tv["state"]  + " " +  str(time.time()))
        return tv["state"] == "on"
    except Exception as e:
        print(f"Error checking TV state: {e}")
        return is_tv_on(count+1)
    
def turn_on_light(count=0):
    if count >= 3:
        print("Failed to turn on the light after 3 attempts. Exiting the script...")
        exit()
    try:
        rgbww_color = [255, 0, 0, 255, 255]
        api_client.turn_on(entity_id=light_entity_id, brightness_pct=100, rgbww_color=rgbww_color)
    except Exception as e:
        print(f"Error controlling lights: {e}")
        turn_on_light(count+1)

def turn_off_light(count=0):
    if count >= 3:
        print("Failed to turn off the light after 3 attempts. Exiting the script...")
        exit()
    try:
        rgbww_color = [0, 0, 0, 0, 0]
        api_client.turn_on(entity_id=light_entity_id, brightness_pct=0, rgbww_color=rgbww_color)
    except Exception as e:
        print(f"Error controlling lights: {e}")
        turn_off_light(count+1)

def turn_on_set_light(target_color, brightness_pct, count=0):
    if count >= 3:
        print("Failed to turn on the light after 3 attempts. Exiting the script...")
        return
    
    try:
        rgbww_color = target_color + [255, 255]
        print(f"Setting light color to: {rgbww_color} with brightness: {brightness_pct}%")
        api_client.turn_on(entity_id=light_entity_id, brightness_pct=brightness_pct, rgbww_color=rgbww_color)
    except Exception as e:
        print(f"Error controlling lights: {e}")
        turn_on_set_light(target_color, brightness_pct, count+1)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', smoothing_factor=smoothing_factor, update_interval=update_interval, light_entity_id=light_entity_id, media_player_entity_id=media_player_entity_id)

@app.route('/turn_on', methods=['POST'])
def turn_on():
    color = request.json.get('color', [255, 255, 255, 255, 255])
    brightness = request.json.get('brightness', 100)
    api_client.turn_on(entity_id=light_entity_id, brightness_pct=brightness, rgbww_color=color)
    return jsonify({"status": "success"})

@app.route('/turn_off', methods=['POST'])
def turn_off():
    api_client.turn_off(entity_id=light_entity_id)
    return jsonify({"status": "success"})

@app.route('/set_smoothing_factor', methods=['POST'])
def set_smoothing_factor():
    global smoothing_factor
    smoothing_factor = request.json.get('smoothing_factor', 0.05)
    return jsonify({"status": "success", "smoothing_factor": smoothing_factor})

@app.route('/set_update_interval', methods=['POST'])
def set_update_interval():
    global update_interval
    update_interval = request.json.get('update_interval', 0.5)
    return jsonify({"status": "success", "update_interval": update_interval})

@app.route('/set_entity_ids', methods=['POST'])
def set_entity_ids():
    global light_entity_id, media_player_entity_id
    light_entity_id = request.json.get('light_entity_id', light_entity_id)
    media_player_entity_id = request.json.get('media_player_entity_id', media_player_entity_id)
    return jsonify({"status": "success", "light_entity_id": light_entity_id, "media_player_entity_id": media_player_entity_id})

@app.route('/get_settings', methods=['GET'])
def get_settings():
    return jsonify({
        "smoothing_factor": smoothing_factor,
        "update_interval": update_interval,
        "light_entity_id": light_entity_id,
        "media_player_entity_id": media_player_entity_id
    })

@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    return jsonify({
        "current_color": prev_dominant_color.tolist(),
        "frame_grab_success": frame_grab_success,
        "updating_colors": updating_colors,
        "error_occurred": error_occurred,
        "flask_thread_alive": flask_thread.is_alive(),
        "video_thread_alive": video_thread.is_alive()
    })

@app.route('/random_frame')
def random_frame():
    global current_frame
    if (current_frame is None):
        return Response(status=404)
    
    random_frame_encoded = cv2.imencode('.jpg', current_frame)[1].tobytes()

    return Response(random_frame_encoded, mimetype='image/jpeg')

@app.route('/restart_flask_thread', methods=['POST'])
def restart_flask_thread():
    global flask_thread
    if flask_thread.is_alive():
        return jsonify({"status": "Flask thread is already running"})
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()
    return jsonify({"status": "Flask thread restarted"})

@app.route('/restart_video_thread', methods=['POST'])
def restart_video_thread():
    global video_thread
    if video_thread.is_alive():
        return jsonify({"status": "Video capture thread is already running"})
    video_thread = threading.Thread(target=run_video_capture)
    video_thread.start()
    return jsonify({"status": "Video capture thread restarted"})

def run_flask():
    while True:
        try:
            app.run(host='0.0.0.0', port=5000)
        except Exception as e:
            print(f"Flask thread encountered an error: {e}")
            time.sleep(1)  # Wait before retrying

def run_video_capture():
    global prev_dominant_color, last_update_time, skipped_frames, frame_grab_success, updating_colors, error_occurred, current_frame
    while True:
        try:
            frame_grab_success = False  # Inicializar la variable
            updating_colors = False  # Inicializar la variable
            error_occurred = False  # Inicializar la variable
            current_frame = None  # Inicializar la variable 
            while True:
                try:
                    if not is_tv_on():
                        print("Samsung TV is off. Pausing the script...")
                        turn_off_light()
                        time.sleep(1)
                        continue

                    if skipped_frames < 10:
                        print("Skipping frames...")
                        cap.read()
                        skipped_frames += 1
                        continue

                    ret, frame = cap.read()

                    if not ret:
                        print("Failed to grab frame")
                        frame_grab_success = False
                        error_occurred = True
                    else:
                        frame_grab_success = True
                        error_occurred = False

                    # Encode the frame to JPEG format at random intervals
                    current_frame = frame

                    # Get the dominant color
                    dominant_color = get_dominant_color(frame, prev_dominant_color)

                    if time.time() - last_update_time > update_interval:
                        try:
                            dominant_color = smooth_color(prev_dominant_color, dominant_color)
                            brightness = calculate_brightness(dominant_color)
                            brightness_pct = int((brightness / 255) * 100)
                            print("Updating LED color to:", dominant_color, "with brightness:", brightness_pct)
                            turn_on_set_light(dominant_color.astype(int).tolist(), brightness_pct)
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
                    time.sleep(1)  # Wait before retrying

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Video capture thread encountered an error: {e}")
            time.sleep(1)  # Wait before retrying

if __name__ == '__main__':
    frame_grab_success = False
    updating_colors = False
    error_occurred = False
    skipped_frames = 0

    flask_thread = threading.Thread(target=run_flask)
    video_thread = threading.Thread(target=run_video_capture)
    flask_thread.start()
    video_thread.start()

    while True:
        if not flask_thread.is_alive():
            print("Flask thread stopped. Restarting...")
            flask_thread = threading.Thread(target=run_flask)
            flask_thread.start()

        if not video_thread.is_alive():
            print("Video capture thread stopped. Restarting...")
            video_thread = threading.Thread(target=run_video_capture)
            video_thread.start()

        time.sleep(1)  # Check thread status every second
