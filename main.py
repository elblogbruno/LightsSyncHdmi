import cv2
import numpy as np
import os
import dotenv
import time
import threading
from flask import Flask, render_template, request, jsonify
from sklearn.cluster import KMeans
from api import CustomAPIClient

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

light_entity_id = os.environ.get("LIGHT_ENTITY_ID", "light.habitacion")
media_player_entity_id = os.environ.get("MEDIA_PLAYER_ENTITY_ID", "media_player.samsung_qn85ca_75_2")

last_update_time = time.time()
update_interval = 0.5

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

def smooth_color(prev_color, new_color, factor=0.1):
    return prev_color * (1 - factor) + new_color * factor

def calculate_brightness(color):
    return np.sqrt(0.299 * color[0]**2 + 0.587 * color[1]**2 + 0.114 * color[2]**2)

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

def run_flask():
    app.run(host='0.0.0.0', port=5000)

def run_video_capture():
    global prev_dominant_color, last_update_time, skipped_frames
    while True:
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
            break

        small_frame = cv2.resize(frame, (160, 120))  # Further reduced frame size

        height, width, _ = small_frame.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (width//4, height//4), (3*width//4, 3*height//4), 255, -1)
        masked_frame = cv2.bitwise_and(small_frame, small_frame, mask=mask)

        pixels = masked_frame.reshape((-1, 3))
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=-1)]

        try:
            kmeans = KMeans(n_clusters=4)  # Reduced number of clusters
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
            print(f"Detected dominant color: {dominant_color}")
        except Exception as e:
            print(f"Error during KMeans clustering: {e}")
            continue

        if time.time() - last_update_time > update_interval:
            try:
                dominant_color = smooth_color(prev_dominant_color, dominant_color)
                brightness = calculate_brightness(dominant_color)
                brightness_pct = int((brightness / 255) * 100)
                print("Updating LED color to:", dominant_color, "with brightness:", brightness_pct)
                turn_on_set_light(dominant_color.astype(int).tolist(), brightness_pct)
                prev_dominant_color = dominant_color
            except Exception as e:
                print(f"Error updating LED color: {e}")
            last_update_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    video_thread = threading.Thread(target=run_video_capture)
    flask_thread.start()
    video_thread.start()
    flask_thread.join()
    video_thread.join()
