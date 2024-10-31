import cv2
import numpy as np
import os
import dotenv
import time
from homeassistant_api import Client
from sklearn.cluster import KMeans
import webcolors

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

light_entity_id = os.environ.get("LIGHT_ENTITY_ID", "light.LedComedorSamsung")
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

# Create a directory to save frames
debug_frames_dir = "debug_frames"
os.makedirs(debug_frames_dir, exist_ok=True)

try:
    print("Turning on the light...")
    turn_on_light()
    time.sleep(5)
    print("Turning off the light...")
    turn_off_light()
except Exception as e:
    print(f"Error controlling lights: {e}")

time.sleep(10)

skipped_frames = 0
frame_count = 0

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

    kmeans = KMeans(n_clusters=4)  # Reduced number of clusters
    kmeans.fit(pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    print(f"Detected dominant color: {dominant_color}")

    # Save the frame for debugging
    frame_filename = os.path.join(debug_frames_dir, f"frame_{frame_count}_color_{dominant_color.astype(int)}.png")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

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
