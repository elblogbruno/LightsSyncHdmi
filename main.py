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
# Create a VideoCapture object for the capture card (0 for webcam)
cap = cv2.VideoCapture(0)

# Set frame width and height for better color capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Video capture object created...")

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Parameters for smoothing
smoothing_factor = 0.05  # Adjusted smoothing factor for less aggressive color transition
prev_dominant_color = np.array([255, 255, 255])  # Initial color white

light_entity_id = os.environ.get("LIGHT_ENTITY_ID", "light.LedComedorSamsung")
media_player_entity_id = os.environ.get("MEDIA_PLAYER_ENTITY_ID", "media_player.samsung_qn85ca_75_2")

# Process frames
last_update_time = time.time()
update_interval = 0.5  # Update LEDs every 0.5 seconds
frame_counter = 0  # For saving a frame once

api_client = CustomAPIClient(os.environ['HASSIO_HOST'], os.environ['HASSIO_TOKEN'])

# Initialize Home Assistant API client
try:
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
            rgbww_color = target_color + [255, 255]  # Assuming WW values are always max
            api_client.turn_on(entity_id=light_entity_id, brightness_pct=brightness_pct, rgbww_color=rgbww_color)
        except Exception as e:
            print(f"Error controlling lights: {e}")
            turn_on_set_light(target_color, brightness_pct, count+1)

    # Suavizado de color
    def smooth_color(prev_color, new_color, factor=0.1):
        return prev_color * (1 - factor) + new_color * factor

    # Calculate brightness
    def calculate_brightness(color):
        return np.sqrt(0.299 * color[0]**2 + 0.587 * color[1]**2 + 0.114 * color[2]**2)

    # Set initial LED color to red
    try:
        print("Turning on the light...")
        turn_on_light()
        time.sleep(5)
        print("Turning off the light...")
        turn_off_light()
    except Exception as e:
        print(f"Error controlling lights: {e}")

    time.sleep(10)  # Wait for the light to turn off and some time for the TV to turn on

    skipped_frames = 0

    while True:
        # Check if the TV is off
        if not is_tv_on():
            print("Samsung TV is off. Pausing the script...")
            turn_off_light()
            time.sleep(1)  # Wait 5 seconds before checking again
            continue  # Skip to the next iteration if the TV is off

        # skip some frames until we have a stable image
        if skipped_frames < 10:
            print("Skipping frames...")
            cap.read()
            skipped_frames += 1
            continue # Skip to the next iteration if the TV is off

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Downscale frame for faster processing
        small_frame = cv2.resize(frame, (320, 240))

        # Save a frame for testing
        if frame_counter < 5:
            cv2.imwrite('captured_frame_' + str(frame_counter) + '.jpg', frame)
            print("Frame saved as 'captured_frame.jpg'")
            frame_counter += 1

        # Reshape the image to be a list of pixels
        pixels = small_frame.reshape((-1, 3))

        # Perform k-means clustering to find the dominant color
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(pixels)
        dominant_color = kmeans.cluster_centers_[0]

        # Dentro del bucle principal
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

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error initializing Home Assistant client: {e}")
