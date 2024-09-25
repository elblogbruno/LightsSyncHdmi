import cv2
import numpy as np
import os
import dotenv
import time
from homeassistant_api import Client
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
smoothing_factor = 0.1
prev_dominant_color = np.array([255, 255, 255])  # Initial color white

light_entity_id = os.environ.get("LIGHT_ENTITY_ID", "light.LedComedorSamsung")
media_player_entity_id = os.environ.get("MEDIA_PLAYER_ENTITY_ID", "media_player.samsung_qn85ca_75_2")

# Process frames
last_update_time = time.time()
update_interval = 0.5  # Update LEDs every 0.5 seconds
frame_counter = 0  # For saving a frame once

def closest_colour(requested_colour):
    min_colours = {}
    for name in webcolors.names("css3"):
        r_c, g_c, b_c = webcolors.name_to_rgb(name)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_color_name(rgb_tuple):
    try:
        # Convert RGB to hex
        hex_value = webcolors.rgb_to_hex(rgb_tuple)
        # Get the color name directly
        return webcolors.hex_to_name(hex_value)
    except ValueError:
        # If exact match not found, find the closest color
        return closest_colour(rgb_tuple)


api_client = CustomAPIClient(os.environ['HASSIO_HOST'], os.environ['HASSIO_TOKEN'])

# Initialize Home Assistant API client
try:
    # with Client(
    #     os.environ['HASSIO_HOST'],
    #     os.environ['HASSIO_TOKEN'],
    # ) as client:

    #     light = client.get_domain("light") 

    # Function to check if TV is on
    def is_tv_on(count=0):
        if count >= 3:
            print("Failed to check TV state after 3 attempts. Exiting the script...")
            return False
        
        try:
            tv = api_client.get_entity(entity_id=media_player_entity_id)  # Replace with your TV entity ID
            if not tv:
                return False
            
            print(tv["state"]  + " " +  str(time.time()))
            return tv["state"] == "on"
        except Exception as e:
            print(f"Error checking TV state: {e}")
            # try again
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
            # try again
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
            # try again
            turn_off_light(count+1)

    def turn_on_set_light(target_color, count=0):
        if count >= 3:
            print("Failed to turn on the light after 3 attempts. Exiting the script...")
            return
        
        try:
            rgbww_color = target_color + [255, 255]  # Assuming WW values are always max
            api_client.turn_on(entity_id=light_entity_id, brightness_pct=100, rgbww_color=rgbww_color)
        except Exception as e:
            print(f"Error controlling lights: {e}")
            # try again
            turn_on_set_light(target_color, count+1)

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
        # small_frame = cv2.resize(frame, (320, 240))
        small_frame = frame  # No need to downscale for better color capture

        # Apply gamma correction to adjust for non-linearities in the display
        gamma = 1.0  # Adjust gamma to 1.0 for more accurate color representation
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        small_frame = cv2.LUT(small_frame, table)

        
        # reduce processing by reducing frame pixel count, since we only need the dominant color
        small_frame = cv2.resize(small_frame, (160, 120))

        # Define regions of interest (ROIs)
        height, width, _ = small_frame.shape
        regions = {
            'top_left': small_frame[:height // 2, :width // 2],
            'top_right': small_frame[:height // 2, width // 2:],
            'bottom_left': small_frame[height // 2:, :width // 2],
            'bottom_right': small_frame[height // 2:, width // 2:],
            'center': small_frame[height // 4: 3 * height // 4, width // 4: 3 * width // 4]
        }

        # Save a frame for testing
        if frame_counter < 5:
            # Save the first frame as 'captured_frame.jpg'
            cv2.imwrite('captured_frame_' + str(frame_counter) + '.jpg', frame)
            print("Frame saved as 'captured_frame.jpg'")

            for region_name, region in regions.items():
                cv2.imwrite(f'captured_{region_name}_{frame_counter}.jpg', region)
                print(f"Region '{region_name}' saved as 'captured_{region_name}_{frame_counter}.jpg'")

            frame_counter += 1

        dominant_colors = []

        # Process each region to find dominant color
        for region_name, region in regions.items():
            region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

            # Get the most dominant color using k-means
            pixels = np.float32(region_rgb.reshape(-1, 3))
            n_colors = 10  # Increase clusters for better accuracy
            _, labels, palette = cv2.kmeans(pixels, n_colors, None,
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1), 10,
                                            cv2.KMEANS_RANDOM_CENTERS)
            _, counts = np.unique(labels, return_counts=True)
            dominant_color = palette[np.argmax(counts)]

            dominant_colors.append(dominant_color)

        # Average the dominant colors from all regions
        average_color = np.mean(dominant_colors, axis=0)

        # Smooth the color transition by blending previous and current dominant colors
        average_color = smoothing_factor * average_color + (1 - smoothing_factor) * prev_dominant_color
        average_color = average_color.astype(int)  # Convert to integer

        # Update previous color for the next iteration
        prev_dominant_color = average_color

        # Ensure the color has enough brightness and saturation
        dominant_hsv = cv2.cvtColor(np.uint8([[average_color]]), cv2.COLOR_RGB2HSV)[0][0]
        dominant_hsv[1] = max(100, dominant_hsv[1])  # Adjust minimum saturation
        dominant_hsv[2] = max(100, dominant_hsv[2])  # Adjust minimum brightness
        
        # Convert back to RGB
        final_color = cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2RGB)[0][0]

        # Update LED color at the defined interval
        if time.time() - last_update_time > update_interval:
            try:
                print("Updating LED color to:", final_color)
                unique_rgb = (final_color[0], final_color[1], final_color[2])
                color_name = get_color_name(unique_rgb)
                print(f"The color name for RGB {unique_rgb} is {color_name}.")
                turn_on_set_light(final_color.tolist())
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
