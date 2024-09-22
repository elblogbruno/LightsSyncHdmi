import cv2
import numpy as np
import os
import dotenv
import time
from homeassistant_api import Client

dotenv.load_dotenv()

# Create a VideoCapture object for the capture card (0 for webcam)
cap = cv2.VideoCapture(0)

# Set frame width and height for better color capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Parameters for smoothing
smoothing_factor = 0.2
prev_dominant_color = np.array([255, 255, 255])  # Initial color white

# Initialize Home Assistant API client
try:
    with Client(
        os.environ['HASSIO_HOST'],
        os.environ['HASSIO_TOKEN'],
    ) as client:

        light = client.get_domain("light")
        media_player = client.get_domain("media_player")

        # Function to check if TV is on
        def is_tv_on():
            try:
                tv = client.get_entity(entity_id="media_player.samsung_qn85ca_75_2")  # Replace with your TV entity ID
                print(tv)
                return tv.state.state == "on"
            except Exception as e:
                print(f"Error checking TV state: {e}")
                return False

        # Set initial LED color to red
        try:
            light.turn_on(entity_id="light.LedComedorSamsung", brightness=255, rgb_color=[255, 0, 0])
            time.sleep(5)
            light.turn_off(entity_id="light.LedComedorSamsung")
        except Exception as e:
            print(f"Error controlling lights: {e}")

        # Process frames
        last_update_time = time.time()
        update_interval = 0.1  # Update LEDs every 0.5 seconds
        frame_counter = 0  # For saving a frame once

        while True:
            # Check if the TV is off
            if not is_tv_on():
                print("Samsung TV is off. Pausing the script...")
                light.turn_off(entity_id="light.LedComedorSamsung")
                time.sleep(5)  # Wait 5 seconds before checking again
                continue  # Skip to the next iteration if the TV is off

            print("Capturing frame")

            # Read the frame
            ret, frame = cap.read()

            if not ret:
                print("Failed to grab frame")
                break

            # Downscale frame for faster processing
            small_frame = cv2.resize(frame, (320, 240))

            # Define regions of interest (ROIs)
            height, width, _ = small_frame.shape
            regions = {
                'top_left': small_frame[:height//2, :width//2],
                'top_right': small_frame[:height//2, width//2:],
                'bottom_left': small_frame[height//2:, :width//2],
                'bottom_right': small_frame[height//2:, width//2:],
                'center': small_frame[height//4: 3*height//4, width//4: 3*width//4]
            }

            dominant_colors = []

            # Process each region to find dominant color
            for region_name, region in regions.items():
                region_rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)

                # Get the most dominant color using k-means
                pixels = np.float32(region_rgb.reshape(-1, 3))
                n_colors = 3  # Fewer clusters for faster processing
                _, labels, palette = cv2.kmeans(pixels, n_colors, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1), 10, cv2.KMEANS_RANDOM_CENTERS)
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
            dominant_hsv[1] = max(100, dominant_hsv[1])  # Ensure minimum saturation
            dominant_hsv[2] = max(100, dominant_hsv[2])  # Ensure minimum brightness

            # Convert back to RGB
            final_color = cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2RGB)[0][0]

            # Update LED color at the defined interval
            if time.time() - last_update_time > update_interval:
                try:
                    print("Updating LED color to:", final_color)
                    light.turn_on(entity_id="light.LedComedorSamsung", brightness=255, rgb_color=final_color.tolist())
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
