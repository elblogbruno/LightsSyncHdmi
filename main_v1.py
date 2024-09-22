import cv2
import numpy as np
import os
import dotenv
import time
from homeassistant_api import Client

dotenv.load_dotenv()

# Create a VideoCapture object for the capture card (0 for webcam)
cap = cv2.VideoCapture(0)

# Set frame width and height to a lower resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Parameters for smoothing
smoothing_factor = 0.1
prev_dominant_color = np.array([255, 255, 255])  # Initial color white

# Initialize Home Assistant API client
with Client(
    os.environ['HASSIO_HOST'],
    os.environ['HASSIO_TOKEN'],
) as client:

    light = client.get_domain("light")

    # Set initial LED color to red
    light.turn_on(entity_id="light.LedComedorSamsung", brightness=255, rgb_color=[255, 0, 0])
    time.sleep(5)
    light.turn_off(entity_id="light.LedComedorSamsung")

    # Process frames
    last_update_time = time.time()
    update_interval = 0.5  # Update LEDs every 0.5 seconds
    frame_counter_r = 0
    frame_counter = 0

    while True:
        print("Capturing frame")

        # Read the frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Save a frame for testing
        if frame_counter_r < 20:
            # Save the first frame as 'captured_frame.jpg'
            cv2.imwrite('captured_frame_original_'+str(frame_counter_r)+'.jpg', frame)
            print("Frame saved as 'captured_frame.jpg'")
            frame_counter_r += 1

        # Downscale frame for faster processing
        small_frame = cv2.resize(frame, (160, 120))

        # Extract the center region of the frame
        height, width, _ = small_frame.shape
        center_frame = small_frame[height//4: 3*height//4, width//4: 3*width//4]

        # Convert frame to RGB
        center_frame = cv2.cvtColor(center_frame, cv2.COLOR_BGR2RGB)

        # Save a frame for testing
        if frame_counter < 20:
            # Save the first frame as 'captured_frame.jpg'
            cv2.imwrite('captured_frame_'+str(frame_counter)+'.jpg', center_frame)
            print("Frame saved as 'captured_frame.jpg'")
            frame_counter += 1

        # Get the most dominant color using k-means
        pixels = np.float32(center_frame.reshape(-1, 3))
        n_colors = 3  # Fewer clusters for faster processing
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1), 10, cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dominant_color = palette[np.argmax(counts)]

        # Smooth the color transition by blending previous and current dominant colors
        dominant_color = smoothing_factor * dominant_color + (1 - smoothing_factor) * prev_dominant_color
        dominant_color = dominant_color.astype(int)  # Convert to integer

        # Update previous color for the next iteration
        prev_dominant_color = dominant_color

        # Ensure the color has enough brightness and saturation
        dominant_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0]
        dominant_hsv[1] = max(100, dominant_hsv[1])  # Ensure minimum saturation
        dominant_hsv[2] = max(100, dominant_hsv[2])  # Ensure minimum brightness

        # Convert back to RGB
        dominant_color = cv2.cvtColor(np.uint8([[dominant_hsv]]), cv2.COLOR_HSV2RGB)[0][0]

        # Update LED color at the defined interval
        if time.time() - last_update_time > update_interval:
            print("Updating LED color to:", dominant_color)
            light.turn_on(entity_id="light.LedComedorSamsung", brightness=255, rgb_color=dominant_color.tolist())
            last_update_time = time.time()

        # If 'q' is pressed, exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close display windows
    cap.release()
    cv2.destroyAllWindows()
