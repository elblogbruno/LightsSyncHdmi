from homeassistant_api import Client
# importing OpenCV library 
import cv2
import numpy as np
import os

import dotenv

dotenv.load_dotenv()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(2)


with Client(
    os.environ['HASSIO_HOST'],
    os.environ['HASSIO_TOKEN'],
) as client:

    light = client.get_domain("light")

    # light.turn_on(entity_id="light.LedComedorSamsung", brightness=255, rgb_color=[255, 0, 0])

    # import time
    # time.sleep(5)

    # light.turn_off(entity_id="light.LedComedorSamsung")

    while True:
        print("Capturing frame")
        # Read the frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get most dominant color
        pixels = np.float32(frame.reshape(-1, 3))
        n_colors = 5
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        _, counts = np.unique(labels, return_counts=True)
        dominant = palette[np.argmax(counts)]

        print("Dominant color: ", dominant)

        # convert to int
        dominant = [int(x) for x in dominant]

        light.turn_on(entity_id="light.habitacion", brightness=255, rgb_color=dominant)

        # The window is closed if q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()