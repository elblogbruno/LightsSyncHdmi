import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def smooth_color(prev_color, new_color, factor=0.1):
    return prev_color * (1 - factor) + new_color * factor

def calculate_brightness(color):
    return np.sqrt(0.299 * color[0]**2 + 0.587 * color[1]**2 + 0.114 * color[2]**2)

def get_dominant_color(frame, prev_dominant_color):
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    small_frame = cv2.resize(blurred_frame, (320, 240))

    height, width, _ = small_frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (width//4, height//4), (3*width//4, 3*height//4), 255, -1)
    masked_frame = cv2.bitwise_and(small_frame, small_frame, mask=mask)

    pixels = masked_frame.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=-1)]

    try:
        kmeans = MiniBatchKMeans(n_clusters=8)
        kmeans.fit(pixels)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        label_counts = np.bincount(labels)
        dominant_color_index = np.argmax(label_counts)

        dominant_color = cluster_centers[dominant_color_index]
        min_distance = np.linalg.norm(dominant_color - prev_dominant_color)

        for i, center in enumerate(cluster_centers):
            distance = np.linalg.norm(center - prev_dominant_color)
            if label_counts[i] > label_counts[dominant_color_index] or (label_counts[i] == label_counts[dominant_color_index] and distance < min_distance):
                dominant_color = center
                dominant_color_index = i
                min_distance = distance

        dominant_color_hsv = cv2.cvtColor(np.uint8([[dominant_color]]), cv2.COLOR_BGR2HSV)[0][0]
        if dominant_color_hsv[1] < 50 or dominant_color_hsv[2] < 50:
            dominant_color = prev_dominant_color

        return dominant_color
    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        return prev_dominant_color
