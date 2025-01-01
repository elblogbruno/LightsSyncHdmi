import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def smooth_color(prev_color, new_color, factor=0.1):
    return prev_color * (1 - factor) + new_color * factor

def calculate_brightness(color):
    return np.sqrt(0.299 * color[0]**2 + 0.587 * color[1]**2 + 0.114 * color[2]**2)

def calculate_ww_values(color):
    # Ejemplo simple: ajustar los valores WW en función del brillo del color
    brightness = calculate_brightness(color)
    ww_value = int((brightness / 255) * 255)
    return [ww_value, ww_value]

def get_dominant_color_kmeans(frame, prev_dominant_color):
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    small_frame = cv2.resize(blurred_frame, (320, 240))

    height, width, _ = small_frame.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (width//4, height//4), (3*width//4, 3*height//4), 255, -1)
    masked_frame = cv2.bitwise_and(small_frame, small_frame, mask=mask)

    pixels = masked_frame.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=-1)]

    try:
        # Convert pixels to HSV color space for better clustering
        pixels_hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)

        kmeans = MiniBatchKMeans(n_clusters=16)  # Increase number of clusters
        kmeans.fit(pixels_hsv)
        cluster_centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        label_counts = np.bincount(labels)
        dominant_color_index = np.argmax(label_counts)

        dominant_color_hsv = cluster_centers[dominant_color_index]
        min_distance = np.linalg.norm(dominant_color_hsv - cv2.cvtColor(np.uint8([[prev_dominant_color]]), cv2.COLOR_BGR2HSV)[0][0])

        for i, center in enumerate(cluster_centers):
            distance = np.linalg.norm(center - cv2.cvtColor(np.uint8([[prev_dominant_color]]), cv2.COLOR_BGR2HSV)[0][0])
            if label_counts[i] > label_counts[dominant_color_index] or (label_counts[i] == label_counts[dominant_color_index] and distance < min_distance):
                dominant_color_hsv = center
                dominant_color_index = i
                min_distance = distance

        # Convert dominant color back to BGR
        dominant_color = cv2.cvtColor(np.uint8([[dominant_color_hsv]]), cv2.COLOR_HSV2BGR)[0][0]

        # Fallback mechanism for low saturation or brightness colors
        if dominant_color_hsv[1] < 50 or dominant_color_hsv[2] < 50:
            dominant_color = prev_dominant_color

        # convert to rgb
        dominant_color = dominant_color[::-1]

        return dominant_color
    except Exception as e:
        print(f"Error during KMeans clustering: {e}")
        return prev_dominant_color

def get_dominant_color_average(frame):
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    small_frame = cv2.resize(blurred_frame, (320, 240))
    average_color = np.mean(small_frame, axis=(0, 1))
    return average_color

def get_dominant_color_median(frame):
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    small_frame = cv2.resize(blurred_frame, (320, 240))
    median_color = np.median(small_frame, axis=(0, 1))
    return median_color

def get_dominant_color_mode(frame):
    blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
    small_frame = cv2.resize(blurred_frame, (320, 240))
    pixels = small_frame.reshape((-1, 3))
    pixels = pixels[np.any(pixels != [0, 0, 0], axis=-1)]
    mode_color = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=pixels)
    return mode_color
