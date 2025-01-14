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

def get_dominant_color_kmeans(frame, prev_color, n_colors=1):
    try:
        # Verificar que el frame no está vacío
        if frame is None or frame.size == 0:
            print("Frame vacío detectado, retornando color previo")
            return prev_color

        # Verificar las dimensiones del frame
        if len(frame.shape) != 3:
            print("Frame inválido (dimensiones incorrectas), retornando color previo")
            return prev_color

        # Redimensionar frame para procesamiento más rápido
        small_frame = cv2.resize(frame, (32, 32))
        
        try:
            # Convertir a RGB (si falla, el frame podría estar corrupto)
            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(f"Error al convertir color: {e}")
            return prev_color

        # Reshape para KMeans
        pixels = small_frame_rgb.reshape(-1, 3)
        pixels = np.float32(pixels)

        # Criterios de parada para K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        # Aplicar K-means
        _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convertir centros a enteros
        centers = np.uint8(centers)
        
        # Retornar el color dominante
        dominant_color = centers[0]
        
        return dominant_color

    except Exception as e:
        print(f"Error en get_dominant_color_kmeans: {e}")
        return prev_color

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
