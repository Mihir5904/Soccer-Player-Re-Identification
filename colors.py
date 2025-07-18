import cv2
import numpy as np
import colorsys


def is_color_in_range(rgb_color, min_hue, max_hue, min_saturation, max_saturation, min_value, max_value):
    # Convert RGB to HSV
    hsv_color = colorsys.rgb_to_hsv(rgb_color[0] / 255.0, rgb_color[1] / 255.0, rgb_color[2] / 255.0)
    hue = hsv_color[0] * 360
    sat = hsv_color[1]
    val = hsv_color[2]

    #print(f"RGB: {rgb_color} → HSV: ({hue:.1f}, {sat:.2f}, {val:.2f})")

    min_hue = min_hue / 360.0
    max_hue = max_hue / 360.0

    return min_hue <= hsv_color[0] <= max_hue and \
           min_saturation <= hsv_color[1] <= max_saturation and \
           min_value <= hsv_color[2] <= max_value



def create_bar(height, width, color):
    bar = np.zeros((height, width, 3), np.uint8)
    bar[:] = color
    red, green, blue = int(color[2]), int(color[1]), int(color[0])
    return bar, (red, green, blue)


def detect_color(img, field_color, number_clusters=4):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    height, width, _ = np.shape(img)
    data = np.reshape(img, (height * width, 3)).astype(np.float32)

    if data.shape[0] < number_clusters:
        return []

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(data, number_clusters, None, criteria, 10, flags)

    pixel_counts = np.bincount(labels.flatten())

    # Combine centers with their pixel counts and sort by count 
    sorted_clusters = sorted([(count, centers[i]) for i, count in enumerate(pixel_counts)], key=lambda x: x[0], reverse=True)

    detected_colors = []
    for count, rgb_center in sorted_clusters:
        rgb = (int(rgb_center[2]), int(rgb_center[1]), int(rgb_center[0])) # Convert BGR to RGB
        if not is_color_in_range(rgb, *field_color):
            detected_colors.append(rgb)
            
    return detected_colors

def get_ranged_groups(data, groups):
    sorted_data = {}
    for key, crange in groups.items():
        group = []
        for c in data:
            for r in crange:
                if is_color_in_range(c, *r):
                    group.append(c)
        if len(group) > 0:
            sorted_data[key] = group
    return sorted_data