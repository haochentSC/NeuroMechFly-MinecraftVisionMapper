import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from flygym.vision.retina import Retina

start_time = time.time()
image_path = "test.jpg"
raw_image = cv2.imread(image_path)
if raw_image is None:
    raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
# Initialize the Retina (using default parameters)
retina = Retina()
# Split the raw image into left and right halves before resizing
height, width, _ = raw_image.shape
mid_point = width // 2
left_raw = raw_image[:, :mid_point, :]
right_raw = raw_image[:, mid_point:, :]
left_resized = cv2.resize(left_raw, (retina.ncols, retina.nrows))
right_resized = cv2.resize(right_raw, (retina.ncols, retina.nrows))
left_resized = np.ascontiguousarray(left_resized)
right_resized = np.ascontiguousarray(right_resized)
vision_left = retina.raw_image_to_hex_pxls(left_resized)
vision_right = retina.raw_image_to_hex_pxls(right_resized)
human_left = retina.hex_pxls_to_human_readable(vision_left, color_8bit=True)
human_right = retina.hex_pxls_to_human_readable(vision_right, color_8bit=True)
human_left_gray = human_left.max(axis=-1)
human_right_gray = human_right.max(axis=-1)

# Calculate brightness (sum of pixel values) for each eye
left_brightness = np.sum(human_left_gray)
right_brightness = np.sum(human_right_gray)

print("Left Eye Brightness:", left_brightness)
print("Right Eye Brightness:", right_brightness)

# --- Movement Logic Based on Brightness ---
if left_brightness > right_brightness:
    movement = (-1, 0, 0)  # Move left
elif right_brightness > left_brightness:
    movement = (1, 0, 0)   # Move right
else:
    movement = (0, 1, 0)   # Move forward if balanced

print(f"Movement Vector: {movement}")

end_time = time.time()
time_usage = end_time - start_time
print(f"Time Usage: {time_usage:.4f} seconds")

# --- Visualization of the Splitted Vision ---

fig, axs = plt.subplots(1, 2, figsize=(10, 4), tight_layout=True)
axs[0].imshow(human_left_gray, cmap="gray")
axs[0].set_title("Left Eye Vision")
axs[0].axis("off")
axs[1].imshow(human_right_gray, cmap="gray")
axs[1].set_title("Right Eye Vision")
axs[1].axis("off")
plt.show()
