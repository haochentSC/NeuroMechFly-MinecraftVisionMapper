import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from flygym.vision.retina import Retina

start_time = time.time()

image_path = "test2.jpg"
raw_image = cv2.imread(image_path)
if raw_image is None:
    raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

mid_col = raw_image.shape[1] // 2
left_image = raw_image[:, :mid_col, :]
right_image = raw_image[:, mid_col:, :]

# --- Initialize the Retina ---
retina = Retina()

left_resized = cv2.resize(left_image, (retina.ncols, retina.nrows), interpolation=cv2.INTER_NEAREST)
right_resized = cv2.resize(right_image, (retina.ncols, retina.nrows), interpolation=cv2.INTER_NEAREST)

left_fly_vision = retina.raw_image_to_hex_pxls(left_resized).astype(np.float32)
right_fly_vision = retina.raw_image_to_hex_pxls(right_resized).astype(np.float32)

left_brightness = np.sum(left_fly_vision[:, 1])
right_brightness = np.sum(right_fly_vision[:, 1])
print(f"Left Eye Brightness: {left_brightness}")
print(f"Right Eye Brightness: {right_brightness}")
#  average brightness and the normalized difference ratio.
avg_brightness = (left_brightness + right_brightness) / 2.0
if avg_brightness > 0:
    diff_ratio = (left_brightness - right_brightness) / avg_brightness
else:
    diff_ratio = 0
print(f"Brightness Difference Ratio: {diff_ratio:.3f}")

threshold = 0.1   
base_forward = 1.0
turning_scale = 5.0  
max_turn = 2.0     

if abs(diff_ratio) < threshold:
    movement = (0, base_forward, 0)
else:
    turning = -turning_scale * diff_ratio
    turning = np.clip(turning, -max_turn, max_turn)
    movement = (turning, base_forward, 0)

print(f"Advanced Movement Vector: {movement}")

end_time = time.time()
time_usage = end_time - start_time
print(f"Optimized Time Usage: {time_usage:.4f} seconds")
left_human = retina.hex_pxls_to_human_readable(left_fly_vision, color_8bit=True).max(axis=-1)
right_human = retina.hex_pxls_to_human_readable(right_fly_vision, color_8bit=True).max(axis=-1)

fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
axs[0].imshow(left_human, cmap='gray')
axs[0].set_title("Left Eye Vision")
axs[0].axis('off')
axs[1].imshow(right_human, cmap='gray')
axs[1].set_title("Right Eye Vision")
axs[1].axis('off')

plt.show()
