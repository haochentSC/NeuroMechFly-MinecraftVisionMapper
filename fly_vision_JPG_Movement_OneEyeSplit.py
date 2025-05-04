import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from flygym.vision.retina import Retina


start_time = time.time()
# Load and prepare the image
image_path = "test2.jpg"
raw_image = cv2.imread(image_path)
if raw_image is None:
    raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")
raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
retina = Retina()
resized_image = cv2.resize(raw_image, (retina.ncols, retina.nrows))

# Process the full image through the retina to obtain the fly vision data
fly_vision = retina.raw_image_to_hex_pxls(resized_image)
print("Fly Vision Data Shape:", fly_vision.shape)  # Expected: (721, 2)

human_vision = retina.hex_pxls_to_human_readable(fly_vision, color_8bit=True)
human_vision_gray = human_vision.max(axis=-1)

mid_point = human_vision_gray.shape[1]// 2
vision_left =human_vision_gray[:, :mid_point]
vision_right = human_vision_gray[:, mid_point:]

# Calculate brightness (sum of pixel values) for each half
left_brightness = np.sum(vision_left)
right_brightness =np.sum(vision_right)

print("Left Eye Brightness:", left_brightness)
print("Right Eye Brightness:", right_brightness)

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

# Plot the splitted vision images for visualization
fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
axs[0].imshow(vision_left, cmap='gray')
axs[0].set_title("Left Eye Vision")
axs[0].axis('off')
axs[1].imshow(vision_right, cmap='gray')
axs[1].set_title("Right Eye Vision")
axs[1].axis('off')
plt.show()



