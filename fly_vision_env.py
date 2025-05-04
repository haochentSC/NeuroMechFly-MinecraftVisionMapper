import numpy as np
import matplotlib.pyplot as plt
from flygym import Fly, Camera, SingleFlySimulation

fly = Fly(enable_vision=True)
cam = Camera(fly=fly, play_speed=1.0)

sim = SingleFlySimulation(fly=fly, cameras=[cam])  #  Initialize simulation
obs, _ = sim.reset()  #  Reset simulation, which also resets Fly
vision_data = obs["vision"]  # (2, 721, 2) -> Left & Right Eye Vision Data

left_eye_brightness = np.sum(vision_data[0])  # Total light intensity in left eye
right_eye_brightness = np.sum(vision_data[1])  # Total light intensity in right eye

if left_eye_brightness > right_eye_brightness:
    movement = (-1, 0, 0)  # Move left
elif right_eye_brightness > left_eye_brightness:
    movement = (1, 0, 0)  # Move right
else:
    movement = (0, 1, 0)  # Move forward if both eyes are balanced

print(f"Movement Vector: {movement}")  # (dx, dy, dz)

#Visualize Vision Input
vision_left = fly.retina.hex_pxls_to_human_readable(vision_data[0], color_8bit=True)
vision_right = fly.retina.hex_pxls_to_human_readable(vision_data[1], color_8bit=True)
fig, axs = plt.subplots(1, 2, figsize=(6, 3), tight_layout=True)
axs[0].imshow(vision_left.max(axis=-1), cmap="gray")  # Left Eye
axs[0].set_title("Left Eye Vision")
axs[0].axis("off")
axs[1].imshow(vision_right.max(axis=-1), cmap="gray")  # Right Eye
axs[1].set_title("Right Eye Vision")
axs[1].axis("off")
plt.show()

