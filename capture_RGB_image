import numpy as np
import matplotlib.pyplot as plt
from flygym import Fly, Camera, SingleFlySimulation


fly = Fly(enable_vision=True, render_raw_vision=True)

cam = Camera(fly=fly, play_speed=1.0)

sim = SingleFlySimulation(fly=fly, cameras=[cam])
obs, info = sim.reset()  # 'info' contains raw vision data

raw_rgb_left = info["raw_vision"][0]  # Left eye full RGB image
raw_rgb_right = info["raw_vision"][1]  # Right eye full RGB image

print("Raw RGB Image Shape (Left Eye):", raw_rgb_left.shape)  # Expected: (Height, Width, 3)
print("Raw RGB Image Shape (Right Eye):", raw_rgb_right.shape)

fig, axs = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
axs[0].imshow(raw_rgb_left.astype(np.uint8))  # Left Eye Image
axs[0].set_title("Left Eye Full RGB Image")
axs[0].axis("off")
axs[1].imshow(raw_rgb_right.astype(np.uint8))  # Right Eye Image
axs[1].set_title("Right Eye Full RGB Image")
axs[1].axis("off")
plt.show()
