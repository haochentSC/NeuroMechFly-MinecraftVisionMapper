import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from flygym.vision.retina import Retina

# --- Helpers ---------------------------------------------------------------
def has_cuda():
    return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

def cuda_resize_np(img_np, dsize, interpolation=cv2.INTER_NEAREST):
    """Resize a NumPy HxWxC image on the GPU and return a NumPy result."""
    g = cv2.cuda_GpuMat()
    g.upload(img_np)
    g_res = cv2.cuda.resize(g, dsize, interpolation=interpolation)
    return g_res.download()

def maybe_cuda_cvt_color_bgr2rgb(img_bgr, use_cuda):
    """Optional GPU BGR->RGB; returns NumPy RGB image."""
    if not use_cuda:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    g = cv2.cuda_GpuMat()
    g.upload(img_bgr)
    g_rgb = cv2.cuda.cvtColor(g, cv2.COLOR_BGR2RGB)
    return g_rgb.download()

def resize_img(img, size_wh, use_cuda, interpolation=cv2.INTER_NEAREST):
    """Resize on GPU if available, else CPU."""
    if use_cuda:
        return cuda_resize_np(img, size_wh, interpolation=interpolation)
    else:
        return cv2.resize(img, size_wh, interpolation=interpolation)

# --- Main ------------------------------------------------------------------
start_time = time.time()

image_path = "test2.jpg"
raw_bgr = cv2.imread(image_path)
if raw_bgr is None:
    raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")

USE_CUDA = has_cuda()
print(f"[INFO] OpenCV CUDA available: {USE_CUDA}")

# Convert to RGB (optionally on GPU)
raw_image = maybe_cuda_cvt_color_bgr2rgb(raw_bgr, USE_CUDA)

# Split into left/right halves (NumPy)
mid_col = raw_image.shape[1] // 2
left_image  = raw_image[:, :mid_col, :]
right_image = raw_image[:, mid_col:, :]

# Retina target size
retina = Retina()
target_size = (retina.ncols, retina.nrows)  # (width, height)

# --- GPU-accelerated resize (with CPU fallback) ----------------------------
left_resized  = resize_img(left_image,  target_size, USE_CUDA, interpolation=cv2.INTER_NEAREST)
right_resized = resize_img(right_image, target_size, USE_CUDA, interpolation=cv2.INTER_NEAREST)

# Retina transforms (NumPy/CPU)
left_fly_vision  = retina.raw_image_to_hex_pxls(left_resized).astype(np.float32)
right_fly_vision = retina.raw_image_to_hex_pxls(right_resized).astype(np.float32)

left_brightness  = np.sum(left_fly_vision[:, 1])
right_brightness = np.sum(right_fly_vision[:, 1])

print(f"Left Eye Brightness:  {left_brightness}")
print(f"Right Eye Brightness: {right_brightness}")

if left_brightness > right_brightness:
    movement = (-1, 0, 0)  # Move left
elif right_brightness > left_brightness:
    movement = (1, 0, 0)   # Move right
else:
    movement = (0, 1, 0)   # Move forward if balanced

print(f"Movement Vector: {movement}")

end_time = time.time()
time_usage = end_time - start_time
print(f"Optimized Time Usage: {time_usage:.4f} seconds")

# Human-readable views for plotting (still CPU)
left_human_vision  = retina.hex_pxls_to_human_readable(left_fly_vision,  color_8bit=True).max(axis=-1)
right_human_vision = retina.hex_pxls_to_human_readable(right_fly_vision, color_8bit=True).max(axis=-1)

fig, axs = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
axs[0].imshow(left_human_vision, cmap='gray')
axs[0].set_title("Left Eye Vision")
axs[0].axis('off')

axs[1].imshow(right_human_vision, cmap='gray')
axs[1].set_title("Right Eye Vision")
axs[1].axis('off')

plt.show()