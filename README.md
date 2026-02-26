# NeuroMechFly-MinecraftVisionMapper 
<img src="images/banner_snmall.jpg" width="600" />
<img src="images/banner_snmall.jpg" width="600" />

A sandbox environment for simulating and testing insect vision and movement pipelines using the FlyGym framework and Minecraft-based arenas.

---

## Installation

Follow the official NeuroMechFly installation guide for best results:

[https://neuromechfly.org/installation.html](https://neuromechfly.org/installation.html)

1. **Clone the repository**

   ```bash
   git clone https://github.com/haochentSC/flygym-sandbox.git
   cd flygym-sandbox
   ```

2. **Install FlyGym and dependencies**

   ```bash
   pip install "flygym"
   ```

3. **Activate the FlyGym environment**

   ```bash
   conda activate flygym
   ```

---

## Usage

### Run the Sandbox Environment

```bash
python fly_sandbox_env.py
```

### Process JPEG Frames

```bash
python fly_vision_readJPG.py --input path/to/frames --output processed_data.npy
```

### Capture Live RGB Images

```bash
python capture_RGB_image.py --duration 10 --output live_capture.npy
```

### Apply Rule-Based Controller

```bash
python vision_ruleBased_controller.py --env fly_vision_env --policy simple
```

Replace flags and paths as needed for your setup.

---

## File Documentation

### `capture_RGB_image.py`

* **Purpose:** Captures live RGB frames from the simulation environment.
* **Key Functions/Classes:**

  * `main(duration, output_path)`: Records frames for `duration` seconds and saves to `output_path`.
* **Usage Example:**

  ```bash
  python capture_RGB_image.py --duration 5 --output frames.npy
  ```

### `fly_vision_readJPG.py`

* **Purpose:** Loads and preprocesses JPEG frame sequences for analysis.
* **Key Functions/Classes:**

  * `load_frames(input_dir)`: Reads JPEGs from `input_dir`.
  * `preprocess(frames)`: Applies resizing and normalization.
* **Usage Example:**

  ```bash
  python fly_vision_readJPG.py --input ./raw_frames --output preprocessed.npy
  ```
  ![test2](https://github.com/user-attachments/assets/2032e716-b474-4d8b-a24d-7e7f10f215ac)

<img width="1000" height="400" alt="fly_vision_JPG_Movement_output" src="https://github.com/user-attachments/assets/ef3f35c2-3d80-4cd3-b502-a285a4d1c995" />

### `fly_vision_JPG_Movement.py`

* **Purpose:** Demonstrates movement control using single-eye JPEG sequences.
* **Usage:** Modify input path and control parameters at the top of the script.

### `fly_vision_JPG_Movement_OneEyeSplit.py`

* **Purpose:** Splits left/right eye inputs and runs movement logic independently.

### `fly_vision_JPG_Movement_Quick.py`

* **Purpose:** A minimal prototype for rapid testing of binocular vision-to-action loops using JPEG input.  
* **Key Features:**
  * Splits an input image into left/right halves to simulate compound eyes.
  * Uses the FlyGym `Retina` model to convert images into hexagonal photoreceptor arrays.
  * Compares left vs. right brightness to decide a simple movement vector (left, right, forward).
  * Supports GPU-accelerated resizing and color conversion with OpenCV CUDA (fallback to CPU if unavailable).
  * Generates human-readable grayscale plots of left and right eye vision.  

### `fly_vision_Movement_advanced.py`

* **Purpose:** Advanced algorithms for complex movement behaviors.

### `fly_sandbox_env.py`

* **Purpose:** Sets up the main FlyGym sandbox environment with arena configuration.
* **Key Functions/Classes:**

  * `create_arena(blocks, size)`: Builds the Minecraft arena.
  * `run_simulation(config)`: Starts the DM Control loop.

### `fly_vision_env.py`

* **Purpose:** Wraps the sandbox with a Gymnasium-compatible vision API.
* **Key Classes:**

  * `FlyVisionEnv(gym.Env)`: Implements `step()`, `reset()`, and rendering.

### `vision_ruleBased_controller.py`

* **Purpose:** A simple rule-based policy that maps vision inputs to movement commands.
* **Key Functions:**

  * `decide_action(observation)`: Returns control signals based on image features.

---

## MC2SandboxMapping

The `MC2SandboxMapping` folder contains utilities and arena builders for mapping Minecraft region data to MuJoCo sandbox environments.

### `anvil_parser.py`

* **Purpose:** Generates a sample Minecraft region file (`.mca`) by populating blocks (stone or dirt) in a 16×16×16 cube.
* **Key Functions:**

  * Uses `anvil.EmptyRegion` to create regions and `region.set_block(...)` to assign random blocks.
  * `region.save('r.0.0.mca')` writes the file.
* **Usage Example:**

  ```bash
  python MC2SandboxMapping/anvil_parser.py
  ```

### `mca_surface_extraction.py`

* **Purpose:** Reads a Minecraft region file and extracts the topmost non-air block for each column.
* **Key Functions:**

  * `anvil.Region.from_file(f)` to load region.
  * Iterates over local X/Z and scans Y downward to find surface blocks.
  * Prints a list of `(world_x, y, world_z, block_id)` tuples.
* **Usage Example:**

  ```bash
  python MC2SandboxMapping/mca_surface_extraction.py
  ```

### `mca_to_mjcf_arena.py`

* **Purpose:** Converts extracted surface block data into a MuJoCo XML arena.
* **Key Classes:**

  * `extract_surface_blocks(region_path, chunk_x, chunk_z)`: Returns surface block list.
  * `MCAArena(BaseArena)`: Builds an MJCF model with box geoms for each block.
* **Usage Example:**

  ```bash
  python MC2SandboxMapping/mca_to_mjcf_arena.py
  ```

### `multiBlockArena.py`

* **Purpose:** Defines a demo arena with five box geoms arranged around the origin and renders a fly simulation.
* **Key Classes:**

  * `MultiCenterBlockArena(BaseArena)`: Builds floor plane plus five block geoms.
  * Uses `SingleFlySimulation` and `Camera` to capture frames and save outputs.
* **Usage Example:**

  ```bash
  python MC2SandboxMapping/multiBlockArena.py
  ```

### `sandbox_custom_arena.py`

* **Purpose:** Provides a customizable single-block arena with simulation and video/video preview capabilities.
* **Key Classes:**

  * `SingleBlockArena(BaseArena)`: Sets up a plane and one block.
  * Includes a `__main__` demo for recording simulation frames, saving MP4 and PNG snapshots.
* **Usage Example:**

  ```bash
  python MC2SandboxMapping/sandbox_custom_arena.py
  ```

### `SingleBlockArena.py`

* **Purpose:** A library module defining the `SingleBlockArena` class for use in custom simulations.
* **Key Classes:**

  * `SingleBlockArena(BaseArena)`: Implements `get_model()` to return an MJCF root with floor and block.
* **Usage Example:** Import and instantiate in your own scripts:

  ```python
  from MC2SandboxMapping.SingleBlockArena import SingleBlockArena
  arena = SingleBlockArena(block_size=30, block_height=10)
  ```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add feature description"`
4. Push to GitHub: `git push origin feature/my-feature`
5. Open a Pull Request describing your changes.

Adhere to PEP 8 style and include tests where appropriate.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Haochen Tong – [https://github.com/haochentSC](https://github.com/haochentSC)

Repository Link: [https://github.com/haochentSC/flygym-sandbox](https://github.com/haochentSC/flygym-sandbox)
