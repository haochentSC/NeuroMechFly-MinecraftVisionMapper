from dm_control import mjcf
from flygym.arena.base import BaseArena
from flygym import Fly, Camera, SingleFlySimulation
from flygym.preprogrammed import all_leg_dofs
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========== Custom Arena Definition ====================
class SingleBlockArena(BaseArena):
    """
    A custom arena with a flat floor and a single square block.
    """
    def __init__(self, block_size=50, block_height=20):
        super().__init__()
        self.block_size = block_size
        self.block_height = block_height
        self._build_arena()

    def _build_arena(self):
        self.root_element = mjcf.RootElement(model="single_block_arena")
        worldbody = self.root_element.worldbody

        # Flat floor
        worldbody.add(
            "geom",
            name="floor",
            type="plane",
            size=[500, 500, 0.1],
            pos=[0, 0, 0],
            rgba=[0.8, 0.8, 0.8, 1]
        )

        # Center block
        half_block = self.block_size / 2.0
        half_height = self.block_height / 2.0
        worldbody.add(
            "geom",
            name="single_block",
            type="box",
            size=[half_block, half_block, half_height],
            pos=[0, 0, half_height],
            rgba=[0.5, 0.2, 0.2, 1]
        )

    def get_model(self):
        return self.root_element

    def _get_max_floor_height(self):
        return 0.0

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

# ==================== Arena Visualization ====================
if __name__ == "__main__":
    # Create arena and fly
    arena = SingleBlockArena(block_size=50, block_height=20)
    fly = Fly(init_pose="stretch", control="position")

    # Top-down fixed camera parameters
    cam_params = {
        "mode": "fixed",
        "pos": (0, -150, 100),  # behind and above the fly
        "euler": (np.deg2rad(60), 0, 0),  # tilt downward
        "fovy": 60
    }

    # Attach the camera
    cam = Camera(
        attachment_point=fly.model.worldbody,
        camera_name="cam_top",
        targeted_fly_names=fly.name,
        camera_parameters=cam_params,
        play_speed=1.0,
        draw_contacts=False,
        window_size=(800, 600)
    )

    # Create simulation
    sim = SingleFlySimulation(fly=fly, cameras=[cam], arena=arena)

    # Step a few times just to generate frames
    for _ in range(100):
        sim.step({"joints": [0.0] * len(all_leg_dofs)})
        sim.render()

    # Save output directory
    output_dir = Path("outputs/arena_preview/")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save video
    cam.save_video(output_dir / "arena_preview.mp4")
    print(f"Video saved at: {output_dir / 'arena_preview.mp4'}")

    # Save single snapshot
    plt.imshow(cam._frames[-1])
    plt.axis("off")
    plt.savefig(output_dir / "arena_snapshot.png", bbox_inches="tight", pad_inches=0)
    print(f"Snapshot saved at: {output_dir / 'arena_snapshot.png'}")

    sim.close()
