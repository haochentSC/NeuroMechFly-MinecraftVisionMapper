#!/usr/bin/env python
"""
multiBlockArena.py

1. Defines a MultiCenterBlockArena with five boxes around the origin.
2. Runs a SingleFlySimulation with a fixed top-down camera.
3. Captures every frame via physics.render() into a list.
4. Writes out an MP4 (imageio-ffmpeg) and a PNG snapshot.
"""

import os
# (No MUJOCO_GL override here—using default GLFW on Windows)

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import imageio  # pip install imageio imageio-ffmpeg

from dm_control import mjcf
from flygym.arena.base import BaseArena
from flygym import Fly, Camera, SingleFlySimulation
from flygym.preprogrammed import all_leg_dofs


class MultiCenterBlockArena(BaseArena):
    """Five blocks clustered around (0,0), flush with the floor plane."""
    def __init__(self, block_size=50, block_height=20):
        super().__init__()                 # must call before building MJCF
        self.block_size   = block_size
        self.block_height = block_height
        self._build_arena()

    def _build_arena(self):
        self.root_element = mjcf.RootElement(model="multi_center_arena")
        worldbody = self.root_element.worldbody

        # 1) Big flat floor
        worldbody.add(
            "geom",
            name="floor",
            type="plane",
            size=[500, 500, 0.1],
            pos=[0, 0, 0],
            rgba=[0.8, 0.8, 0.8, 1]
        )

        # 2) Precompute half‐sizes
        half_xy  = self.block_size  / 2.0
        half_z   = self.block_height / 2.0

        # 3) Offsets (in world mm) around the center
        offsets = [
            (0, 0),
            ( self.block_size,  0),
            (-self.block_size,  0),
            (0,  self.block_size),
            ( self.block_size, self.block_size),
        ]

        # 4) Add one box per offset
        for i, (dx, dy) in enumerate(offsets):
            worldbody.add(
                "geom",
                name=f"block_{i}",
                type="box",
                size=[half_xy, half_xy, half_z],
                pos=[dx, dy, half_z],  # so bottom sits at Z=0
                rgba=[0.5, 0.2, 0.2, 1]
            )

    def get_model(self):
        return self.root_element

    def _get_max_floor_height(self):
        # required abstract method
        return 0.0

    def get_spawn_position(self, rel_pos, rel_angle):
        # required abstract method
        return rel_pos, rel_angle


def main():
    # 1) Build arena & fly
    arena = MultiCenterBlockArena(block_size=50, block_height=20)
    fly   = Fly(init_pose="stretch", control="position")

    # 2) Top-down fixed camera
    cam_params = {
        "mode": "fixed",
        "pos":   (0, -150, 100),
        "euler": (-np.pi/2, 0.0, 0.0),  # rotate -90° about X → look straight down
        "fovy":  60
    }
    cam = Camera(
        attachment_point=arena.root_element.worldbody,
        camera_name="cam_top",
        targeted_fly_names=fly.name,
        camera_parameters=cam_params,
        play_speed=1.0,
        draw_contacts=False,
        window_size=(800, 608)  # make height divisible by 16 for video
    )

    # 3) Create the simulation
    sim = SingleFlySimulation(fly=fly, cameras=[cam], arena=arena)
    physics = sim.physics

    # 4) Resolve the true camera_id
    for cid in range(physics.model.ncam):
        if physics.model.id2name(cid, "camera").endswith("cam_top"):
            cam.camera_id = cid
            break

    # 5) Step & capture frames manually
    WIDTH, HEIGHT = 800, 608
    frames = []
    for _ in range(100):
        sim.step({"joints": [0.0] * len(all_leg_dofs)})
        img = physics.render(width=WIDTH, height=HEIGHT, camera_id=cam.camera_id)
        frames.append(img)

    # 6) Save outputs
    out_dir = Path("outputs/arena_preview/")
    out_dir.mkdir(exist_ok=True, parents=True)

    # a) MP4 via imageio
    video_path = out_dir / "arena_preview.mp4"
    imageio.mimwrite(str(video_path), frames, fps=30)
    print("Video saved at:", video_path)

    # b) PNG snapshot via matplotlib
    snapshot_path = out_dir / "arena_snapshot.png"
    plt.imsave(snapshot_path, frames[-1])
    print("Snapshot saved at:", snapshot_path)

    sim.close()


if __name__ == "__main__":
    main()
