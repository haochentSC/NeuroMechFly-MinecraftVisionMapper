import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import trange

import gymnasium as gym
from gymnasium import spaces
from flygym import Fly, Camera, SingleFlySimulation, get_data_path
from flygym.preprogrammed import all_leg_dofs


class FlySandboxEnv(gym.Env):
    """A sandbox environment where the fly moves based on pre-recorded kinematic data."""

    def __init__(self, run_time=1, timestep=1e-4):
        super().__init__()

        self.run_time = run_time
        self.timestep = timestep
        self.actuated_joints = all_leg_dofs

        # load recorded kinematics that are included with the FlyGym package
        data_path = get_data_path("flygym", "data")
        behavior_file = data_path / "behavior" / "210902_pr_fly1.pkl"
        
        try:
            with open(behavior_file, "rb") as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing behavior file: {behavior_file}")

        # The dataset is provided at 2000 Hz. We will try to run the simulation at 10000 Hz, so let’s interpolate it 5x
        self.target_num_steps = int(self.run_time / self.timestep)
        self.data_block = np.zeros((len(self.actuated_joints), self.target_num_steps))
        input_t = np.arange(len(self.data["joint_LFCoxa"])) * self.data["meta"]["timestep"]
        output_t = np.arange(self.target_num_steps) * self.timestep
        for i, joint in enumerate(self.actuated_joints):
            self.data_block[i, :] = np.interp(output_t, input_t, self.data[joint])
        #note: skipped the  time series of DoF angles, see https://neuromechfly.org/tutorials/gym_basics_and_kinematic_replay.html on dof angles time stamp
        # Define Gym observation & action spaces
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 42), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.actuated_joints),), dtype=np.float32)

        # Initialize Fly Simulation
        self.fly = Fly(init_pose="stretch", actuated_joints=self.actuated_joints, control="position")
        self.cam = Camera(fly=self.fly, play_speed=0.2, draw_contacts=True)
        self.sim = SingleFlySimulation(fly=self.fly, cameras=[self.cam])

    def reset(self, seed=None):
        """Resets the environment and returns initial observation."""
        super().reset(seed=seed)
        obs, info = self.sim.reset()
        self.current_step = 0
        return obs, info

    def step(self, action):
        """Executes a step based on the given action (kinematic replay)."""
        if self.current_step >= self.target_num_steps:
            return self.sim.observation, 0, True, False, {}

        # Use pre-recorded joint angles as action
        joint_pos = self.data_block[:, self.current_step]
        action = {"joints": joint_pos}
        obs, reward, terminated, truncated, info = self.sim.step(action)

        self.current_step += 1
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Renders the simulation."""
        self.sim.render()

    def close(self):
        """Closes the environment properly."""
        self.sim.close()


# ===================== RUNNING THE ENVIRONMENT =====================
if __name__ == "__main__":
    env = FlySandboxEnv()

    obs, info = env.reset()
    print("Starting Simulation:")

    for step in range(1000): # let's simulate 1000 steps max
        action = np.random.uniform(-1, 1, size=(len(env.actuated_joints),))   # your controller decides what to do based on obs: random
        obs, reward, terminated, truncated, info = env.step(action)
        #print(f"Step {step}: Reward = {reward}, Terminated = {terminated}") 
        env.render()
        if terminated or truncated:
            print("Simulation Ended Early.")
            break

    # Save the rendered video
    output_dir = Path("outputs/gym_basics/")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    env.sim.cameras[0].save_video(output_dir / "fly_simulation.mp4") 

    print(f"Simulation video saved at: {output_dir / 'fly_simulation.mp4'}")

    env.close()
    print("Simulation finished")