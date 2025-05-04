import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
from flygym import Fly, Camera, SingleFlySimulation
from flygym.examples.locomotion import PreprogrammedSteps, RuleBasedController
from flygym.preprogrammed import all_leg_dofs
from tqdm import trange

# ----- Setup Output Directory -----
output_dir = Path("./outputs/rule_based_controller")
output_dir.mkdir(parents=True, exist_ok=True)

# ----- Preprogrammed Steps -----
preprogrammed_steps = PreprogrammedSteps()

# ----- Construct the Rules Graph -----
edges = {
    "rule1": {"LM": ["LF"], "LH": ["LM"], "RM": ["RF"], "RH": ["RM"]},
    "rule2": {
        "LF": ["RF"],
        "LM": ["RM", "LF"],
        "LH": ["RH", "LM"],
        "RF": ["LF"],
        "RM": ["LM", "RF"],
        "RH": ["LH", "RM"],
    },
    "rule3": {
        "LF": ["RF", "LM"],
        "LM": ["RM", "LH"],
        "LH": ["RH"],
        "RF": ["LF", "RM"],
        "RM": ["LM", "RH"],
        "RH": ["LH"],
    },
}

rules_graph = nx.MultiDiGraph()
for rule_type, mapping in edges.items():
    for src, tgt_nodes in mapping.items():
        for tgt in tgt_nodes:
            if rule_type == "rule1":
                detailed_rule = rule_type
            else:
                side = "ipsi" if src[0] == tgt[0] else "contra"
                detailed_rule = f"{rule_type}_{side}"
            rules_graph.add_edge(src, tgt, rule=detailed_rule)

def filter_edges(graph, rule, src_node=None):
    return [(src, tgt) for src, tgt, r in graph.edges(data="rule")
            if (r == rule) and (src_node is None or src == src_node)]

# ----- Simulation Parameters -----
run_time = 1.0        # seconds
timestep = 1e-4       # simulation timestep
weights = {
    "rule1": -10,
    "rule2_ipsi": 2.5,
    "rule2_contra": 1,
    "rule3_ipsi": 3.0,
    "rule3_contra": 2.0,
}

# ----- Initialize Rule-Based Controller -----
controller = RuleBasedController(
    timestep=timestep,
    rules_graph=rules_graph,
    weights=weights,
    preprogrammed_steps=preprogrammed_steps,
)

# ----- Setup the Fly and Simulation Environment -----
fly = Fly(
    init_pose="stretch",
    actuated_joints=all_leg_dofs,
    control="position",
    enable_adhesion=True,
    draw_adhesion=True,
)
# Use the standard Camera with only supported arguments.
cam = Camera(fly=fly, play_speed=0.1)
sim = SingleFlySimulation(
    fly=fly,
    cameras=[cam],
    timestep=timestep,
)
obs, info = sim.reset()

# ----- Main Simulation Loop -----
num_steps = int(run_time / sim.timestep)
for i in trange(num_steps):
    controller.step()
    joint_angles = []
    adhesion_onoff = []
    for leg, phase in zip(controller.legs, controller.leg_phases):
        joint_angles_arr = preprogrammed_steps.get_joint_angles(leg, phase)
        joint_angles.append(joint_angles_arr.flatten())
        adhesion_onoff.append(preprogrammed_steps.get_adhesion_onoff(leg, phase))
    action = {
        "joints": np.concatenate(joint_angles),
        "adhesion": np.array(adhesion_onoff),
    }
    obs, reward, terminated, truncated, info = sim.step(action)
    sim.render()
    if terminated or truncated:
        obs, _ = sim.reset()

# ----- Save the Simulation Video -----
cam.save_video(output_dir / "rule_based_controller.mp4")

