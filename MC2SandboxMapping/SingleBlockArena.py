from dm_control import mjcf
from flygym.arena.base import BaseArena


class SingleBlockArena(BaseArena):
    """
    A custom arena with a flat floor and a single square block.
    
    Parameters:
        block_size (float): Length of each side of the block (in mm).
        block_height (float): Height of the block (in mm).
    """
    def __init__(self, block_size=50, block_height=20):
        super().__init__()
        self.block_size = block_size  # size (width/length) of the block
        self.block_height = block_height  # height of the block
        self._build_arena()

    def _build_arena(self):
        # Create the MJCF root element for the arena.
        self.root = mjcf.RootElement(model="single_block_arena")
        
        worldbody = self.root.worldbody
        
        # Create the flat floor (plane geom).
        worldbody.add(
            "geom",
            name="floor",
            type="plane",
            size=[500, 500, 0.1],  # Make the floor large enough
            pos=[0, 0, 0],
            rgba=[0.8, 0.8, 0.8, 1]
        )
        
        # Add a single square block (box geom) at the center of the arena.
        # The size parameter for a box is half-lengths along each axis.
        half_block = self.block_size / 2.0
        half_height = self.block_height / 2.0
        worldbody.add(
            "geom",
            name="single_block",
            type="box",
            size=[half_block, half_block, half_height],
            pos=[0, 0, half_height],  # Position so that the block sits on the floor
            rgba=[0.5, 0.2, 0.2, 1]
        )

    def get_model(self):
        return self.root
