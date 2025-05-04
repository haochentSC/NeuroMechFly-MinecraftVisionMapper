import anvil
from pathlib import Path
from dm_control import mjcf
from flygym.arena.base import BaseArena

REGION_PATH = Path("r.0.0.mca")
CHUNK_X, CHUNK_Z = 0, 0

def extract_surface_blocks(region_path: Path, chunk_x: int, chunk_z: int):
    """
    Reads the specified region file, loads the chunk at (chunk_x, chunk_z),
    and returns a list of (world_x, y, world_z, block_id) for the first
    non-air block in each column.
    """
    if not region_path.exists():
        raise FileNotFoundError(f"Region file not found: {region_path}")

    with open(region_path, "rb") as f:
        region = anvil.Region.from_file(f)

    try:
        chunk = region.get_chunk(chunk_x, chunk_z)
    except anvil.errors.ChunkNotFound:
        raise RuntimeError(f"Chunk at ({chunk_x}, {chunk_z}) not found in region.")

    surface_blocks = []
    for local_x in range(16):
        for local_z in range(16):
            world_x = chunk_x * 16 + local_x
            world_z = chunk_z * 16 + local_z

            # Search from top (255) down for first non-air block
            for y in range(255, -1, -1):
                block = chunk.get_block(local_x, y, local_z)
                if block.id != "air":
                    surface_blocks.append((world_x, y, world_z, block.id))
                    break
    return surface_blocks

# ----------------- Arena Builder ----------------
class MCAArena(BaseArena):
    """
    Builds an MJCF arena with blocks placed according to surface_blocks.
    """
    def __init__(self,
                 surface_blocks,
                 block_size: float = 10,
                 block_height: float = 10):
        super().__init__()
        self.surface_blocks = surface_blocks
        self.block_size = block_size
        self.block_height = block_height
        self._build_model()

    def _build_model(self):
        # Create root element
        self.root_element = mjcf.RootElement(model="mca_arena")
        worldbody = self.root_element.worldbody

        # Base floor plane
        worldbody.add(
            "geom", name="floor", type="plane",
            size=[500, 500, 0.1], pos=[0, 0, 0], rgba=[0.9, 0.9, 0.9, 1]
        )

        # Add a box for each surface block
        for x, y, z, block_type in self.surface_blocks:
            # Color by block type
            color = (
                (0.3, 0.6, 0.3, 1) if "grass" in block_type else
                (0.5, 0.3, 0.1, 1) if "dirt"  in block_type else
                (0.5, 0.5, 0.5, 1)
            )

            # Position in MJCF meters (or mm, depending on your scale)
            xpos = x * self.block_size
            ypos = z * self.block_size
            zpos = self.block_height / 2.0

            worldbody.add(
                "geom",
                name=f"{block_type}_{x}_{z}",
                type="box",
                size=[self.block_size/2.0, self.block_size/2.0, self.block_height/2.0],
                pos=[xpos, ypos, zpos],
                rgba=color
            )

    def get_model(self):
        return self.root_element

    def _get_max_floor_height(self):
        return self.block_height

    def get_spawn_position(self, rel_pos, rel_angle):
        return rel_pos, rel_angle

# -------------------- Main ----------------------
def main():
    # 1) Extract blocks
    blocks = extract_surface_blocks(REGION_PATH, CHUNK_X, CHUNK_Z)
    print(f"Extracted {len(blocks)} surface blocks:")
    for b in blocks[:5]:  # show first few
        print(" ", b)

    # 2) Build arena
    arena = MCAArena(blocks, block_size=10, block_height=10)

    # 3) Save MJCF for inspection
    out_dir = Path("out_mjcf")
    out_dir.mkdir(exist_ok=True)
    xml_path = out_dir / "mca_arena.xml"
    xml_path.write_text(arena.get_model().to_xml_string())
    print(f"MJCF arena written to: {xml_path}")

if __name__ == "__main__":
    main()
