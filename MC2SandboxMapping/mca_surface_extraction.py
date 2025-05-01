import anvil
from pathlib import Path

region_path = Path("r.0.0.mca")  
chunk_x = 0
chunk_z = 0

with open(region_path, "rb") as f:
    region = anvil.Region.from_file(f)

try:
    chunk = region.get_chunk(chunk_x, chunk_z)
except anvil.errors.ChunkNotFound:
    print(f"Chunk at ({chunk_x}, {chunk_z}) not found in the region.")
    exit()

surface_blocks = []

for local_x in range(16):
    for local_z in range(16):
        world_x = chunk_x * 16 + local_x
        world_z = chunk_z * 16 + local_z

        # Search from top down for first non-air block
        for y in reversed(range(256)):
            try:
                block = chunk.get_block(local_x, y, local_z)
                if block.id != "air":
                    surface_blocks.append((world_x, y, world_z, block.id))
                    break
            except Exception:
                continue


if not surface_blocks:
    print("No surface blocks found.")
else:
    print("Extracted surface blocks:")
    for x, y, z, block_id in surface_blocks:
        print(f"Block at ({x}, {y}, {z}): {block_id}")
