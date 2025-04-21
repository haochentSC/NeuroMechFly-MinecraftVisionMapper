import anvil
from pathlib import Path

# Specify the path to your .mca file
region_path = Path("r.0.0.mca")  # Replace with your actual file path
region = anvil.Region.from_file(region_path)

# Define the chunk coordinates within the region (0 to 31)
chunk_x = 0
chunk_z = 0

try:
    # Load the specified chunk
    chunk = region.get_chunk(chunk_x, chunk_z)
except anvil.errors.ChunkNotFound:
    print(f"Chunk at ({chunk_x}, {chunk_z}) not found in the region.")
    exit()

surface_blocks = []

# Iterate over each column in the chunk (16x16)
for local_x in range(16):
    for local_z in range(16):
        world_x = chunk_x * 16 + local_x
        world_z = chunk_z * 16 + local_z

        # Scan from top to bottom to find the first non-air block
        for y in reversed(range(320)):  # Y range from 0 to 319 in modern Minecraft
            try:
                block = chunk.get_block(local_x, y, local_z)
                if block.id != "minecraft:air":
                    surface_blocks.append((world_x, y, world_z, block.id))
                    break  # Stop after finding the surface block
            except Exception as e:
                # Handle any exceptions (e.g., missing sections)
                continue

# Output the extracted surface blocks
for x, y, z, block_type in surface_blocks:
    print(f"Block at ({x}, {y}, {z}): {block_type}")
