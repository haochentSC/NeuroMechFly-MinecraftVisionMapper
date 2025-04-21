import anvil
from random import choice

# Create a new region with the `EmptyRegion` class at 0, 0 (in region coords)
region = anvil.EmptyRegion(0, 0)

stone = anvil.Block('minecraft', 'stone') # Create `Block` objects that are used to set blocks
dirt = anvil.Block('minecraft', 'dirt')

for y in range(16): # Make a 16x16x16 cube of either stone or dirt blocks 
    for z in range(16):
        for x in range(16):
            region.set_block(choice((stone, dirt)), x, y, z)

# Save to a file
region.save('r.0.0.mca')