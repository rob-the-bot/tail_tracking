# %% imports
from pathlib import Path

import numpy as np
import napari
import tifffile

# %%
parent_folder = Path("..") / "20231102" / "tail"
images = tifffile.imread(
    parent_folder
    / "231102_pd60_h2bGCAMP7f_6dpf_63Hz_tail_natural_sleep_2p_1_MMStack_Pos0.ome.tif",
)

MIN, MAX = 3000, 20000
images = np.clip(images[1:, 10:73, :68], MIN, MAX)

# %%

%store -r bgd

for y in range(35, 45):
    for x in range(60):
        bgd[y, x] = np.random.randn() * 500 + 3000

images2 = images - bgd

# %%

viewer = napari.Viewer()
viewer.add_image(images, contrast_limits=[3000, 20000])
viewer.add_image(bgd)
viewer.add_image(images2, contrast_limits=[3000, 20000])
napari.run()

# %%
