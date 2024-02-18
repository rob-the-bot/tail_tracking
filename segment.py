# %% imports

from pathlib import Path

import av
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm.auto import tqdm

# %% Load in data

parent_folder = Path("..") / "20231102" / "tail"
images = tifffile.imread(
    parent_folder
    / "231102_pd60_h2bGCAMP7f_6dpf_63Hz_tail_natural_sleep_2p_1_MMStack_Pos0.ome.tif"
)

# %%

MIN, MAX = 3000, 20000
images = np.clip(images[1:, 10:72, :68], MIN, MAX)

# %% MOG Background estimation (optional)

backSub = cv2.createBackgroundSubtractorMOG2(
    history=5000, varThreshold=16, detectShadows=False
)

for img in tqdm(images[::5]):
    # update the background model and obtain foreground mask
    fg_mask = backSub.apply(img.astype(np.float32))


# %%

bgd = backSub.getBackgroundImage().copy()

for y in range(35, 45):
    for x in range(60):
        bgd[y, x] = 0


# %%  display

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
g = ax1.imshow(bgd, cmap="gray", vmin=MIN, vmax=MAX)
# set colorbar
cbar = plt.colorbar(g, cmap="gray", ax=ax1)

ax1.set_title("Background")
ax2.imshow(fg_mask, cmap="gray")
ax2.set_title("Foreground Mask")


# %% substracting the background

bgd = tifffile.imread(parent_folder / "background.tif")
images2 = np.clip(images - bgd.astype(int), 0, MAX).astype(np.uint16)

# %%

MIN, MAX = 5500, 10000


def scale(img, MIN, MAX, dtype=np.uint8):
    scaled = (np.clip(img, MIN, MAX) - MIN) / (MAX - MIN) * np.iinfo(dtype).max
    return scaled.astype(dtype)


# %% write to mp4 file

container = av.open(str(parent_folder / "tail.mp4"), "w")
stream = container.add_stream("h264", rate=30)
stream.width = images.shape[2]
stream.height = images.shape[1]
stream.pix_fmt = "gray"
stream.options = {"crf": "17", "vsync": "0", "tune": "film"}

for img in tqdm(images[::10]):

    frame = av.VideoFrame.from_ndarray(scale(img, MIN, MAX), format="gray")

    for packet in stream.encode(frame):
        container.mux(packet)


# flush
for packet in stream.encode():
    container.mux(packet)

# Close the file
container.close()

# %% connected component analysis

from typing import Union, Tuple
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
import logging


def get_tail(img):
    label_img = label(img, connectivity=1)
    regions = regionprops(label_img)

    tail_img = None
    for i, props in enumerate(regions):
        # test if point is inside bounding box
        if props.bbox[0] < 40 and props.bbox[2] > 40 and props.bbox[1] >= 0:
            tail_img = label_img == i + 1
            break

    if tail_img is None:
        logging.error("No tail found")
        return None, None

    return tail_img, skeletonize(tail_img)


def bin_img(
    img: np.ndarray, bgd: np.ndarray, th1: Union[int, float], th2: Union[int, float]
):
    """
    Binarize the image the thresholds, in combination with the background image
    For each pixel, if the pixel value is greater than th1 and the background pixel value is 0 (for the tail), it is set to 1
    If the pixel value is greater than th2 and the background value is 0, it is set to 1,
    otherwise it is set to 0
    """
    bgd_bin = bgd.astype(bool)
    img_bin1 = (img * bgd_bin) > th2
    img_bin2 = (img * ~bgd_bin) > th1

    return img_bin1 | img_bin2


# %% write to mp4 file

container = av.open(str(parent_folder / "tail.mp4"), "w")
stream = container.add_stream("h264", rate=30)
stream.width = images.shape[2]
stream.height = images.shape[1]
# stream.pix_fmt = "gray"
stream.options = {"crf": "17", "vsync": "0", "tune": "film"}

MIN, MAX = 5500, 10000

for img_orig, img in tqdm(zip(images[::200], images2[::200])):

    # need to do adaptive thresholding!!!
    tail_img, skeleton = get_tail(bin_img(img[:, 20:], bgd[:, 20:], 6000, 100))

    if tail_img is None:
        continue

    tail_points = np.array(np.where(skeleton)).T
    tail_points[:, 1] += 20

    frame_data = scale(img_orig, MIN, MAX)
    # convert to color
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
    # draw points using cv2
    for y, x in tail_points:
        frame_data = cv2.circle(frame_data, (x, y), 0, (255, 0, 0), 1)
    frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")

    for packet in stream.encode(frame):
        container.mux(packet)

# flush
for packet in stream.encode():
    container.mux(packet)

# Close the file
container.close()


# %% Define a function that takes a grayscale image and a seed pixel as input

from IPython import get_ipython

ipython = get_ipython()
ipython.magic("load_ext autoreload")
ipython.magic("autoreload 1")
ipython.magic("aimport search")

from search import search_path_dp


def search_path(image, seed, threshold: float = 0.1, max_length: int = 50):
    image = image.astype(np.float32)

    # Get the height and width of the image
    height, width = image.shape

    # Initialize the path as a list of the seed pixel
    path = [seed]

    # Initialize the current pixel as the seed pixel
    current = seed

    previous_direction = (0, 1)

    # Loop until the path reaches the maximum length or the right edge of the image
    while len(path) < max_length and current[1] < width - 1:
        # Get the current pixel intensity
        current_intensity = image[current]

        # Initialize the minimum pixel difference as infinity
        min_diff = np.inf

        # Initialize the next pixel as None
        next: Tuple[int, int] = None

        # Loop through the three possible directions: right, right-down, right-up
        for direction in [(0, 1), (1, 1), (-1, 1)]:
            # Calculate the next pixel coordinates by adding the direction to the current pixel
            next_x = current[0] + direction[0]
            next_y = current[1] + direction[1]

            # Check if the next pixel is within the image bounds
            if 0 <= next_x < height and 0 <= next_y < width:
                # Get the next pixel intensity
                next_intensity = image[next_x, next_y]

                # Calculate the pixel difference by subtracting the current and next intensities
                diff = abs(current_intensity - next_intensity)

                # Multiply the pixel difference by 2 if the direction is not right
                if direction[0] != previous_direction:
                    diff *= 2

                # Check if the pixel difference is smaller than the minimum difference
                if diff < min_diff:
                    # Update the minimum difference
                    min_diff = diff
                    # Update the next pixel
                    next = (next_x, next_y)
                    # Update the previous direction
                    previous_direction = direction

        # Check if the minimum difference is below the threshold
        if min_diff < current_intensity * threshold:
            # Add the next pixel to the path
            path.append(next)

            # Update the current pixel
            current = next
        else:
            # Break the loop
            break

    # Return the path
    return np.array(path)


# %%

container = av.open(str(parent_folder / "tail.mp4"), "w")
stream = container.add_stream("h264", rate=30)
stream.width = images.shape[2]
stream.height = images.shape[1]
# stream.pix_fmt = "gray"
stream.options = {"crf": "17", "vsync": "0", "tune": "film"}

MIN, MAX = 5500, 10000
seed = (39, 10)  # Define the seed pixel

for img in tqdm(images[::200]):

    tail_points = search_path_dp(img.astype(np.float32), seed)

    frame_data = scale(img, MIN, MAX)
    # convert to color
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
    # draw points using cv2
    for y, x in tail_points:
        frame_data = cv2.circle(frame_data, (x, y), 0, (255, 0, 0), 1)
    frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")

    for packet in stream.encode(frame):
        container.mux(packet)

# flush
for packet in stream.encode():
    container.mux(packet)

# Close the file
container.close()

# %% enumerate all possible paths in the next N steps, with itertools and product

from itertools import product


def get_path(image, seed, N=5, M=10):

    directions = [(0, 1), (1, 1), (-1, 1)]
    best_overall_path = [seed]

    for _ in range(M):
        all_paths = product(directions, repeat=N)
        best_cost = np.inf

        for path in all_paths:
            # calculate the absolute path
            path_abs = seed + np.cumsum(path, axis=0)

            # calculate the cost of the path
            path_intensity = image[path_abs[:, 0], path_abs[:, 1]]
            path_diff_sum = np.abs(np.diff(path_intensity)).sum()
            path_integral = path_intensity.sum()

            cost = path_diff_sum * 20 - path_integral

            if cost < best_cost:
                best_cost = cost
                best_path = path_abs

        # add the seed to the top of best path
        best_overall_path.append(best_path)
        # update the seed to the last point of the best path
        seed = best_path[-1]

    return np.vstack(best_overall_path)


image = images[200].astype(np.float32)
# Get the height and width of the image
height, width = image.shape

import functools


@functools.cache
def get_path_recursive(N, seed, previous_seed):
    cost = abs(image[seed] - image[previous_seed]) * 1 - image[seed]
    if N == 0:
        return cost, [seed]

    best_cost = float('inf')
    best_path = []

    for dx, dy in [(0, 1), (1, 1), (-1, 1)]:
        next_seed = (seed[0] + dx, seed[1] + dy)

        # Check if the next pixel is within the image bounds
        if 0 <= next_seed[0] < height and 0 <= next_seed[1] < width:
            next_cost, next_path = get_path_recursive(N - 1, next_seed, seed)
            total_cost = cost + next_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_path = [seed] + next_path

    return best_cost, best_path


seed = (39, 10)  # Define the seed pixel
cost1, best_path = get_path_recursive(10, seed, seed)
best_path = np.array(best_path)

print(cost1)
print(best_path)

plt.imshow(image, cmap="gray", vmin=MIN, vmax=MAX)
plt.plot(best_path[:, 1], best_path[:, 0], "r-")

# %%
path_abs = get_path(image, seed, N=10, M=1)

# calculate the cost of the path
path_intensity = image[path_abs[:, 0], path_abs[:, 1]]
path_diff_sum = np.abs(np.diff(path_intensity)).sum()
path_integral = path_intensity.sum()

cost = path_diff_sum - path_integral

print(cost1, cost)

# %%

container = av.open(str(parent_folder / "tail_tracked.mp4"), "w")
stream = container.add_stream("h264", rate=30)
stream.width = images.shape[2]
stream.height = images.shape[1]
# stream.pix_fmt = "gray"
stream.options = {"crf": "17", "vsync": "0", "tune": "film"}

container2 = av.open(str(parent_folder / "tail.mp4"), "w")
stream2 = container2.add_stream("hevc", rate=30)
stream2.width = images.shape[2]
stream2.height = images.shape[1]
stream2.pix_fmt = "gray"
stream2.options = {"crf": "22", "vsync": "0"}

MIN, MAX = 5500, 10000
seed = (39, 10)  # Define the seed pixel

for img in tqdm(images):
    # important for difference calculation!
    image = img.astype(np.float32)
    # important to clear the cache, since it may not recognize the new image
    get_path_recursive.cache_clear()
    cost1, best_path = get_path_recursive(45, seed, seed)
    tail_points = np.array(best_path)
    
    frame_data = scale(img, MIN, MAX)

    frame = av.VideoFrame.from_ndarray(frame_data, format="gray")

    for packet in stream2.encode(frame):
        container2.mux(packet)


    # convert to color
    frame_data = cv2.cvtColor(frame_data, cv2.COLOR_GRAY2RGB)
    # draw points using cv2
    for y, x in tail_points:
        frame_data = cv2.circle(frame_data, (x, y), 0, (255, 0, 0), 1)
    frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")

    for packet in stream.encode(frame):
        container.mux(packet)

# flush
for packet in stream.encode():
    container.mux(packet)
# flush
for packet in stream2.encode():
    container.mux(packet)
# Close the file
container.close()
container2.close()
