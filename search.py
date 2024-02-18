# Import numpy and scipy for array operations and optimization
import numpy as np
import scipy.optimize as opt


# Define a function that calculates the pixel difference between two pixels
def pixel_diff(image, p1, p2):
    # Get the pixel intensities
    i1 = image[p1]
    i2 = image[p2]

    # Calculate the absolute difference
    diff = abs(i1 - i2)

    # Multiply the difference by 2 if the pixels are not in the same row
    if p1[0] != p2[0]:
        diff *= 2

    # Return the difference
    return diff


# Define a function that updates the table for a given pixel
def update_table(image, table, seed, pixel):
    # Get the row and column indices of the pixel
    r, c = pixel

    # Check if the pixel is already in the table
    if table[r, c, 0] is not None:
        # Return the table value
        return table[r, c]

    # Check if the pixel is the seed pixel
    if pixel == seed:
        # Set the table value to be the seed pixel and zero difference
        table[r, c] = ([seed], 0)
        # Return the table value
        return table[r, c]

    # Check if the pixel is on the left edge of the image
    if c == 0:
        # Set the table value to be an empty path and infinity difference
        table[r, c] = ([], np.inf)
        # Return the table value
        return table[r, c]

    # Initialize the minimum difference as infinity
    min_diff = np.inf

    # Initialize the optimal path as None
    opt_path = None

    # Loop through the three possible previous pixels: left, left-up, left-down
    for prev in [(r, c - 1), (r - 1, c - 1), (r + 1, c - 1)]:
        # Check if the previous pixel is within the image bounds
        if 0 <= prev[0] < height and 0 <= prev[1] < width:
            # Update the table for the previous pixel recursively
            prev_path, prev_diff = update_table(image, table, seed, prev)

            # Calculate the pixel difference between the previous and current pixels
            diff = pixel_diff(image, prev, pixel)

            # Calculate the total difference by adding the previous and current differences
            total_diff = prev_diff + diff

            # Check if the total difference is smaller than the minimum difference
            if total_diff < min_diff:
                # Update the minimum difference
                min_diff = total_diff

                # Update the optimal path by appending the current pixel to the previous path
                opt_path = prev_path + [pixel]

    # Set the table value to be the optimal path and minimum difference
    table[r, c] = (opt_path, min_diff)

    # Return the table value
    return table[r, c]


# Define a function that evaluates the pixel difference for a given sequence of pixels
def evaluate(image, sequence):
    # Initialize the total difference as zero
    total_diff = 0

    # Loop through the sequence of pixels
    for i in range(len(sequence) - 1):
        # Get the current and next pixels
        current = sequence[i]
        next = sequence[i + 1]

        # Calculate the pixel difference between them
        diff = pixel_diff(image, current, next)

        # Add the difference to the total difference
        total_diff += diff

    # Return the total difference
    return total_diff


# Define a function that takes a grayscale image and a seed pixel as input
def search_path_dp(image, seed, N=10):
    # Get the height and width of the image
    height, width = image.shape

    # Initialize the path as a list of the seed pixel
    path = [seed]

    # Initialize the current pixel as the seed pixel
    current = seed

    # Initialize a table to store the optimal path and pixel difference for each pixel
    # The table is a 3D array of shape (height, width, 2)
    # The first dimension is the row index, the second dimension is the column index
    # The third dimension is a tuple of (optimal path, pixel difference)
    table = np.empty((height, width, 2), dtype=object)

    # Loop until the path reaches the right edge of the image
    while current[1] < width - 1:
        # Initialize a list of candidate pixels for the next N steps
        candidates = []

        # Loop through the next N steps
        for i in range(N):
            # Initialize a list of possible pixels for the current step
            possible = []

            # Loop through the three possible directions: right, right-down, right-up
            for direction in [(0, 1), (1, 1), (-1, 1)]:
                # Calculate the next pixel coordinates by adding the direction to the current pixel
                next_x = current[0] + direction[0]
                next_y = current[1] + direction[1]

                # Check if the next pixel is within the image bounds
                if 0 <= next_x < height and 0 <= next_y < width:
                    # Add the next pixel to the possible list
                    possible.append((next_x, next_y))

            # Check if there are any possible pixels for the current step
            if possible:
                # Add the possible list to the candidate list
                candidates.append(possible)

                # Update the current pixel to be the first possible pixel
                current = possible[0]
            else:
                # Break the loop
                break

        # Check if there are any candidate pixels for the next N steps
        if candidates:
            # Find the optimal sequence of pixels by minimizing the total difference
            diff_list = [evaluate(image, sequence) for sequence in candidates]
            idx = np.argmin(diff_list)
            opt_sequence = candidates[idx]

            # Convert the optimal sequence to a list of tuples
            opt_sequence = list(map(tuple, opt_sequence))

            # Extend the path with the optimal sequence
            path.extend(opt_sequence)

            # Update the current pixel to be the last pixel in the optimal sequence
            current = opt_sequence[-1]
        else:
            # Break the loop
            break

    # Return the path
    return path
