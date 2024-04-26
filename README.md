# Tail tracking

Larval zebrafish tail tracking in head fixed experiment.
The project aims to provide a simple and efficient way to track the tail movement of larval zebrafish,
this code is tuned to perform in low-resolution and low signal-to-noise ratio settings.

## The algorithm

The algorithm works by assuming there is a pixel which is (relatively) invariant through the entire video,
from this pixel we try construct a line segment by searching to one direction of the video.
We seek to find the line segment which minimizes the sum of the L1 norm of the differences of consecutive terms in the sequence defined by the line segment.

$$p_{\text{opt}} = \text{arg} \min_p \sum_{i=1}^{N-1} ||p_{i+1} - p_i||_1$$

where $p = \lbrace p_1, p_2, \ldots, p_N \rbrace$ is a sequence of the greyscale values of points.
