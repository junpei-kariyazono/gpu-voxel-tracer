# gpu-voxel-tracer
[![DOI](https://zenodo.org/badge/1125218273.svg)](https://doi.org/10.5281/zenodo.18112351)  

A high-performance, **GPU-accelerated** implementation of the **Amanatides & Woo 3D Voxel Traversal Algorithm**. 

Written in pure **Python** using **Numba (CUDA)**, this library allows for extremely fast ray-voxel intersection calculations without the need for C++ compilation. It is designed for LiDAR processing, ray tracing simulations, and volumetric analysis.

## Features

* **🚀 GPU Accelerated:** Leveraging CUDA via Numba for massive parallel processing.
* **🐍 Pure Python:** No complex C++ build chains required. Just install dependencies and run.
* **📊 Comprehensive Outputs:**
    * **Beam Counts:** Number of rays passing through each voxel.
    * **Point Counts:** Number of rays terminating (hitting a point) within each voxel.
    * **Path Lengths:** Total distance traveled by rays inside each voxel.

## Requirements

* **Hardware:** NVIDIA GPU (Compute Capability 2.0 or above)
* **Software:**
    * Python 3
    * `cudatoolkit`
    * `numpy`
    * `numba`

### Tested Environment  
This library has been verified to work with the following configuration:

| Component        | Version    |
|:-----------------|:-----------|
| **OS**           | Windows 11 |
| **Python**       | 3.12       |
| **CUDA Toolkit** | 12.9.86    |
| **Numba**        | 0.63.1     |
| **Numpy**        | 2.3.5      |

## Installation

### 1. Install Dependencies
We recommend using `conda` to handle CUDA libraries automatically:

```bash
conda install numpy numba cudatoolkit
```
### For non-Conda users:

Please install the CUDA Toolkit manually from the NVIDIA website.

Important: Ensure that you install a CUDA version below 13 (e.g., CUDA 12.x).
The current version of Numba (0.63.1) is not compatible with CUDA 13 or newer.

CUDA Toolkit Archive: https://developer.nvidia.com/cuda-toolkit-archive

### 2. Get the Code
Clone this repository or simply download the `gpu_voxel_tracer.py` file and place it in your project directory.

```bash
git clone https://github.com/your-username/gpu-voxel-tracer.git
```

## Usage
For detailed usage and visualization, please refer to the [Jupyter Notebook](/tutorial.ipynb).
### 1. Initialize tracer

First, import the library and initialize the Tracer with your 3D space boundaries and voxel resolution.

```python
from gpu_voxel_tracer import Tracer

# bounds: Execution area (xmin, xmax, ymin, ymax, zmin, zmax)
# voxel_size: Resolution of cubic voxels
tracer = Tracer(bounds=(-100, 100, -100, 100, 0, 50), voxel_size=0.5)
```
### 2. Execute ray-tracing

```python
import numpy as np

# Create dummy data for demonstration
# starts: Start coordinates of rays (N, 3)
# ends:   End coordinates of rays (N, 3)
starts = np.random.uniform(-50, 50, (10000, 3))
ends   = np.random.uniform(-50, 50, (10000, 3))

results = tracer.run(starts, ends)
```
### 3. Access result
The results are returned as a dictionary containing 3D grids (numpy arrays).
You can access the specific data grids using dictionary keys.
```python
beams = results["beam_counts"]
points = results["point_counts"]
lengths = results["path_lengths"]
```

## Outputs Explained

The `run()` method returns a dictionary containing three 3D numpy arrays with the shape `(N_x, N_y, N_z)`.
* N_x, N_y, N_z: Number of voxels along the X, Y, and Z axes respectively.
* Indexing: For example, `beam_counts[i, j, k]` corresponds to the voxel at grid index `(i, j, k)`.

| Key                | data type     | Description                                                     |                                                                                                            
|:-------------------|:--------------|:----------------------------------------------------------------|
| **`beam_counts`**  | `np.int32`    | Number of rays passing through each voxel.                      |
| **`point_counts`** | `np.int32`    | Number of rays terminating (hitting a point) within each voxel. |
| **`path_lengths`** | `np.float64`  | Total distance traveled by rays inside each voxel.              |

## Algorithm

This implementation is based on the fast voxel traversal algorithm described in:

> **Amanatides, J., & Woo, A. (1987).** [A Fast Voxel Traversal Algorithm for Ray Tracing.](http://www.cse.yorku.ca/~amana/research/grid.pdf) *Eurographics '87*, 3–10.

It extends the original 2D/3D DDA (Digital Differential Analyzer) logic to run in parallel on thousands of GPU threads.

## License

MIT License

## Citation

If you use this software in your research, please cite it as:

```bibtex
@software{gpu_voxel_tracer,
  author = {Junpei Kariyazono},
  title = {GPU-Accelerated 3D Voxel Traversal (Amanatides & Woo) in Python},
  year = {2026},
  publisher = {Zenodo},
  url = {https://doi.org/10.5281/zenodo.18112351}
}
```
