# Odometry Module

This directory contains the central algorithms and pipelines for radar-based continuous odometry estimation. The system relies on sequence matching between radar scans to estimate sensor motion, implementing key components from CFEAR and Cen2019 reference pipelines.

## File Breakdown

### 1. `keypoint_extraction.py`
**Purpose**: Extracts keypoints from raw radar images. 
- Implements filters such as CFAR (Constant False Alarm Rate).
- Supports CFEAR (filtering out low-range bins, Z-percentile thresholding, extracting k-strongest points per azimuth).
- Supports Cen2019 point extraction (using Saliency/Zero-mean maps).

### 2. `descriptors.py`
**Purpose**: Computes geometric or appearance-based descriptors for the extracted keypoints.
- Calculates surface normals and converts points into oriented point formats used by the CFEAR pipeline.
- Handles descriptor representation transformations needed for sequence correlation and matching for Cen2019 pipeline.

### 3. `data_association.py`
**Purpose**: Handles the matching and pairing of keypoints between consecutive radar frames.
- Implements unary matching and pairwise compatibility scoring for Cen2019 pipeline.
- Solves data association problems and directly resolves point-to-point and point-to-line registrations (specifically normal equations used in CFEAR).

### 4. `motion_estimation.py`
**Purpose**: Computes the rigid body transformation (translation and rotation) from associated point pairs.
- Features SVD (Singular Value Decomposition) and RANSAC-based algebraic solvers.
- Used predominantly for non-CFEAR pipelines (like Cen2019) that require direct outlier rejection from generic point clouds.

### 5. `ablation_study.py`
**Purpose**: The central processing library and main odometry integration loop.
- Contains `main_odometry()`, which acts as the core loop ingesting subsets of radar images, running extraction -> association -> estimation.
- Configurable through parameters to run different combinations of pipelines (CFEAR vs Cen2019) to compare metrics and calculate offsets.
- Handles GPS integration and absolute coordinate estimations over the calculated trajectory.

### 6. `evaluation.py`
**Purpose**: Analytics and visualizations.
- Extracts and interpolates Ground Truth motion from GPS constraints.
- Handles Geodesic math to apply rigid-body (lever-arm) corrections from the GPS antenna sensor to the radar origin.
- Visualizes RANSAC models, overlay comparisons, and renders interactive maps with Folium (`gps_map.html`).

### 7. Execution Entry Points
**Purpose**: Scripts and notebooks for running the pipeline.
- `main_odometry.py`: Command Line Interface  version of the pipeline, meant to batch process data, generate `odometry_results.csv`, and run via cluster jobs (`jobscript.sh`).
- `main.py`: Old execution script for Cen2019 pipeline
- `main.ipynb`: Interactive Jupyter Notebook wrapping `ablation_study.py` imports for data exploration and iterative testing.