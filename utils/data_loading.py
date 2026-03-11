import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import os
import matplotlib.pyplot as plt

# Folder and file paths to the radar images and GPS data
FOLDER_PATH = r"C:\Users\leona\Documents\Masterstudium\Special Course - Object Detection\Data\decoded\_radar_data_b_scan_image"
CSV_FILE_PATH = r"C:\Users\leona\Documents\Masterstudium\Special Course - Object Detection\Data\decoded\snowborbus_gps_data.csv"

# Transformation functions for radar data
def polar_to_cartesian_points(ranges, angles, range_resolution = 0.155, angle_resolution = 2*np.pi/400):
    """
    Convert polar coordinates (ranges and angles) to Cartesian coordinates (x, y) and scaling them according to the resolution.
    """
    x = ranges * range_resolution* np.cos(angles*angle_resolution)
    y = ranges * range_resolution * np.sin(angles*angle_resolution)
    points = np.stack((x, y), axis=-1)

    return points

def polar_to_cartesian_image(image, size=1024, max_range=1.0):
    """
    Convert a radar image in polar coordinates to Cartesian coordinates.
    """
    num_angles, num_ranges = image.shape[:2]
    
    # Define cartesian output grid
    x_cart = np.linspace(-max_range, max_range, size)
    y_cart = np.linspace(-max_range, max_range, size)
    x_cart_grid, y_cart_grid = np.meshgrid(x_cart, y_cart)
    
    # Convert to polar coordinates
    r_cart = np.sqrt(x_cart_grid**2 + y_cart_grid**2)
    theta_cart = np.arctan2(y_cart_grid, x_cart_grid)
    theta_cart_normalized = np.where(theta_cart < 0, theta_cart + 2*np.pi, theta_cart)
    
    # Map to image indices
    row_indices = (theta_cart_normalized / (2 * np.pi)) * (num_angles - 1)
    col_indices = r_cart  * (num_ranges - 1)

    # Interpolate
    f = RegularGridInterpolator((np.arange(num_angles), np.arange(num_ranges)), image,
                             bounds_error=False, fill_value=0)
    
    points = np.array([row_indices.ravel(), col_indices.ravel()]).T
    img_cartesian = f(points).reshape(size, size)
    
    return img_cartesian

# Function to extract timestamp from filename
def extract_timestamp(filename):
    """Extract timestamp from filename in the format: radar_YYYY-MM-DD_HH-MM-SS_ms.png"""
    parts = filename.split('_')
    date = parts[1]  # YYYY-MM-DD
    time = parts[2]  # HH-MM-SS
    time = time.replace('-', ':')  # Convert to HH:MM:SS
    ms = parts[3].replace('.png', '')  # ms
    return f"{date}T{time}.{ms}"

# Load functions for radar images and GPS data
def load_radar_images(num_images = 5,folder_path = FOLDER_PATH):
    # List all PNG files in the folder and sort them by timestamp
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')], key=extract_timestamp)

    # Load the first num_images radar images
    images = []
    for i in range(min(num_images, len(image_files))):
        img = plt.imread(os.path.join(folder_path, image_files[i]))
        images.append(img)

    return image_files, images

def load_gps_data(csv_file_path = CSV_FILE_PATH):
    gps_data = pd.read_csv(csv_file_path)
    gps_data['time-string'] = pd.to_datetime(gps_data['time-string'])
    return gps_data

# Add function which corrects the blank space in radar images 


if __name__ == "__main__":
    # Load radar image 
    image_file, image = load_radar_images(num_images=1)
    img = image[0]

    time_stamp = extract_timestamp(image_file[0])
    print(f"Loaded radar image with timestamp: {time_stamp}")
    
    cartesian_image = polar_to_cartesian_image(img)
    
    plt.figure(figsize=(12, 12))

    plt.subplot(1, 2, 1)
    plt.title("Radar Image (Polar)")
    plt.imshow(img, cmap='gray', aspect='auto')
    
    plt.subplot(1, 2, 2)
    plt.title("Radar Image (Cartesian)")
    plt.imshow(cartesian_image, cmap='gray')
    
    plt.show()