import numpy as np
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import os
import matplotlib.pyplot as plt

FOLDER_PATH = r"C:\Users\leona\Documents\Masterstudium\Special Course - Object Detection\Data\decoded\_radar_data_b_scan_image"
CSV_FILE_PATH = r"C:\Users\leona\Documents\Masterstudium\Special Course - Object Detection\Data\decoded\snowborbus_gps_data.csv"

def polar_to_cartesian_points(ranges, angles):
    """
    Convert polar coordinates (ranges and angles) to Cartesian coordinates (x, y).
    
    Parameters:
    - ranges: Array of range measurements (distance from radar)
    - angles: Array of angle measurements (bearing from radar)
    
    Returns:
    - points: Nx2 array of Cartesian coordinates (x, y)
    """
    range_resolution = 0.155 #m
    angle_resolution = 2*np.pi/400 # 400 azimuth bins

    x = ranges * range_resolution* np.cos(angles*angle_resolution)
    y = ranges * range_resolution * np.sin(angles*angle_resolution)
    points = np.stack((x, y), axis=-1)
    
    return points

def polar_to_cartesian_image(image, size=1024, max_range=1.0):
    """
    Convert a radar image in polar coordinates to Cartesian coordinates.
    
    Parameters:
    - image: 2D array with shape (num_angles, num_ranges)
    - size: Output Cartesian image size in pixels
    - max_range: Maximum range in normalized units (0 to 1)
    
    Returns:
    - cartesian_image: 2D array in Cartesian coordinates
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

def extract_timestamp(filename):
    """Extract timestamp from filename in the format: radar_YYYY-MM-DD_HH-MM-SS_ms.png"""
    parts = filename.split('_')
    date = parts[1]  # YYYY-MM-DD
    time = parts[2]  # HH-MM-SS
    time = time.replace('-', ':')  # Convert to HH:MM:SS
    ms = parts[3].replace('.png', '')  # ms
    return f"{date}T{time}.{ms}"

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