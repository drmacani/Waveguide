import cv2
import numpy as np
import glob
import re

img_array = []

# Specify the path where individual images are located
path = r"G:/Shared drives/TrEnT/2. Max Planck - Gecko-Inspired Textures for Surgical Grasping/UMT trial testing/Image Processing - Adhesion/2pt5N/Contact Area Analysis/*.png"

# Function to improve the sorting order of the frames read from the directory based on numerical order
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Load all the images from the specified path and sort them based on their numerical order
for filename in sorted(glob.glob(path), key=numericalSort):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

# Specify the path where you want the video file to be saved
save_path = r"G:/Shared drives/TrEnT/2. Max Planck - Gecko-Inspired Textures for Surgical Grasping/UMT trial testing/Image Processing - Adhesion/2pt5N/2pt5N.mp4"  # CHANGE this to your desired directory

# Create a VideoWriter object to save the video. Currently set at 20 fps (can be adjusted).
# Using the DIVX codec to save the video in .mp4 format
out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

# Add each image in the img_array to the video
for i in range(len(img_array)):
    out.write(img_array[i])

# Release the VideoWriter object once video creation is done
out.release()


