# -*- coding: utf-8 -*-
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from matplotlib import image as mp_image
import skimage as sk
from skimage import io as sk_io
from skimage import data, img_as_float, exposure
import numpy as np
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
from skimage import morphology
import glob
import time
import re
import os
import cv2
from skimage.filters.rank import enhance_contrast


#sets path and gets list of files
path = r"E:/Mac/20sdwell_0pt1mmpsretraction/5N/Undistorted Split Frames"
save_folder = "E:/Mac/20sdwell_0pt1mmpsretraction/5N/Contact Area Analysis"

def normalize_grayimage(image):
	image = cv2.equalizeHist(image)
	#cv2.imshow("Equalized img", image)
	return image

#for improving the sorting order for the frames read from the directory
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

filenames = sorted(glob.glob(path + "/*.png"), key = numericalSort)
pixels=np.zeros(len(filenames))

#creates for loop to run through all of the files in the path above
sublist1=filenames[:] #[455:466]#[68:761]
print (sublist1)

count=0
for filename in sublist1:
    # t0=time.time()
   
    
    # grayscale image
    sk_image = sk_io.imread(filename) 
    sk_image = sk_image[200:500,400:800] # [0:820,40:1700] #[0:1500,0:2000]
    
    sk_image = np.array(sk_image, dtype = 'uint8')     
    
    gray_image = sk.color.rgb2gray(sk_image)
    
    # blur the image to denoise
    blurred_image = sk.filters.gaussian(gray_image, sigma=1.0)
        
    # Threshold whichever shows the clearest seperation from background. 
    #The contact is the bright area (above 0.6). The contact you want is probably the bulge of pixels around 0.9.
    #Looking at the histogram above, a theshold is taken of 0.6, giving the image below:
  
    # perform automatic thresholding
    #thresh = sk.filters.threshold_otsu(blurred_image)
    #print("Found automatic threshold thresh = {}.".format(thresh))
    # manual thresholding
    thresh = 0.15
    
    binary_mask = blurred_image>thresh     
    
    # Remove artifacts from the image. 
    #To do this, dilate and then eliminate all but the largest body. 
    binary_mask = binary_mask.astype(np.uint8)
    label_img = label(binary_mask)
    props = regionprops_table(label_img, properties=('area','centroid','major_axis_length','minor_axis_length'))
    kernel = disk(10) # Dilate to make a large 
    dilated = dilation(binary_mask,kernel)
    dilated_label_img = label(dilated)
    d_props = regionprops_table(dilated_label_img, properties=('area','centroid','major_axis_length','minor_axis_length'))
    d_DF = pd.DataFrame(d_props)
       
    # Make a mask without errors, then find the similarity between this and non-dilated image
    pix_thresh = 2000
    Mask = morphology.remove_small_objects(dilated_label_img, pix_thresh)
    BWfinal = binary_mask*Mask
    
    # Pixels for determining contact areaa
    Mask2=Mask>0
    BWfinal2 = BWfinal>0    
    
    # Use this mask to look at pixel (and therefore pressure) distribution across the contact area. This is plotted below with colourbar and histogram.
    gray_threshed = blurred_image*(Mask2)
    plt.figure() #figure 7
    plt.imshow(gray_threshed, cmap=plt.cm.gray)
    
    pixels[count]=np.sum(BWfinal2)
    
    # Pressure Distribution image
    plt.ioff() #figure 1
    plt.imshow(gray_threshed,'jet')
    plt.clim(0,1) #sets the colour range for pressure distribution
    plt.colorbar()
    image_save_path = os.path.join(save_folder, f"image{count}.png")
    plt.savefig(image_save_path)
    plt.close()
    
    if count % 100 ==0:
         print('frame number:',count)
    count += 1
    # t1=time.time()
    # print(t1-t0)

#10. create figure of pixesl 
x=np.arange (0, len(filenames), 1)
t=x/25 #30  #Check this video recording frame rate used
plt.figure() #figure 2
plt.plot(t,pixels)
plt.xlabel("Time (s)")
plt.ylabel("Pixels")
#plt.show()
pixels_vs_time_path = os.path.join(save_folder, "pixels_vs_time.png")
plt.savefig(pixels_vs_time_path)

#save pixels to csv file
#pixel_time = {'Time':[t], 'Pixels':[pixels]}
pixel_res = pd.DataFrame(pixels)
pixels_result_path = os.path.join(save_folder, "pixels_result.xlsx")
pixel_res.to_excel(pixels_result_path, index=False)


# contact area vs time graph 
area=pixels*0.001118413 #0.001118935 Contact area calculation, Use calibration card as a preliminary test to determine the conversion factor for contact area approximation
plt.figure() #figure 3
plt.plot(t,area)
plt.xlabel("Time (s)")
plt.ylabel("Contact Area (mm²)")
#plt.show()
area_vs_time_path = os.path.join(save_folder, "area_vs_time.png")
plt.savefig(area_vs_time_path)

print('complete)')