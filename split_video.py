import cv2

#load up the video path
capture = cv2.VideoCapture('E:/Mac/20sdwell_0pt1mmpsretraction/5N/Gecko_adhesion_210C_waveguide_2023-03-23_5pt0N_20sdwell_0pt1mmpsretraction.h264')

frameNr = 0
 
while (True):
 
    success, frame = capture.read()
 
    if success:
        #write the split images to this path
        cv2.imwrite(f'E:/Mac/20sdwell_0pt1mmpsretraction/5N/Distorted Split Frames/frame_0000{frameNr}.png', frame)
 
    else:
        break
 
    frameNr = frameNr+1
 
capture.release()
