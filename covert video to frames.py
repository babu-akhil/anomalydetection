# Code to extract frames from video

import cv2
import os


pathvar = r'F:/LSTM-AE CODE/input_videos/'
os.chdir(pathvar)

# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    if (vidObj.isOpened()== False):
	    print("Error opening video, check path name")
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
  
        # Saves the frames with frame-count 
        cv2.imwrite(r'F:/LSTM-AE/frames/frame%d.jpg' % count, image) 
  
        count += 1
  
# Driver Code 
if __name__ == '__main__': 
  
	# Calling the function 
    FrameCapture('./out.mp4') 
