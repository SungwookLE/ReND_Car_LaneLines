import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from helper import *
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def process_image(image, mode='off'):
    x_len = image.shape[1]
    y_len = image.shape[0]

    img = image.copy()
    
    # step 1. cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = grayscale(img)

    # step option. equalizeHist
    equ = cv2.equalizeHist(gray)

    # step 2. gaussian blue (if needed,)
    blur = gaussian_blur(equ, 7)

    # step 3. Canny(img, low_threshold, high_threshold)
    canny_img=canny(blur, 50, 100)

    # step 4. region_of_interest
    x_off = 150
    trapezoidal_x_off = 220
    y_off = 320
    region = np.array([[x_off+trapezoidal_x_off, y_off], [x_len-x_off-trapezoidal_x_off,y_off], [x_len-x_off, y_len] , [x_off,y_len]], np.int32)
    roi = region_of_interest(canny_img, region)

    # step 5. HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    hough_img=hough_lines(roi, 1, np.pi/180, 45, 25, 3)

    # step 6. visualizaing with addWeighted
    res=weighted_img(hough_img, image)

    if mode == 'off':
        return res, roi
    elif mode == 'real':
        return res
    else:
        return res,roi

""" OFFLINE TEST """
images = os.listdir("test_images/")
fig , ax = plt.subplots(2,len(images)//2)
fig.suptitle('Lane Detect', fontsize=10)

fig2 , ax2 = plt.subplots(2,len(images)//2)
fig2.suptitle('ROI', fontsize=10)

idx =0
for img in images:
    #reading in an image
    pick_img = mpimg.imread("test_images/"+img)
    print('This image is:', type(pick_img), 'with dimensions:', pick_img.shape)
    test,roi=process_image(pick_img,'off')

    ax[idx%2,idx//2].imshow(test)
    ax2[idx%2,idx//2].imshow(roi, cmap='gray')
    idx+=1

fig.tight_layout()
fig2.tight_layout()

fig.savefig("./test_images_output/LaneDetect.png")
fig2.savefig("./test_images_output/ROI.png")
plt.show()

""" VIDEO SAVE
white_output = 'test_videos_output/solidWhiteRight.mp4'
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)

challenge_output = 'test_videos_output/challenge.mp4'
##clip3 = VideoFileClip('test_videos/challenge.mp4').subclip(0,5)
clip3 = VideoFileClip('test_videos/challenge.mp4')
challenge_clip = clip3.fl_image(process_image)
challenge_clip.write_videofile(challenge_output, audio=False)
"""