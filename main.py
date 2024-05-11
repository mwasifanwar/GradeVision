import cv2
import numpy as np
import utilis as utl

img_width, img_height = 500, 500 # img dimensions

img = cv2.imread("images/test_img.jpg")
img = cv2.resize(img, (img_width, img_height))
imgContours = img.copy()
imgBiggestContours = img.copy()

# ----------------------Image Pre-Processing-------------------------------------

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  # Covert img to grayscale
imgBlur = cv2.GaussianBlur(imgGray,(3,3),1)  # Apply gaussian blur, parameter=>(grayscaled_img,kernel,sigma)

imgCanny = cv2.Canny(imgBlur,338,0)  # Canny edge detector to Detect edges, Parameters=>(Blurred Image, Thresh values)

# ----------------------Finding All Countours-------------------------------------
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0,255,0), 5)  # Parameters=>(imgContours, contours, <<how many | in my case all>>, <<color | in my case GREEN>>, <<Thickness>>)

# ----------------------Find Rectangles-------------------------------------
rectCont = utl.rectContours(contours)
firstCont = utl.getCornerPoints(rectCont[0])
secondCont = utl.getCornerPoints(rectCont[1])
thirdCont = utl.getCornerPoints(rectCont[2])
fourthCont = utl.getCornerPoints(rectCont[3])


if firstCont.size != 0 and secondCont.size != 0 and thirdCont.size != 0 and fourthCont.size != 0:
    cv2.drawContours(imgBiggestContours, firstCont,-1,(0,255,0),10)  # Label: BOX#2
    cv2.drawContours(imgBiggestContours, secondCont,-1,(0,255,0),10)  # Label: BOX#3
    cv2.drawContours(imgBiggestContours, thirdCont,-1,(0,255,0),10)  # Label: BOX#1
    cv2.drawContours(imgBiggestContours, fourthCont,-1,(0,255,0),10)  # Label: BOX#4
   
    firstCont = utl.reorder(firstCont)
    secondCont = utl.reorder(secondCont)
    thirdCont = utl.reorder(thirdCont)
    fourthCont = utl.reorder(fourthCont)

    # ------------------For Bird Eye View---------------------------
    first_pt1 = np.float32(firstCont)
    first_pt2 = np.float32([[0,0], [200, 0], [0, img_height], [200, img_height]])
    first_matrix = cv2.getPerspectiveTransform(first_pt1, first_pt2)
    first_imgWarpColored = cv2.warpPerspective(img, first_matrix, (200, img_height))

    second_pt1 = np.float32(secondCont)
    second_pt2 = np.float32([[0,0], [200, 0], [0, img_height], [200, img_height]])
    second_matrix = cv2.getPerspectiveTransform(second_pt1, second_pt2)
    second_imgWarpColored = cv2.warpPerspective(img, second_matrix, (200, img_height))

    third_pt1 = np.float32(thirdCont)
    third_pt2 = np.float32([[0,0], [200, 0], [0, img_height], [200, img_height]])
    third_matrix = cv2.getPerspectiveTransform(third_pt1, third_pt2)
    third_imgWarpColored = cv2.warpPerspective(img, third_matrix, (200, img_height))

    fourth_pt1 = np.float32(fourthCont)
    fourth_pt2 = np.float32([[0,0], [200, 0], [0, img_height], [200, img_height]])
    fourth_matrix = cv2.getPerspectiveTransform(fourth_pt1, fourth_pt2)
    fourth_imgWarpColored = cv2.warpPerspective(img, fourth_matrix, (200, img_height))
    
    # ------------------Now we are going to find which boxes(options) are marked and which are not based on the pixel values in the image because markde options will have dense pixel values---------------------------
    first_imgWarpGray = cv2.cvtColor(first_imgWarpColored, cv2.COLOR_BGR2GRAY)
    first_imgThresh = cv2.threshold(first_imgWarpGray,140,255,cv2.THRESH_BINARY_INV)[1]
    # Define the coordinates of the region to crop
    first_x1, first_y1 = 0, 30  # Top-left corner
    first_x2, first_y2 = first_imgThresh.shape[1],first_imgThresh.shape[0]  # Bottom-right corner
    # Crop the region
    first_imgThresh_final = first_imgThresh[first_y1:first_y2, first_x1:first_x2]
    
    second_imgWarpGray = cv2.cvtColor(second_imgWarpColored, cv2.COLOR_BGR2GRAY)
    second_imgThresh = cv2.threshold(second_imgWarpGray,140,255,cv2.THRESH_BINARY_INV)[1]
    # Define the coordinates of the region to crop
    second_x1, second_y1 = 0, 30  # Top-left corner
    second_x2, second_y2 = second_imgThresh.shape[1],second_imgThresh.shape[0]  # Bottom-right corner
    # Crop the region
    second_imgThresh_final = second_imgThresh[second_y1:second_y2, second_x1:second_x2]
    
    third_imgWarpGray = cv2.cvtColor(third_imgWarpColored, cv2.COLOR_BGR2GRAY)
    third_imgThresh = cv2.threshold(third_imgWarpGray,140,255,cv2.THRESH_BINARY_INV)[1]
    # Define the coordinates of the region to crop
    third_x1, third_y1 = 0, 30  # Top-left corner
    third_x2, third_y2 = third_imgThresh.shape[1],third_imgThresh.shape[0]  # Bottom-right corner
    # Crop the region
    third_imgThresh_final = third_imgThresh[third_y1:third_y2, third_x1:third_x2]
    
    fourth_imgWarpGray = cv2.cvtColor(fourth_imgWarpColored, cv2.COLOR_BGR2GRAY)
    fourth_imgThresh = cv2.threshold(fourth_imgWarpGray,140,255,cv2.THRESH_BINARY_INV)[1]
    # Define the coordinates of the region to crop
    fourth_x1, fourth_y1 = 0, 30  # Top-left corner
    fourth_x2, fourth_y2 = fourth_imgThresh.shape[1],fourth_imgThresh.shape[0]  # Bottom-right corner
    # Crop the region
    fourth_imgThresh_final = fourth_imgThresh[fourth_y1:fourth_y2, fourth_x1:fourth_x2]
    
imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny, imgContours, imgBiggestContours, first_imgWarpColored, first_imgThresh_final, second_imgWarpColored, second_imgThresh_final, third_imgWarpColored, third_imgThresh_final, fourth_imgWarpColored, fourth_imgThresh_final])

# utl.display_images_scrollable(imageArray)

utl.splitOptions(fourth_imgThresh_final)
