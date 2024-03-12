from pytesseract import Output
import pytesseract
import imutils
import cv2
import numpy as np


#uses text to orient the image the right way
def osd_rotate(img):
    results = pytesseract.image_to_osd(img, output_type=Output.DICT)
    # display the orientation information
    print("[INFO] detected orientation: {}".format(
        results["orientation"]))
    print("[INFO] rotate by {} degrees to correct".format(
        results["rotate"]))
    print("[INFO] detected script: {}".format(results["script"]))
    # rotate the image to correct the orientation
    return imutils.rotate_bound(img, angle=results["rotate"])


#This function takes an input image
#looks for a check
#then rotates the check so it is upright
#and crops the background
def orient_crop(image_path):

    #read the image
    image = cv2.imread(image_path)

    #convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #detect edges with canny
    edges = cv2.Canny(gray, 50, 150)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rotated = None



    #this process detects if a contour is rectangle (that is not too small), and then crops and rotates it
    for contour in contours:

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if (len(approx)) == 4 and cv2.contourArea(contour) > 1000:
            points = approx.reshape(4, 2)
            #print(points)
            warped = four_point_transform(image, points)
#            rect = cv2.minAreaRect(approx)
#            box = cv2.boxPoints(rect)
#            box = np.intp(box)
#            width = int(rect[1][0])
#            height = int(rect[1][1])
#
#            src_pts = box.astype("float32")
#
#            dst_pts = np.array([[0, height-1],
#                                [0, 0], 
#                                [width-1, 0],
#                                [width-1, height-1]], dtype="float32")
#            
#            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
#            warped = cv2.warpPerspective(image, M, (width, height))
            try:
                #cv2.imshow("warped", cv2.resize(warped, (600, 1280)))
                rotated = osd_rotate(warped)
                #cv2.imshow("rotated", cv2.resize(rotated, (1280, 600)))
            except pytesseract.pytesseract.TesseractError:
                print("too few characters")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

    return rotated

def four_point_transform(image, pts):
    # Obtain a consistent order of the points and unpack them individually
    rect = pts.astype("float32")
    (tl, tr, br, bl) = pts

    # Compute the width of the new image, which will be the maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Compute the height of the new image, which will be the maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Now that we have the dimensions of the new image, construct the set of destination points to obtain a
    # "birds eye view", (i.e. top-down view) of the image, again specifying points in the top-left, top-right,
    # bottom-right, and bottom-left order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image
    return warped


check_img = orient_crop("images/IMG_1599.jpg")
cv2.imwrite("aligned_IMG_1599.jpg", check_img)
cv2.imshow("check_img", cv2.resize(check_img, (1280, 800)))
cv2.waitKey(0)