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
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")

            dst_pts = np.array([[0, height-1],
                                [0, 0], 
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))
            try:
                rotated = osd_rotate(warped)
            except pytesseract.pytesseract.TesseractError:
                print("too few characters")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return rotated