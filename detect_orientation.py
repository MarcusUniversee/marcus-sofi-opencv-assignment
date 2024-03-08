from pytesseract import Output
import pytesseract
import imutils
import cv2
import numpy as np
# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to determine the text orientation
image = cv2.imread("images/IMG_1599.jpg")

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

#convert to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#apply blur to increase accuracy by ignoring small details
BLUR_DEGREE = (15, 15)
blurred = cv2.GaussianBlur(gray, BLUR_DEGREE, 0)
edges = cv2.Canny(gray, 50, 150)
contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
rotated = None
for contour in contours:
    #this process detects if a contour is rectangle (that is not too small), and then crops and rotates it
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


#I had to resize because it would not fit the entire image onto my computer screen, so the ratios are probably off
cv2.imshow('Original Image', cv2.resize(image, (1280, 800)))
try:
    cv2.imshow('Rotated', cv2.resize(rotated, (1280, 600)))
except:
    print("Unable to find valid image")
cv2.waitKey(0)
cv2.destroyAllWindows()