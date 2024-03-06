from pytesseract import Output
import pytesseract
import imutils
import cv2
# load the input image, convert it from BGR to RGB channel ordering,
# and use Tesseract to determine the text orientation
image = cv2.imread("images/IMG_1599.jpg")
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pytesseract.image_to_osd(rgb, output_type=Output.DICT)
# display the orientation information
print("[INFO] detected orientation: {}".format(
	results["orientation"]))
print("[INFO] rotate by {} degrees to correct".format(
	results["rotate"]))
print("[INFO] detected script: {}".format(results["script"]))
# rotate the image to correct the orientation
rotated = imutils.rotate_bound(image, angle=results["rotate"])
# show the original image and output image after orientation
# correction
#cv2.imshow("Original", cv2.resize(image, (1440, 900)))
#cv2.imshow("Output", cv2.resize(rotated, (1440, 900)))
#cv2.waitKey(0)

#convert to gray scale
gray = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)

#apply blur to increase accuracy by ignoring small details
blurred = cv2.GaussianBlur(gray, (15, 15), 0)
ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
count = 0
largest_contour = None
max_area = 0
RATIO = 6/2.75
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    #print(f"area: {cv2.contourArea(contour)}\nwidth: {w}\n height: {h}")
    #cv2.imshow(f"{cv2.contourArea(contour)}", cv2.resize(rotated[y:y+h, x:x+w], (1000, 700)))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if (cv2.contourArea(contour) > max_area and w > h*RATIO):
        max_area = cv2.contourArea(contour)
        largest_contour = contour
        
x, y, w, h = cv2.boundingRect(largest_contour)
cropped_image = rotated[y:y+h, x:x+w]
cv2.imshow('Original Image', cv2.resize(image, (1280, 800)))
cv2.imshow('Rotated', cv2.resize(rotated, (1280, 800)))
cv2.imshow("Cropped", cv2.resize(cropped_image, (1280, 800)))
cv2.waitKey(0)
cv2.destroyAllWindows()