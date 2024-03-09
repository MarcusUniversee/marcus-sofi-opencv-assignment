from pytesseract import Output
import pytesseract
import imutils
import cv2
import numpy as np
from orient_crop import orient_crop


check_img = orient_crop("images/IMG_1599.jpg")

#finds a word in img and returns a bounding box in x,y position and width and height
def find_word(img, word):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, output_type=Output.DICT)
    for i in range(len(data['text'])):
        if word in data['text'][i].lower():
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            return (x, y, w, h)
    return (-1, -1, -1, -1)

date_box = find_word(check_img, "date")

#find the horizontal line
gray = cv2.cvtColor(check_img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150,)

#these numbers can be tweaked
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=20)
bottom_line_date_box = date_box[1] + date_box[3]
min_line = None
min_dist = float("inf")

HORIZONTAL_PRECISION = 5

for line in lines:
    for x1, y1, x2, y2 in line:
        #checks if it is horizontal and within the bounding box
        if (abs(y2-y1) < HORIZONTAL_PRECISION) and 0 <= x1 - date_box[0] <= date_box[2] and 0 <= x2 - date_box[0] <= date_box[2]:
            #if it is closer than previous lines, replace the minimum line
            if (abs(y1 - bottom_line_date_box) < min_dist):
                min_dist = abs(y1 - bottom_line_date_box)
                min_line = line

if min_line is not None:
    cv2.rectangle(check_img, (date_box[0], date_box[1]), (date_box[0] + date_box[2], date_box[1] + date_box[3]), (0, 255, 0), 2)
    for x1, y1, x2, y2 in min_line:
        cv2.line(check_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
else:
    print("no lines found")

cv2.imshow('Rotated with date line', cv2.resize(check_img, (1280, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()

