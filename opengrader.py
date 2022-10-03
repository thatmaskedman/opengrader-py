import cv2 as cv
import numpy as np
from document import DocumentProcessor
def main():
    img = cv.imread('sheet/answered.jpg')
    processor = DocumentProcessor(img)
    processor.process()
    processor._write_steps()
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imwrite('gray.jpg', img_gray)
    # blur_img = cv.GaussianBlur(img_gray, (5,5),0)
    # cv.imwrite('blur.jpg', blur_img)

    # edged = cv.Canny(blur_img, 75, 200)

    # cv.imwrite('edged.jpg', edged)

    # contours, _ = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # cv.drawContours(img, contours, -1, (255,0,0), 2)

    # for index, contour in enumerate(contours):
    #     M = cv.moments(contour)
    #     if M['m00'] == 0:
    #         continue
    #     cx = int(M['m10'] / M['m00']) 
    #     cy = int(M['m01'] / M['m00'])
    #     points = (cx, cy)
    #     cv.putText(img, str(index), points, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

if __name__ == '__main__':
    main()