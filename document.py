import cv2 as cv
import numpy as np
import numpy.typing as npt
from typing import Any

class DocumentProcessor:
    _HEIGHT = 1056
    _WIDTH = 816

    _BGR_RED: tuple[int, int, int] = (0, 0, 255)
    
    def __init__(self, img: npt.NDArray[Any]) -> None:
        self.img = img
        self.img_original = img
        self.img_grayscaled: npt.NDArray[Any] = np.copy(img)
        self.img_blured: npt.NDArray[Any] = np.copy(img)
        self.img_warped = np.array([])
        self.img_scaled = np.array([])
        self.img_marked_points: npt.NDArray[Any] = np.copy(img)
        self.img_marked_borders: npt.NDArray[Any] = np.copy(img)
        
        self.img_dilated: npt.NDArray[Any] = np.array([])
        self.img_detected: npt.NDArray[Any] = np.array([])
        self.img_thresh: npt.NDArray[Any] = np.array([])
        self.img_warped: npt.NDArray[Any] = np.array([])
        self.img_scaled: npt.NDArray[Any] = np.array([])
        self.img_edged: npt.NDArray[Any] = np.copy(self.img) 
        self.contours: npt.NDArray[Any] = np.array([])

        self.img

    def process(self):
        self.img_grayscaled = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.img_blured = cv.GaussianBlur(self.img, (5, 5), 0)
        self.img_edged = cv.Canny(self.img_blured, 0, 100)
        self.img_dilated = cv.dilate(self.img_edged, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        self.contours, _ = cv.findContours(self.img_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(self.contours, key=lambda c: cv.contourArea(c), reverse=True)
        doc_contour = sorted_contours[0]

        rect = cv.minAreaRect(doc_contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        print(box)
        ordered_points: npt.NDArray[Any] = np.zeros((4,2), dtype='float32')
        s = box.sum(axis=1)
        diff = np.diff(box, axis=1)

        ordered_points[0] = box[np.argmin(s)]
        ordered_points[2] = box[np.argmax(s)]
 
        ordered_points[1] = box[np.argmin(diff)]
        ordered_points[3] = box[np.argmax(diff)]

        (tl, tr, br, bl) = ordered_points
        
        for i, point in enumerate(ordered_points):
            px, py =  tuple(map(int, point))
            cv.putText(self.img_marked_points, f'P{i}', (px, py), cv.FONT_HERSHEY_PLAIN, 5, self._BGR_RED, 2)
            cv.circle(self.img_marked_points, (px, py), 2, self._BGR_RED, 10)   


        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
 
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")

        M = cv.getPerspectiveTransform(ordered_points, dst)
        self.img_warped = cv.warpPerspective(self.img, M, (maxWidth, maxHeight))
        self.img_scaled = cv.resize(self.img_warped, (self._WIDTH, self._HEIGHT))
        
        cv.drawContours(self.img_marked_borders,[box],0,(0,0,255),2)

    def _write_steps(self):
        steps: dict['str', Any] =  {
            'original': self.img,
            'warped': self.img_warped,
            'grayscaled': self.img_grayscaled,
            'edged': self.img_edged, 
            'dilated': self.img_dilated,
            'marked_points': self.img_marked_points,
            'scaled': self.img_scaled,

            # # 'threshed': self.img_thresh,
            # 'blured': self.img_blured, 
            'marked_borders': self.img_marked_borders,
        }

        for i, item, in enumerate(steps.items()):
            k,v = item
            cv.imwrite(f'{i}_{k}.jpg', v)