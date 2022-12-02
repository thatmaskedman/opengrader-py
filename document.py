import cv2 as cv
import numpy as np
import numpy.typing as npt
import itertools as it 
import os
from typing import Any

class DocumentProcessor:
    _HEIGHT = 1056
    _WIDTH = 816

    _SQUARE_MARKER = np.array([
        [0, 0], 
        [0, 50], 
        [50, 50], 
        [50, 0]
    ])

    _BGR_RED: tuple[int, int, int] = (0, 0, 255)

    #TODO Clean up the constructor
    def __init__(self, img_path: str) -> None:
        self.img_path = img_path 
        self.img = np.array([])
        self.img_original = img
        self.img_grayscaled: npt.NDArray[Any] = np.copy(img)
        self.img_blured: npt.NDArray[Any] = np.copy(img)
        self.img_warped = np.array([])
        self.img_scaled = np.array([])
        self.img_marked_points: npt.NDArray[Any] = np.copy(img)
        self.img_marked_borders: npt.NDArray[Any] = np.copy(img)
        
        self.img_scaled_adaptive_thresh = npt.NDArray[Any]

        self.img_scaled_edged = np.array([])
        self.img_scaled_regionA = np.array([])
        self.img_scaled_regionB = np.array([])
        self.img_scaled_regionC = np.array([])

        self.img_dilated: npt.NDArray[Any] = np.array([])
        self.img_detected: npt.NDArray[Any] = np.array([])
        self.img_thresh: npt.NDArray[Any] = np.array([])
        self.img_warped: npt.NDArray[Any] = np.array([])
        self.img_scaled: npt.NDArray[Any] = np.array([])
        self.img_edged: npt.NDArray[Any] = np.copy(self.img)
        self.contours: npt.NDArray[Any] = np.array([])

        self._choice_points = np.array([])
        self._choice_boxes = np.array([])
        self._choice_intensity = np.array([])


    def scale(self):
        pass
        
        
    def process(self):
        self.img = cv.imread(self.img_path)
        def cont_centre_point(c):
            (x, y), _ = cv.minEnclosingCircle(c)
            return (int(x), int(y))

        def points_to_box(p, width):
            px, py = p
            w = width // 2
            return np.array([
                [px-w, py+w],
                [px-w, py-w],
                [px+w, py-w],
                [px+w, py+w]
            ])

        def order_points(points: npt.ArrayLike):
            points = points[points[:, 1].argsort()]
        
            ret = []
            for point in points.reshape(-1,5,2):
                ret.append(np.sort(point, axis=0))
            
            return np.array(ret) 
                        
        self.img_grayscaled = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.img_grayscaled = cv.copyMakeBorder(self.img_grayscaled, 30, 30, 30, 30, cv.BORDER_CONSTANT, value=(0,0,0))
        self.img_blured = cv.GaussianBlur(self.img_grayscaled, (5, 5), 0)
        self.img_edged = cv.Canny(self.img_blured, 0, 100)
        self.img_dilated = cv.dilate(self.img_edged, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

        self.contours, _ = cv.findContours(self.img_dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        sorted_contours = sorted(self.contours, key=lambda c: cv.contourArea(c), reverse=True)

        doc_contour = sorted_contours[0]

        rect = cv.minAreaRect(doc_contour)
        box = cv.boxPoints(rect)
        box = np.int0(box)

        ordered_points: npt.NDArray[Any] = np.zeros((4,2), dtype='float32')
        s = box.sum(axis=1)
        diff = np.diff(box, axis=1)

        ordered_points[0] = box[np.argmin(s)]
        ordered_points[2] = box[np.argmax(s)]
 
        ordered_points[1] = box[np.argmin(diff)]
        ordered_points[3] = box[np.argmax(diff)]

        (tl, tr, br, bl) = ordered_points

        
        # for i, point in enumerate(ordered_points):
        #     px, py =  tuple(map(int, point))
        #     cv.putText(self.img_marked_points, f'P{i}', (px, py), cv.FONT_HERSHEY_PLAIN, 5, self._BGR_RED, 2)
        #     cv.circle(self.img_marked_points, (px, py), 2, self._BGR_RED, 10)   


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
            [0, maxHeight - 1]], dtype = "float32"
        )

        M = cv.getPerspectiveTransform(ordered_points, dst)
        self.img_warped = cv.warpPerspective(self.img, M, (maxWidth, maxHeight))
        self.img_scaled = cv.resize(self.img_warped, (self._WIDTH, self._HEIGHT))

        print(cv.contourArea(doc_contour) / self.img_grayscaled.size, "NIxG")
        if cv.contourArea(doc_contour) / self.img_grayscaled.size < 0.70:
            raise Exception
        


        # cv.drawContours(self.img_marked_borders,[box],0,(0,0,255),2)
        
        self.img_scaled_grayscaled = cv.cvtColor(self.img_scaled, cv.COLOR_BGR2GRAY)
        self.img_scaled_blured = cv.GaussianBlur(self.img_scaled_grayscaled, (5, 5), 0)
        self.img_scaled_adaptive_thresh = cv.adaptiveThreshold(self.img_scaled_blured, 255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY_INV,9,2)
        _, self.img_scaled_foobar = cv.threshold(self.img_scaled_blured, 127, 255, cv.THRESH_BINARY_INV)
        self.img_scaled_edged = cv.Canny(self.img_scaled_blured, 50, 200)

        # self.img_scaled_dilated = cv.dilate(self.img_scaled_edged, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
        conts, hierarchy = cv.findContours(self.img_scaled_adaptive_thresh, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []

        for i, z in enumerate(zip(conts, hierarchy[0])):
            c, h = z
            _,_,child, _, = h
            if child == -1:
                continue
            # (x,y), radius = cv.minEnclosingCircle(c)
            # center = (int(x),int(y))
            
            filtered_contours.append(c)
            # cv.drawContours(self.img_scaled, [c], -1, self._BGR_RED, 2)
            # cv.putText(self.img_scaled, str(i), center, cv.FONT_HERSHEY_SIMPLEX, 1, self._BGR_RED, 2)
        # cv.drawContours(self.img_scaled, [conts[24]], -1, self._BGR_RED, 1)
        # print("TEST", conts[24])

        sorted_c = sorted(filtered_contours, key=lambda x: cv.matchShapes(self._SQUARE_MARKER, x, cv.CONTOURS_MATCH_I3, 0))
        sorted_a = sorted(sorted_c[:16], key=cv.contourArea, reverse=True)

        # for c in map(cv.contourArea, sorted_c[:3]):
            # print(c)
        marker_borders = map(cv.minEnclosingCircle, sorted_a[:8])
        marker_center = []
        
        for i, c in enumerate(marker_borders):
            (x, y), _ = c
            p = int(x), int(y)
            marker_center.append(p)
            # cv.rectangle(self.img_scaled, 
            # cv.putText(self.img_scaled, str(i), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, self._BGR_RED, 2)
            
        sorted_y = sorted(marker_center, key=lambda c: c[1], reverse=True)

        bottom_markers = sorted_y[:4]
        top_markers = sorted_y[4:]

        print(len(top_markers))

        bottom_markers = sorted(bottom_markers, key=lambda c: c[0])
        top_markers = sorted(top_markers, key=lambda c: c[0])

        marker_center = bottom_markers + top_markers 
        
        for i, c in enumerate(marker_center):
            x, y = c
            # marker_center.append((x,y))
            # cv.rectangle(self.img_scaled, 
            cv.putText(self.img_scaled, str(i), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, self._BGR_RED, 2)
        
        regionA = np.array([
            marker_center[0], marker_center[1], 
            marker_center[5], marker_center[4]
        ])

        regionB = np.array([
            marker_center[1], marker_center[2], 
            marker_center[6], marker_center[5]
        ])

        regionC = np.array([
            marker_center[2], marker_center[3], 
            marker_center[7], marker_center[6]
        ])

        maskA = np.zeros(self.img_scaled.shape[:2], dtype=np.uint8)
        maskB = np.zeros(self.img_scaled.shape[:2], dtype=np.uint8)
        maskC = np.zeros(self.img_scaled.shape[:2], dtype=np.uint8)

        regions = [regionA, regionB, regionC]
        mask_regions = [maskA, maskB, maskC]

        for region, mask in zip(regions, mask_regions):
            (x,y,w,h) = cv.boundingRect(region)
            cv.rectangle(mask, (x+15,y+10), (x+w-20, y+h), (255,255,255), -1)
            cv.putText(self.img_scaled, str(i), (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, self._BGR_RED, 2)
        
        maskA, maskB, maskC = tuple(mask_regions)
        self.img_scaled_regionA = cv.bitwise_and(self.img_scaled_adaptive_thresh, maskA)
        self.img_scaled_regionB = cv.bitwise_and(self.img_scaled_adaptive_thresh, maskB)
        self.img_scaled_regionC = cv.bitwise_and(self.img_scaled_adaptive_thresh, maskC)

        regionA_cont, _ = cv.findContours(self.img_scaled_regionA, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        regionA_cont = sorted(regionA_cont, key=cv.contourArea, reverse=True)[:100]

        regionB_cont, _ = cv.findContours(self.img_scaled_regionB, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        regionB_cont = sorted(regionB_cont, key=cv.contourArea, reverse=True)[:100]
        
        regionC_cont, _ = cv.findContours(self.img_scaled_regionC, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        regionC_cont = sorted(regionC_cont, key=cv.contourArea, reverse=True)[:50]

        regionA_point = list(map(cont_centre_point, regionA_cont))
        regionA_point = np.array(regionA_point)
        regionA_point = order_points(regionA_point)

        regionB_point = list(map(cont_centre_point, regionB_cont))
        regionB_point = np.array(regionB_point)
        regionB_point = order_points(regionB_point)

        regionC_point = list(map(cont_centre_point, regionC_cont))
        regionC_point = np.array(regionC_point)
        regionC_point = order_points(regionC_point)

        choice_points = np.concatenate((regionA_point, regionB_point, regionC_point))
        print(choice_points)
        # print(order_points(choice_points))
        # choice_points.reshape((-1,5,2))


        for i, p in enumerate(choice_points.reshape((-1,2))):
            px, py = tuple(p)
            cv.putText(self.img_scaled, str(i), (px-10,py+10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        
        self._choice_points = choice_points
        choice_boxes = np.array(
            list(map(lambda p: points_to_box(p, 10), choice_points.reshape((-1,2))))
        )
        choice_intensity = []
        for c in choice_boxes:
            m = np.zeros(self.img_scaled.shape[:2], np.uint8)
            cv.fillPoly(m, [c], (255,255,255))
            intensity, _, _, _ = cv.mean(self.img_scaled_foobar, mask=m)
            choice_intensity.append(intensity)

        self.choice_intensity = np.array(choice_intensity)
        self.contours = choice_boxes

        cv.drawContours(self.img_scaled, choice_boxes, -1, self._BGR_RED, 1)
        # cv.drawContours(self.img_scaled, regionC_cont, -1, self._BGR_RED, 1)
        cv.imwrite('out/foo.jpg', self.img_scaled_regionC)
        # regionB_rect = cv.minAreaRect(regionB)
        # regionC_rect = cv.minAreaRect(regionC)
        # print(regionA)
        # print(regionA_rect)

        # cv.fillPoly(self.img_scaled, [regionA_rect], self._BGR_RED)

        cv.drawContours(self.img_scaled, sorted_a[:8], -1, self._BGR_RED, 1)

            # radius = int(radius)
            # center = (int(x),int(y))
            # cv.circle(self.img_scaled,center,10, self._BGR_RED,2)
            # cv.putText(self.img_scaled, str(i), center, cv.FONT_HERSHEY_SIMPLEX, 1, self._BGR_RED, 2)


        # cv.drawContours(self.img_scaled, conts, -1, self._BGR_RED, 1)

        # self.img_scaled_dilated = cv.dilate(self.img_scaled_edged, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

    def get_choice_boxes(self):
        return self._choice_boxes
    
    def get_intensities(self):
        return self.choice_intensity
    
    def get_choice_points(self):
        return self._choice_points

    def _write_steps(self):
        steps: dict['str', Any] =  {
            'original': self.img,
            'warped': self.img_warped,
            'grayscaled': self.img_grayscaled,
            'edged': self.img_edged, 
            'dilated': self.img_dilated,
            'marked_points': self.img_marked_points,
            'scaled': self.img_scaled,
            'scaled_adaptive_thresh': self.img_scaled_adaptive_thresh,
            'scaled_blured': self.img_scaled_blured,
            'scaled_edged': self.img_scaled_edged,
            'foobar': self.img_scaled_foobar,
            # 'scaled_dilated': self.img_scaled_dilated,
            # 'scaled_dilated': self.img_scaled_dilated,

            # # 'threshed': self.img_thresh,
            # 'blured': self.img_blured, 
            'marked_borders': self.img_marked_borders,
        }

        if not os.path.exists('out/'):
            os.mkdir('out')

        for i, item, in enumerate(steps.items()):
            k,v = item
            cv.imwrite(f'out/{i}_{k}.jpg', v)