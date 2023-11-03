import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from rotate.rotate import rotate
class OrientationDetection:
    def __init__(self, image):
        # self.image = cv2.imread(image)
        self.image = image
        self.orientation = None
        self.bbox = None
        self.min_non_zero_ratio = 0.45

    def text_detection(self):

        rgb = cv2.pyrDown(self.image)
        small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

        _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        mask = np.zeros(bw.shape, dtype=np.uint8)

        for idx in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[idx])

            mask[y:y + h, x:x + w] = 0
            cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
            r = float(cv2.countNonZero(mask[y:y + h, x:x + w])) / (w * h)
            cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

            # vertical text
            if r > self.min_non_zero_ratio:

                cv2.rectangle(rgb, (x, y), (x + w - 1, y + h - 1), (0, 255, 0), 2)

                if h > w and 200 >= h > 50:
                    self.orientation = 'vertical'
                    self.bbox = rgb[y:y + h, x:x + w]
                    break

                elif h < w and 200 >= w > 50:
                    self.orientation = 'horizontal'
                    self.bbox = rgb[y:y + h, x:x + w]
                    break

            else:
                pass

        if self.bbox is not None:
            cv2_imshow(self.bbox)
        print(self.orientation)
        return self.bbox

    def left_right_rotation(self):
        if self.orientation == 'vertical':
            self.image = rotate(self.image, 90, (0, 0, 0))
            self.text_detection()
            print("done")

    def up_down_rotate(self):
        if self.orientation == 'horizontal':

            bbox = self.bbox.copy()
            gray = cv2.cvtColor(bbox, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)[1]
            cv2_imshow(thresh)

            x, y, w, h = 0, 0, bbox.shape[1], bbox.shape[0]

            top_half = ((x, y), (x + w, y + h / 2))
            bottom_half = ((x, y + h / 2), (x + w, y + h))

            top_x1, top_y1 = top_half[0]
            top_x2, top_y2 = top_half[1]
            bottom_x1, bottom_y1 = bottom_half[0]
            bottom_x2, bottom_y2 = bottom_half[1]

            # Split into top/bottom ROIs
            top_image = thresh[int(top_y1):int(top_y2), int(top_x1):int(top_x2)]
            bottom_image = thresh[int(bottom_y1):int(bottom_y2), int(bottom_x1):int(bottom_x2)]

            cv2_imshow(top_image)
            cv2_imshow(bottom_image)

            # Count non-zero array elements
            top_pixels = cv2.countNonZero(top_image)
            bottom_pixels = cv2.countNonZero(bottom_image)

            # print('top', top_pixels)
            # print('bottom', bottom_pixels)

            # Rotate if upside down
            if top_pixels > bottom_pixels:
                rotated = rotate(self.image, 180, (0, 0, 0))
                print("rotated")
                return rotated
            else:
                print("original_image")
                return self.image

        return self.image