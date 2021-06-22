import cv2
import dlib
import numpy as np

KERNEL = np.ones((9, 9), np.uint8)

def shape_to_np(shape, dtype="int"):
    """
    Convert dlib detected object into a numpy array

    Parameters:
        shape (dlib.full_object_detection): dlib detected object to convert

    Returns:
        coords (np.ndarray): coordinates of each facial landmark
    """
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # iterate and convert facial landmarks to coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return list of coordinates
    return coords

def eye_on_mask(mask, side, shape):
    """
    Create ROI on mask of the size of the eyes and also return the extreme points

    Parameters:
        mask (np.uint8): mask to draw eyes on
        side (list[int]): facial landmark numbers of eye
        shape (list[uiint32]): facial landmarks

    Returns:
        mask (np.uint8): mask with ROI drawn
        min_max (list[tuple]): top left and bottom right coordinates of ROI's AABB
    """
    # create mask for eye ROI
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)

    # find minimum and maximum coordinate locations of eye
    min_x = points[0][0]
    min_y = (points[1][1] + points[2][1]) >> 1
    max_x = points[3][0]
    max_y = (points[4][1] + points[5][1]) >> 1
    min_max = [(min_x, min_y), (max_x, max_y)]

    return mask, min_max

def process_mask(img, left, right, shape):
    """
    Mask image such that only the detected eyes are visible, return bounding boxes of eyes

    Parameters:
        img (np.ndarray): original image
        left (list[int]): facial landmark numbers of left eye
        right (list[int]): facial landmark numbers of right eye
        shape (list[uint32]): facial landmarks

    Returns:
        gray (np.ndarray): processed masked image
        left_min_max (list[tuple]): top left and bottom right coorindates of left eye's AABB
        right_min_max (list[tuple]): top left and bottom right coorindates of right eye's AABB
    """
    # mask image such that only eye ROIs are visible
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask, left_min_max = eye_on_mask(mask, left, shape)
    mask, right_min_max = eye_on_mask(mask, right, shape)
    mask = cv2.dilate(mask, KERNEL, 5)
    eyes = cv2.bitwise_and(img, img, mask=mask)
    mask = (eyes == [0, 0, 0]).all(axis=2)
    eyes[mask] = [255, 255, 255]

    # convert image to grayscale
    gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

    # use histogram equalization to improve contrast of eyes
    equal = cv2.equalizeHist(gray[gray != 255])
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe = clahe.apply(equal)
    gray[gray != 255] = clahe.flatten()

    return gray, left_min_max, right_min_max


def process_thresh(thresh):
    """
    Preprocess threshold image

    Parameters:
        thresh (np.ndarray): thresholded image to preprocess

    Returns:
        thresh (np.ndarray): processed thresholded image
    """
    # apply erosion, dilation, median blur, and bitwise not
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)

    return thresh

def find_eyeball_position(min_max, cx, cy):
    """
    Find and return the position of the eyeball (normal, left, right, or top)

    Parameters:
        min_max (list[tuple]): top left and bottom right coordinates of ROI's AABB
        cx (int): detected x coordinate of pupil
        cy (int): detected y coordinate of pupil

    Returns:
        pos (int): the position of the eyeball
            normal = 0, left = -1, right = 1
    """
    x_ratio = (min_max[0][0] - cx) / (cx - min_max[1][0])

    # TODO: more testing on these thresholds to determine direction
    if x_ratio > 2.5:
        return -1
    elif x_ratio < 0.33:
        return 1
    else:
        return 0


def contouring(thresh, mid, img, min_max, right=False):
    """
    Find the largest contour of an image divided by a midpoint and find the eye position

    Parameters:
        thresh (np.ndarray): thresholded image of one side containing the eyeball
        mid (int): midpoint between the eyes
        img (np.ndarray): original image
        min_max (list[tuple]): top left and bottom right coordinates of ROI's AABB
        right (boolean, optional): whether calculating for the right eye or left eye.
            defaults to false

    Returns:
        pos (int): the position of the eyeball
    """
    _, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        pos = find_eyeball_position(min_max, cx, cy)
        return pos
    except:
        return 0

def print_eye_pos(img, left, right):
    """
    Print where the eyes are looking and display on the image

    Parameters:
        img (np.ndarray): image to display on
        left (int): position obtained from left eye
        right (int): position obtained from right eye

    Returns:
        None
    """
    directions = {
        -1: 'Left',
        1: 'Right',
    }
    if left != 0:
        text = "Left: " + directions[left]
        cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    if right != 0:
        text = "Right: " + directions[right]
        cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

class Webcam:
    def __init__(self, display=False):
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self._left = [36, 37, 38, 39, 40, 41]
        self._right = [42, 43, 44, 45, 46, 47]

        self._cap = cv2.VideoCapture(0) # initialize video capture

        self.display = display

    def get_eye_pos(self):
        ret, img = self._cap.read()
        if not ret:
            return 0, 0

        thresh = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
        rects = self._detector(gray, 1) # rects contain all the faces detected
        left_pos, right_pos = 0, 0
        
        # TODO: find the rect closest to the center of the image, use that only
        for rect in rects:
            # get shape
            shape = shape_to_np(self._predictor(gray, rect))

            # get masked grayscale image and bounding boxes for each eye
            mask, left_min_max, right_min_max = process_mask(img, self._left, self._right, shape)

            # convert the equalized grayscale image to binary image
            _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            thresh = process_thresh(thresh)

            # calculate midpoint and get left and right eye position
            mid = (shape[42][0] + shape[39][0]) >> 1
            left_pos = contouring(thresh[:, 0:mid], mid, img, left_min_max)
            right_pos = contouring(thresh[:, mid:], mid, img, right_min_max, True)

        if self.display:
            print_eye_pos(img, left_pos, right_pos)
            cv2.imshow("Result", img)
            
        return left_pos, right_pos


def test():
    wc = Webcam()
    while True:
        print(wc.get_eye_pos())
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    test()
