import numpy as np
import cv2
import dlib

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
            normal = 0, left = 1, right = 2, up = 3
    """
    x_ratio = (min_max[0][0] - cx) / (cx - min_max[1][0])
    y_ratio = (cy - min_max[0][1]) / (min_max[1][1] - cy)

    if x_ratio > 3:
        return 1
    elif x_ratio < 0.33:
        return 2
    elif y_ratio < 0.33:
        return 3
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
            normal = 0, left = 1, right = 2, up = 3
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
            1: 'Left',
            2: 'Right',
            3: 'Up',
        }
    if left != 0:
        text = "Left: " + directions[left]
        cv2.putText(img, text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
    if right != 0:
        text = "Right: " + directions[right]
        cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    kernel = np.ones((9, 9), np.uint8)

    left = [36, 37, 38, 39, 40, 41]
    right = [42, 43, 44, 45, 46, 47]

    cap = cv2.VideoCapture(0) # initialize video capture

    while True:
        ret, img = cap.read()
        if not ret:
            continue

        thresh = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
        rects = detector(gray, 1) # rects contain all the faces detected

        # use histogram equalization to improve contrast
        equal = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe = clahe.apply(equal)

        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            mask, left_min_max = eye_on_mask(mask, left, shape)
            mask, right_min_max = eye_on_mask(mask, right, shape)
            mask = cv2.dilate(mask, kernel, 5)
            eyes = cv2.bitwise_and(img, img, mask=mask)
            mask = (eyes == [0, 0, 0]).all(axis=2)
            eyes[mask] = [255, 255, 255]
            mid = (shape[42][0] + shape[39][0]) >> 1
            eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

            # convert the equalized grayscale image to binary image
            threshold = np.median(eyes_gray[eyes_gray != 255])
            cv2.putText(img, str(threshold), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
            thresh = process_thresh(thresh)

            left_pos = contouring(thresh[:, 0:mid], mid, img, left_min_max)
            right_pos = contouring(thresh[:, mid:], mid, img, right_min_max, True)
            print_eye_pos(img, left_pos, right_pos)

            # for (x, y) in shape:
                # cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        cv2.imshow("Result", img)
        cv2.imshow("Gray", eyes_gray)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("CLAHE", clahe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

