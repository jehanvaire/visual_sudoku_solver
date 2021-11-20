import cv2
import numpy as np

def getGridContour(image):
    #turn the image into gray scales
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #blur the image
    blur = cv2.medianBlur(gray, 3)
    #create a thresh : will turn the image in black or white values
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)

    #find the contour of the grid in the image
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        #gets the top down view of the grid
        transformed = getTopDownView(image, approx)
        break
    return transformed


def getTopDownView(image, corners):
    def order_corners(corners):
        # Separate corners into individual points
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_l, bottom_l, bottom_r, top_r  = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corners(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


#return each corner location in a numpy array
def findCellLocation(image, y, x):
    #the step is a will help to find a location
    step_x = image.shape[1] // 9
    step_y = image.shape[0] // 9

    top_r, top_l, bottom_r, bottom_l = [[[x*step_x + step_x, y*step_y]], [[x*step_x, y*step_y]], [[x*step_x + step_x, y*step_y + step_y]], [[x*step_x, y*step_y + step_y]]]

    corners = np.array([top_l, bottom_l, bottom_r, top_r])
    return corners

