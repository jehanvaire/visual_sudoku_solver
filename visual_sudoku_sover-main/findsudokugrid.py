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


def getGridLines(image):
    #turn the new image into grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #will create a black and white image
    edges = cv2.Canny(gray,90,150,apertureSize = 3)

    #will get the lines from the image
    lines = cv2.HoughLines(edges,1,np.pi/180,150)

    if not lines.any():
        print('No lines were found')
        exit()

    if filter:
        rho_threshold = 15
        theta_threshold = 0.1

        # how many lines are similar to a given one
        similar_lines = {i : [] for i in range(len(lines))}
        for i in range(len(lines)):
            for j in range(len(lines)):
                if i == j:
                    continue

                rho_i,theta_i = lines[i][0]
                rho_j,theta_j = lines[j][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    similar_lines[i].append(j)

        # ordering the INDECES of the lines by how many are similar to them
        indices = [i for i in range(len(lines))]
        indices.sort(key=lambda x : len(similar_lines[x]))

        # line flags is the base for the filtering
        line_flags = len(lines)*[True]
        for i in range(len(lines) - 1):
            if not line_flags[indices[i]]: # if we already disregarded the ith element in the ordered list then we don't care (we will not delete anything based on it and we will never reconsider using this line again)
                continue

            for j in range(i + 1, len(lines)): # we are only considering those elements that had less similar line
                if not line_flags[indices[j]]: # and only if we have not disregarded them already
                    continue

                rho_i,theta_i = lines[indices[i]][0]
                rho_j,theta_j = lines[indices[j]][0]
                if abs(rho_i - rho_j) < rho_threshold and abs(theta_i - theta_j) < theta_threshold:
                    line_flags[indices[j]] = False # if it is similar and have not been disregarded yet then drop it now

    print('number of Hough lines:', len(lines))

    filtered_lines = []

    if filter:
        for i in range(len(lines)): # filtering
            if line_flags[i]:
                filtered_lines.append(lines[i])

        print('Number of filtered lines:', len(filtered_lines))
    else:
        filtered_lines = lines

    for line in filtered_lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        #put the lines on the image
        cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)

    #then return the image
    return image

#return each corner location in a numpy array
def findCellLocation(image, x, y):
    #the step is a will help to find a location
    step_x = image.shape[1] // 9
    step_y = image.shape[0] // 9

    top_r, top_l, bottom_r, bottom_l = [[[x*step_x + step_x, y*step_y]], [[x*step_x, y*step_y]], [[x*step_x + step_x, y*step_y + step_y]], [[x*step_x, y*step_y + step_y]]]

    corners = np.array([top_l, bottom_l, bottom_r, top_r])
    return corners

#will crop the image to 28*28, on its center
def cropTo28(cell):
    diff_hauteur = cell.shape[0] - 28
    diff_largeur = cell.shape[1] - 28
    crop = cell[int(diff_hauteur/2):28+int(diff_hauteur/2), int(diff_largeur/2):28+int(diff_largeur/2)]
    return crop