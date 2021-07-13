import cv2
import numpy as np

import json

CANNY_LOW = 15
CANNY_HIGH = 300

# save a set of colours to file
def save_colors(colors, name):
    d = [c.tolist() for c in colors]
    with open(f'{name}.json', 'w') as fp:
        json.dump(d, fp)

# load set of colours from file
def load_colors(name):
    with open(f'{name}.json') as fp:
        data = json.load(fp)
    return [np.array(c) for c in data]

# create colour boundaries for 
def create_color_boundaries(colors):
    boundaries = [(c, c) for c in colors]
    return boundaries

# draw filled (by color) rectangles with identifiable edges 
def draw_bounding_rectangles(image, color, canny_low=CANNY_LOW, canny_high=CANNY_HIGH):

    edges = cv2.Canny(image.astype('uint8'), canny_low, canny_high)
    edges = cv2.dilate(edges, None)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    # if no contours then set single area to 0
    rectAreas = [0]*(max(1,len(contours)))
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 1, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        rectAreas[i] = abs(int(boundRect[i][2] * boundRect[i][3]))

    drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)

    for i in range(len(contours)):
        cv2.drawContours(drawing, contours_poly, i, color)
        cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
          (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, -1)
    return drawing, rectAreas

# create a mask of areas in image where the colour is inside the 
# supplied colour boundaries 
def mask_by_color(image, boundaries):
    accumMask = np.zeros(image.shape[:2], dtype="uint8")
    for (lower, upper) in boundaries:
        # find the colors within the specified boundaries
        mask = cv2.inRange(image, lower, upper)
        # merge the mask into the accumulated masks    
        accumMask = cv2.bitwise_or(accumMask, mask)

    return cv2.bitwise_or(image, image, mask=accumMask) 

# create an image with filled rectangles 
# where the images contains colours that are within 
# the range identified by boundaries
def rectangles_color(image, 
                     boundaries, 
                     rectangle_color,
                     canny_low=CANNY_LOW,
                     canny_high=CANNY_HIGH):
    masked = mask_by_color(image, boundaries)
    return draw_bounding_rectangles(masked, 
                                    rectangle_color,
                                    canny_low=canny_low,
                                    canny_high=canny_high)


# No longer used - attempt to exclude background rather than extract the bananas
def create_mask_exclude_background(img):
    accumMask = np.zeros(img.shape[:2], dtype="uint8")

    boundaries = [
    ([95,    70,    27],[105,82,50]),
    ([100,83,70],[109,109,117]),
    ([205,197,84],[205,197,84]),
    ([199, 192, 184], [199, 192, 184]),
    ([166, 150, 125], [230, 200, 180]),
    ([110, 90, 60], [160, 160, 160]),
    ([210, 201, 171], [241, 246, 242]),
    ([180,163,140],[236,228,210]),
    ([47,55,45], [98,98, 110])
    ]

    # loop over the boundaries
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype="uint8")/255
        upper = np.array(upper, dtype="uint8")/255

        # find the colors within the specified boundaries
        mask = cv2.inRange(img, lower, upper)

        # merge the mask into the accumulated masks
        accumMask = cv2.bitwise_or(accumMask, mask)
    return accumMask

def process_image(img, 
                  colour1_boundaries=([120,145,50], [130,255,130]), 
                  colour2_boundaries=([18, 176, 235], [22, 227, 255]),
                  canny_low=CANNY_LOW,
                  canny_high=CANNY_HIGH):
    img = (img * 255).astype('uint8')
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # fill colour
    blue_color = (25, 100,  100)

    blue_boundaries = [(np.array(colour1_boundaries[0],np.uint8), 
                        np.array(colour1_boundaries[1],np.uint8))]
    
    # image with rectangles for blue bananas
    blue_rectangles, blueRectAreas = rectangles_color(hsv, 
                                                      blue_boundaries, 
                                                      blue_color,
                                                      canny_low=canny_low,
                                                      canny_high=canny_high)
    # fill colour
    yellow_color = (130, 255,  255)

    yellow_boundaries = [(np.array(colour2_boundaries[0],np.uint8), 
                        np.array(colour2_boundaries[1],np.uint8))]
    # image with rectangles for yellow bananas
    yellow_rectangles, yellowRectAreas = rectangles_color(hsv, yellow_boundaries, yellow_color)

    # combine images with rectangles for each colour
    # also return the area of the biggest rectangle for each colour
    combined = cv2.bitwise_or(yellow_rectangles, blue_rectangles) 
    return combined/255, np.max(yellowRectAreas), np.max(blueRectAreas) 
