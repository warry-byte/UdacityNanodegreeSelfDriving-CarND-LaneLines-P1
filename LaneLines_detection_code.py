import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def get_img_ROI(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image in place (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lane_lines(img, lines, color=[255, 0, 0], thickness=10, slope_threshold_px=0.02, y_min=0, y_max=100):
    """
    Draws and connects lines with a similar slope. It aggregates lines which slope is within +[0,slope_threshold_px]
    from the current slope.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image in place (mutates the image).

    """
    # Clean up data
    y_min = int(y_min)
    y_max = int(y_max)
    lines = np.squeeze(lines)  # make proper matrix out of lines ndarray

    # check if lines is not empty (could indicate a blank frame) AND there is strictly more than 1 line detected
    if(lines.item(0) != None and np.size(lines.shape) > 1):

        # computing each of the lines slopes
        slopes = (lines[:, 3] - lines[:, 1]) / (lines[:, 2] - lines[:, 0])  # deltaY / deltaX

        # sort lines by slope - this is mandatory for the following to work
        lines_sorted = lines[slopes.argsort()]
        slopes.sort()

        origins = lines_sorted[:, 3] - np.multiply(lines_sorted[:, 2],
                                                   slopes)  # element wise mult to get y2 - m.x2, with m = slope

        # Right and left lanes to be detected - format will be x1, y1, x2, y2, slope, origin
        left_lane = []
        right_lane = []
        aggregated_lines = []

        search_for_lines = True
        next_slope_min_index = 0
        # current slope to be aggregated is pointed to by the indexes of the items lower than (min + threshold)
        next_slope_min_indexes = np.where(slopes < slopes[next_slope_min_index] + slope_threshold_px)
        last_slope = 0

        while (search_for_lines):

            # find similar lines according to slope threshold
            match_indices = (slopes >= slopes[next_slope_min_index]) & (
                    slopes < slopes[next_slope_min_index] + slope_threshold_px)

            av_slope = np.average(slopes[match_indices])
            av_origin = np.average(origins[match_indices])

            # solving line eq to find the x's
            # horizontal lines
            if(av_slope == 0):
                x1 = int(np.average(lines_sorted[match_indices, 0]))
                x2 = int(np.average(lines_sorted[match_indices, 2]))
                y1 = np.min(lines_sorted[match_indices, 1])
                y2 = y1 # arbitrary - they area almost equal anyway...
            elif(av_slope == np.inf or av_slope == -np.inf):
                x1 = 0
                x2 = 2000 # any big value
                y1 = np.average(lines_sorted[match_indices, 1])
                y2 = np.average(lines_sorted[match_indices, 3])
            else: # normal case: right and left lanes detected
                x1 = int((y_min - av_origin) / av_slope)
                x2 = int((y_max - av_origin) / av_slope)
                y1 = y_min
                y2 = y_max

            # draw new line and store
            aggregated_lines.append([x1, y1, x2, y2, av_slope, av_origin])

            if (next_slope_min_indexes[0][-1] == np.size(slopes) - 1):  # the line we reached is the last one to be aggregated
                search_for_lines = False  # this will break the loop after drawing this last line
            else:
                # move index pointing to the new min in the slope array
                next_slope_min_index = next_slope_min_indexes[0][-1] + 1
                next_slope_min_indexes = np.where((slopes >= slopes[next_slope_min_index]) & (
                        slopes < slopes[next_slope_min_index] + slope_threshold_px))
                last_slope = av_slope # storing last aggreg line slope to detect sign change

        # finding the left and right lane: as the slopes were sorted, and as the closest lanes to the POV have max. slope, we can simply choose the first line accumulated (most negative slope) and the last one (most positive) as the left / right lanes, resp.
        left_lane = aggregated_lines[0]
        right_lane = aggregated_lines[-1]
        cv2.line(img, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), color, thickness)
        cv2.line(img, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, y_min = 0, y_max = 100):
    """
    `img` should be the output of a Canny transform.

    Returns an image with probabilistic hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    draw_lane_lines(line_img, lines, y_min = y_min, y_max = y_max)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

### TESTING PIPELINE
# src_img = mpimg.imread('test_images/challenge.jpg')
# # src_img = mpimg.imread('test_images/whiteCarLaneSwitch.jpg')
# gray_img = grayscale(src_img)
#
# kernel_size = 5
# blur_gray_img = gaussian_blur(gray_img, kernel_size)
#
# low_threshold = 50
# high_threshold = 150
# edge_img = cv2.Canny(blur_gray_img, low_threshold, high_threshold)
#
# BL = (0,edge_img.shape[0]) # bottom left
# BR = (edge_img.shape[1], edge_img.shape[0])
# TR = (700, 350)
# TL = (300, 350)
# ROI_vertices = np.array([[BL, BR, TR, TL]])
# masked_img = get_img_ROI(edge_img, ROI_vertices)
# # At this point, we will ignore the small lines
# # Those will be filtered out later
#
# # test result
# plt.imshow(masked_img, cmap = 'gray')
# plt.show()
#
# # Define the Hough transform parameters
# rho_reso_px = 1 # distance resolution in pixels of the Hough grid
# theta_reso_rad = np.pi/180 # angular resolution in radians of the Hough grid
#
# # Produce line image with the final parameters
# hough_vote_threshold = 30
# min_line_length = 10
# max_line_gap = 100
#
# aggreg_line_img = hough_lines(masked_img,
#                               rho_reso_px,
#                               theta_reso_rad,
#                               hough_vote_threshold,
#                               min_line_length,
#                               max_line_gap,
#                               y_min = TL[1],
#                               y_max = BL[1])
#
# # test: insert lines in a table with slopes etc
# # print(line_img)
#
# # Create a "color" binary image to combine with line image
# # color_edges = np.dstack((edges, edges, edges))
# #
# # # Draw the lines on the edge image
# # lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)
# final_img = weighted_img(aggreg_line_img, src_img)
# plt.imshow(final_img)
# plt.show()

### FINAL PIPELINE VIDEO
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    gray_img = grayscale(image)

    kernel_size = 5
    blur_gray_img = gaussian_blur(gray_img, kernel_size)

    #     low_threshold = 50
    #     high_threshold = 150
    low_threshold = 100
    high_threshold = 400
    edge_img = cv2.Canny(blur_gray_img, low_threshold, high_threshold)

    BL = (100,                         edge_img.shape[0] - 80) # bottom left
    BR = (edge_img.shape[1] - 100,     edge_img.shape[0])
    TR = (int(edge_img.shape[1] / 2 + 100), int(edge_img.shape[0] / 2 + 100))
    TL = (int(edge_img.shape[1] / 2 - 100), int(edge_img.shape[0] / 2 + 100))
    ROI_vertices = np.array([[BL, BR, TR, TL]])
    masked_img = get_img_ROI(edge_img, ROI_vertices)
    # At this point, we will ignore the small lines
    # Those will be filtered out later

    # debug
    # plt.imshow(masked_img)

    # Define the Hough transform parameters
    rho_reso_px = 1 # distance resolution in pixels of the Hough grid
    theta_reso_rad = np.pi/180 # angular resolution in radians of the Hough grid

    # Produce line image with the final parameters
    hough_vote_threshold = 30
    min_line_length = 30
    max_line_gap = 110

    aggreg_line_img = hough_lines(masked_img,
                                  rho_reso_px,
                                  theta_reso_rad,
                                  hough_vote_threshold,
                                  min_line_length,
                                  max_line_gap,
                                  y_min = TL[1],
                                  y_max = image.shape[0]-1)

    # Create a "color" binary image to combine with line image
    final_img = weighted_img(aggreg_line_img, image)

    return final_img

white_output = 'test_videos_output/challenge_accepted.mp4'
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/challenge.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)