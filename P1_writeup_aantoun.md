# **Finding Lane Lines on the Road** 
---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

The presented pipeline consists of the following steps:

* Conversion from RGB to grescale
* Gaussian blur (prepare for edge detection)
* Canny edge detection (detect gradients in the image)
* ROI selection (polygon): the ROI selects the portion of the image that is most likely to contain the two lane lines
* Probabilistic Hough transform: detect lines in the edge image
* Select and display lane lines: in the line image, select the lines that most likely correspond to the right and left lane lines

To select and display the right and left lane lines, we use a simple heuristic that consists of selecting the lines with lowest slope (left) and highest slope (right).
In order to draw a single line on the left and right lanes, the following steps were followed inside the modified draw_lines() method:

- Pack the lines in an np.array object
- Compute slopes of all lines
- Sort lines according to their slopes, in ascending order (negative to positive)
- Compute origins of all lines

After these steps, the method aggregates the lines with similar slope, according to parameter __slope_threshold_px__. If the slopes differ more than this parameter, they will be aggregated separately in the final __aggregate_lines__ array. In this manner, the function loops and aggregates lines according to their similarity in terms of slope, until there is no more line to aggregate. 

Finally, the left lane and right lane are selected as those with lowest and highest slope (respectively) in the __aggregated_lines__ array.


### 2. Identify potential shortcomings with your current pipeline

Edge cases were not fully covered properly, for example when the lines have zero or infinite slope. The Canny edge detection parameters selected are very aggressive and could be lowered to select a bigger number of lines, while adjusting the parameters of the Hough transform to filter out remaining small lines.

Additionally, the proposed line detection pipeline parameters shows a significant amount of flickering in the final rendered file. 

### 3. Suggest possible improvements to your pipeline

The method draw_lane_lines() is an attempt at overcoming those issues, although not correcting them perfectly. The edge cases of the different challenges bring a number of numerical issues that are dealt with in the method, so that it does not crash but sometimes fails to accomplish its function. This part could be largely improved in production code. More time could have been allocated to tweak the different parameters of the pipeline to cope with the different images (i.e. the "challenge" parameters are not usable for the Yellow line challenge). Therefore, a thorough investigation and proper optimization of the different parameters could be required to cover a wider range of cases and conditions (brightness, road condition, lane change, etc).

Such an optimization would also reduce the flickering effect of the detection pipeline in its current condition.

No real-time aspects were considered in this project. This aspect is an important requirement in the potential productization of the solution.

Finally, the method could be extended to detect lanes via linear segments in the road lines rather than lines, therefore allowing to detect the curvature of the road lane lines and extend the detection range.
