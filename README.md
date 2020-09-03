## AR-Tag Detection and replacement with another image and a virtual 3D cube.

<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/tag_detect.gif">
</p>

### Pipeline
* AR-Tag detected using OpenCV findcontour() function. Homography computed to warp the detected tag to world reference.
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/detectedTag.png" wwidth=500, height = 300>>

<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/Intermediate/warped_detection.png" wwidth=500, height = 300><br>
Detected tag warped to world reference
</p>

* Detection output pre-processing for better result. Scaled the detected tag to fit the grid size as per the reference AR-tag.
<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/ref_marker_grid.png" ><br>
Reference AR-Tag
</p>

<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/Intermediate/warped_thresh.jpg" wwidth=985><br>
Thresholding to get binary image
</p>

<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/Intermediate/warped_thresh_scaled.png" wwidth=80%><br>
Scaled to match 4X4 grid size
</p>

* Orientation and tag-id was computed as per the encoding scheme provided, by analyzing each grid cell.

* Computed homograpghy between the detected tag and a reference tag. Using this, computed the camera projection matrix (with calibration matrix already available).

* Implemented a image warping function using the camera projection matrix to replace the tag with another image, and a 3D cube.
<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/Single_lena.png" width=500, height = 300>
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/virtualCube.jpeg" width=500, height = 300>
</p>


#### Dependencies

- python3
- OpenCV 4.1.1
- numpy


#### How to run 

tagDetection.py detects the tag and shows the ID of tag. You need to provide a video file in which tag is present. You can run this file using following command
```
$ cd AR-Tag-Detetction/Code
$ python3 tagDetection.py --Video="Provide path to your video here"
```
To exit the video, press Q.

tagLena.py places an image provided by you, on the AR tag. You need to provide a video which contains AR tags and an image which you want place on the tag. You can use following command to run this file.

```
$ cd AR-Tag-Detetction/Code
$ python3 tagLena.py --Video="Path to video file" --Image="Path to image"
```
To exit the video, press Q.

cube.py places a virtual cube on the AR tag. Camera calibration parameters are mentioned in the file itself. If you want to work with different cameras, please change the camera parameters. You need to specify a video file which contains AR tags and you can use following command to run this file.

```
$ cd AR-Tag-Detetction/Code
$ python3 cube.py --Video="Provide path to your video file"
```
To exit the video, press Q.

For more details, read the Report.pdf

### Output Images
Tag Detection-
<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/detectedTag.png">
</p>

Tag Replacement with an image
<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/Single_lena.png">
</p>

<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/maultiple_lena.jpeg">
</p>

Tag Replacement with a virtual 3D cube
<p align="center">
<img src="https://github.com/varunasthana92/AR-Tag-Detetction/blob/master/images/virtualCube.jpeg">
</p>

