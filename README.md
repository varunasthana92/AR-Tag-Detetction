## AR-Tag Detection and replacement with another image and a virtual 3D cube.

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

