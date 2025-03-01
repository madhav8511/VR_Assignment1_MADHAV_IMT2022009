# Task-1 : Coin Detection and Segmentation

This project uses computer vision techniques to detect, segment, and count coins in an image. The process involves detecting key contours, filtering based on area , and then segmenting each detected coin into individual images.

### Dependencies

To run this project, you'll need the following Python libraries:

- **OpenCV**
- **NumPy**

Make sure all these are present there on your system or otherwise install them using command :

```bash
pip install opencv-python numpy
```

### To Run Program

- Make sure your image is present in the input directory of part-1 with correct naming convention.
- Go to part-1 directory and run the following command:
```bash
python3 coins.py
```
- It will output the outlined detected image and each individual segmented and cropped coins image in the output directory of part-1
- It will also output the no of coin in each image on terminal screen.

# Task-2 : Image Stitching with Keypoint Detection

This project uses computer vision techniques to stitch multiple overlapping images into a single panoramic image. The process involves detecting key points, computing descriptors, and using feature matching to align and blend the images into a seamless panorama.

### Dependencies

To run this project, you'll need the following Python libraries:

- **OpenCV**
- **NumPy**

Make sure all these are present there on your system or otherwise install them using command :

```bash
pip install opencv-python numpy
```

### How to Run Program
- Make sure your all images is present in input directory of part-2 with correct naming convention.
- Now go to part-2 and run the follwing command :
```bash
python3 panorama.py 
```
- Your final stiched image and individual image with keypoints marked are present in the output directory of part-2.
- Also the Keypoints matched image between two overlapped image are also present in output directory.