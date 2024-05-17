# Structure from Motion (SFM)

An implementation of a Structure from Motion (SFM) pipeline for 3D reconstruction based on the paper *Building Rome in a Day* ([linked here](http://grail.cs.washington.edu/rome/rome_paper.pdf)). Obtained results from running this pipeline on an Nvidia Jeston Orin Nano are compated against point-cloud information gathered using a ZED stereo camera.


## Authors
<ul>
<li> Lowell Lobo (lorocks@umd.edu)
<li> Vikram Setty (vikrams@umd.edu)
<li> Vinay Lanka (vlanka@umd.edu)
<li> Apoorv Thapliyal (apoorv10@umd.edu)
</ul>

## Image Data
A collection of images are stored within the /data/images directory, all of which can be used as input to the SFM pipeline.


## Code Files
There are three files that can be executed to begin the SFM pipeline,
1. main.py - SFM pipeline that is built purely using custom functions written from scratch
2. main_cv.py - SFM pipeline written using some cv2 functions to reduce execution time
3. main_other.py - SFM pipeline written using a combination of cv2 and custom functions

The main SFM pipeline implementation can be found in main_cv.py.

main.py and main_other.py are different implementation of the SFM pipeline.

## Using Different Databases
To link a specific image for inferencing, the following code lines wille need to be edited,
1. main.py - line number 18
2. main_cv.py - line number 200
3. main_other.py - line number 155

For example, to use the monument images, diretory will be

```bash
    "../data/images/monument/"
```

The intrinsic matrix will also need to be editied and can be found within main_cv.py from lines 64 - 95

## Executing Code
Python file should be executed within the /scripts directory for proper working