PREREQUISITES:
The code is tested on the following environment
python == 3.8
opencv-python == 4.9.0.80 [pip install opencv-python==4.9.0.80]
numpy == 1.19.5 [pip install numpy==1.19.5]


USAGE:
1. To run the solution:

python main.py [-h] [--input INPUT] [--output OUTPUT]

Finds the four non-overlapping 5x5 patches with highest average brightness, takes the patch centers as corners of a quadrilateral, calculates its area in pixels, and draws the
quadrilateral in red into the image and saves it in PNG format.

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Path of input image filename (default='./input.jpg')
  --output OUTPUT  Path to output image filename (default='./output.png')
  
2. To run test cases:
python tests/test_main.py


