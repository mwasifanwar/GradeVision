# GradeVision

GradeVision is an automated grading tool that utilizes OpenCV (Open Source Computer Vision Library) to analyze and evaluate handwritten or printed documents. This tool is particularly useful for grading multiple-choice tests, surveys, or any document where marking regions need to be identified and assessed automatically.

## Installation

1. First Clone this repository to your local machine.
2. To use GradeVision, you need to have Python installed on your system along with the OpenCV library. You can install OpenCV using pip:

```
pip install opencv-python
```

## Usage

1. First, ensure that your document images are placed in the `images` directory within the GradeVision repository.

2. Open the `main.py` script in your preferred Python IDE or text editor.

3. Customize the script according to your document's requirements, such as adjusting image dimensions and file paths.

4. Run the `main.py` script. It will automatically process the images, detect marking regions, and output the results.

## Understanding the Code

The `main.py` script performs the following tasks:

1. **Image Pre-Processing**: The input image undergoes pre-processing steps such as converting to grayscale, applying Gaussian blur, and detecting edges using the Canny edge detector.

2. **Finding Contours**: Contours are detected on the edge-detected image using the `findContours` function from OpenCV. These contours represent the marking regions on the document.

3. **Finding Rectangles**: The script identifies rectangular contours, assuming that they represent the marking areas on the document.

4. **Perspective Transformation**: Perspective transformation is applied to each rectangular region to obtain a bird's-eye view, making it easier to analyze the marking regions.

5. **Thresholding and Analysis**: Each transformed image is thresholded to separate marked and unmarked regions. The script then analyzes these regions to determine which options are marked.

6. **Output Visualization**: The script generates visualizations of the marking regions and the detected marks for verification and analysis.

## Dependencies

- Python 3.x
- OpenCV (cv2)
- NumPy

## Contributing

Contributions to GradeVision are welcome! If you have any ideas for improvements or new features, feel free to open an issue or submit a pull request.
