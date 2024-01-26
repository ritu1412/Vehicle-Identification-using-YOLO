# Vehicle Detection using YOLO

## Overview

This project is part of the Udacity Self-Driving Car Nanodegree program. The goal is to accurately detect vehicles in a dash camera video using the YOLO (You Only Look Once) object detection algorithm. The implementation is detailed in the Jupyter notebook `vehicle_detection.ipynb` and demonstrates the ability to process video at a rate of 21 frames per second without batch processing. The output video can be viewed [here on YouTube](https://www.youtube.com/watch?v=PncSIx8AHTs).

Below, we discuss the pipeline and the steps involved in the vehicle detection process.

## Object Detection

Vehicle detection in a video stream is a form of object detection, which can be treated as a classification or a regression problem. In classification, the image is segmented into patches that are individually classified, whereas in regression, a neural network directly generates bounding boxes for objects.


- Classification: Divides image into patches, classifies each patch, and then draws bounding boxes around highly probable objects.

- Regression: The entire image is processed by a CNN to directly output bounding box coordinates.                         |

For this project, we utilize the tiny-YOLO v1 due to its ease of implementation and reasonable processing speed.

## Tiny-YOLO v1 Architecture

Tiny YOLO v1 consists of 9 convolutional layers and 3 fully connected layers. The initial layers act as feature extractors, while the last layers function as a regression head for bounding box predictions.
The model comprises 45,089,374 parameters, with the architecture detailed in the table included in `Vehicle_identification.ipynb`.

We employ Keras to construct the YOLO model.

## Postprocessing

The network outputs a 1470 vector, containing probabilities, confidences, and box coordinates, segmented into 49 regions (7x7 grid). We parse this vector to generate bounding boxes with probabilities exceeding a set threshold.

The weight loading and postprocessing functions are available in the `utili` class:

```python
load_weights(model, './yolo-tiny.weights')


